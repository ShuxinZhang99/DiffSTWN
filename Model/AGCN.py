import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AdaptiveFullODGraph(nn.Module):
    """
    基于完整起点站和终点站生成两张自适应图。

    参数
    ----------
    num_origin:
        完整起点数量，例如 station_num=198。

    num_destination:
        完整终点数量，例如 station_num=198。

    embedding_dim:
        自适应图节点嵌入维度。

    top_k:
        每个节点保留的最强邻居数量。
        None 表示保留全部连接。

    temperature:
        Softmax 温度。
        数值越小，邻接权重越集中。
    """

    def __init__(
        self,
        num_origin,
        num_destination,
        embedding_dim=16,
        top_k=None,
        temperature=1.0,
        include_self_loop=True,
    ):
        super().__init__()

        self.num_origin = num_origin
        self.num_destination = num_destination
        self.num_od_nodes = num_origin * num_destination

        self.embedding_dim = embedding_dim
        self.top_k = top_k
        self.temperature = temperature
        self.include_self_loop = include_self_loop

        # =====================================================
        # 完整起点站嵌入
        # 用于学习起点站之间的自适应关系 [O, O]
        # =====================================================
        self.origin_source_embedding = nn.Parameter(
            torch.empty(
                num_origin,
                embedding_dim,
            )
        )

        self.origin_target_embedding = nn.Parameter(
            torch.empty(
                num_origin,
                embedding_dim,
            )
        )

        # =====================================================
        # 完整终点站嵌入
        # 用于学习终点站之间的自适应关系 [D, D]
        # =====================================================
        self.destination_source_embedding = nn.Parameter(
            torch.empty(
                num_destination,
                embedding_dim,
            )
        )

        self.destination_target_embedding = nn.Parameter(
            torch.empty(
                num_destination,
                embedding_dim,
            )
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(
            self.origin_source_embedding
        )
        nn.init.xavier_uniform_(
            self.origin_target_embedding
        )
        nn.init.xavier_uniform_(
            self.destination_source_embedding
        )
        nn.init.xavier_uniform_(
            self.destination_target_embedding
        )

    def _build_graph(
        self,
        source_embedding,
        target_embedding,
    ):
        """
        根据可学习节点嵌入生成自适应邻接矩阵。

        A = softmax(ReLU(E1 @ E2^T))
        """

        score = torch.matmul(
            source_embedding,
            target_embedding.transpose(0, 1),
        )

        score = score / math.sqrt(
            self.embedding_dim
        )

        score = F.relu(score)

        num_nodes = score.size(0)

        # 是否保留自环
        if not self.include_self_loop:
            identity_mask = torch.eye(
                num_nodes,
                dtype=torch.bool,
                device=score.device,
            )

            score = score.masked_fill(
                identity_mask,
                float("-inf"),
            )

        # =====================================================
        # Top-k 稀疏化
        # =====================================================
        if (
            self.top_k is not None
            and self.top_k < num_nodes
        ):
            _, topk_indices = torch.topk(
                score,
                k=self.top_k,
                dim=-1,
            )

            mask = torch.zeros_like(
                score,
                dtype=torch.bool,
            )

            mask.scatter_(
                dim=-1,
                index=topk_indices,
                value=True,
            )

            score = score.masked_fill(
                ~mask,
                float("-inf"),
            )

        adjacency = F.softmax(
            score / self.temperature,
            dim=-1,
        )

        return adjacency

    def forward(self):
        """
        返回
        ----
        origin_adjacency:
            完整起点站自适应图，[O, O]。

        destination_adjacency:
            完整终点站自适应图，[D, D]。
        """

        origin_adjacency = self._build_graph(
            self.origin_source_embedding,
            self.origin_target_embedding,
        )

        destination_adjacency = self._build_graph(
            self.destination_source_embedding,
            self.destination_target_embedding,
        )

        return (
            origin_adjacency,
            destination_adjacency,
        )


class AdaptiveFullODGraphConv(nn.Module):
    """
    使用自适应 OD 图卷积。
    input:
        [B, C_in, O*D, T]
    output:
        [B, C_out, O*D, T]
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_origin,
        num_destination,
        order=2,
        dropout=0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.num_origin = num_origin
        self.num_destination = num_destination
        self.num_od_nodes = (num_origin * num_destination)

        self.order = order
        self.dropout = dropout

        # 原始特征 X + 起点图传播 order 阶 + 终点图传播 order 阶
        support_num = 1 + 2 * order

        self.output_projection = nn.Conv2d(
            in_channels=(
                in_channels * support_num
            ),
            out_channels=out_channels,
            kernel_size=1,
            bias=True,
        )

    @staticmethod
    def propagate_origin_dimension(
        x,
        origin_adjacency,
    ):
        """
        沿起点维进行传播。

        x:
            [B, C, O, D, T]

        origin_adjacency:
            [O, O]

        对于相同终点 d：
            聚合不同起点的 OD 信息。
        """

        return torch.einsum(
            "oi,bcidt->bcodt",
            origin_adjacency,
            x,
        )

    @staticmethod
    def propagate_destination_dimension(
        x,
        destination_adjacency,
    ):
        """
        沿终点维进行传播。

        x:
            [B, C, O, D, T]

        destination_adjacency:
            [D, D]

        对于相同起点 o：
            聚合不同终点的 OD 信息。
        """

        return torch.einsum(
            "de,bcoet->bcodt",
            destination_adjacency,
            x,
        )

    def forward(
        self,
        x,
        origin_adjacency,
        destination_adjacency,
        return_adjacency=False,
    ):
        if x.dim() != 4:
            raise ValueError(
                "输入 x 应为四维张量 "
                "[B, C, O*D, T]，"
                f"当前形状为 {tuple(x.shape)}"
            )

        batch_size, channels, nodes, time_steps = (
            x.shape
        )
        origin_adjacency = origin_adjacency.to(x.device)
        destination_adjacency = destination_adjacency.to(x.device)

        if nodes != self.num_od_nodes:
            raise ValueError(
                f"输入节点数为 {nodes}，"
                f"但完整 OD 节点数应为 "
                f"{self.num_origin} × "
                f"{self.num_destination} = "
                f"{self.num_od_nodes}"
            )

        # [B, C, O*D, T] -> [B, C, O, D, T]
        x_od = x.reshape(
            batch_size,
            channels,
            self.num_origin,
            self.num_destination,
            time_steps,
        )

        graph_features = [x_od]

        # ===================
        # 沿起点维进行多阶传播
        # ===================
        origin_feature = x_od

        for _ in range(self.order):
            origin_feature = (
                self.propagate_origin_dimension(
                    origin_feature,
                    origin_adjacency,
                )
            )

            graph_features.append(
                origin_feature
            )

        # ===================
        # 沿终点维进行多阶传播
        # ===================
        destination_feature = x_od

        for _ in range(self.order):
            destination_feature = (
                self.propagate_destination_dimension(
                    destination_feature,
                    destination_adjacency,
                )
            )

            graph_features.append(
                destination_feature
            )

        # [B, C*(1+2K), O, D, T]
        graph_features = torch.cat(
            graph_features,
            dim=1,
        )

        graph_features = graph_features.reshape(
            batch_size,
            channels * (1 + 2 * self.order),
            self.num_od_nodes,
            time_steps,
        )

        output = self.output_projection(
            graph_features
        )

        output = F.dropout(
            output,
            p=self.dropout,
            training=self.training,
        )

        if return_adjacency:
            return (
                output,
                origin_adjacency,
                destination_adjacency,
            )

        return output