import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from typing import Tuple, Optional


class GraphTemporalTransformer(nn.Module):
    """
    Graph-Temporal Transformer Encoder with Edge Attention for financial fraud detection.
    Models high-order topological interaction subgraphs and multi-hop transaction flows.

    Architecture:
    - Layer 1 (Neighborhood Aggregation):
        TransformerConv(165 -> 64 * 4 heads = 256), with residual skip projection,
        ELU activation, dropout, and LayerNorm.
    - Layer 2 (Multi-Hop Syndicate Extraction):
        TransformerConv(256 -> 32, 1 head = 32), with residual skip projection,
        ELU activation, dropout, and LayerNorm.
    - Latent Representation:
        32-dimensional structural node embedding.
    - Auxiliary Classification Head:
        Linear(32 -> 16) -> ReLU -> Dropout(0.2) -> Linear(16 -> 2).
    """

    def __init__(
        self,
        in_channels: int = 165,
        hidden_channels: int = 64,
        num_heads: int = 4,
        embedding_dim: int = 32,
        out_channels: int = 2,
        dropout: float = 0.2,
        beta: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.out_channels = out_channels
        self.dropout = dropout

        layer1_out_dim = hidden_channels * num_heads  # 64 * 4 = 256

        # =====================================================================
        # Layer 1: Multi-Head Neighborhood Attention Aggregation
        # =====================================================================
        self.conv1 = TransformerConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=num_heads,
            concat=True,
            beta=beta,
            dropout=dropout,
        )
        self.skip1 = nn.Linear(in_channels, layer1_out_dim, bias=False)
        self.norm1 = nn.LayerNorm(layer1_out_dim)

        # =====================================================================
        # Layer 2: Multi-Hop Syndicate & Temporal Motif Extraction
        # =====================================================================
        self.conv2 = TransformerConv(
            in_channels=layer1_out_dim,
            out_channels=embedding_dim,
            heads=1,
            concat=False,
            beta=beta,
            dropout=dropout,
        )
        self.skip2 = nn.Linear(layer1_out_dim, embedding_dim, bias=False)
        self.norm2 = nn.LayerNorm(embedding_dim)

        # =====================================================================
        # Auxiliary Classification Head
        # =====================================================================
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, out_channels),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the Graph-Temporal Transformer.

        Args:
            x: Node feature tensor [num_nodes, in_channels]
            edge_index: Edge index tensor [2, num_edges]
            edge_attr: Optional edge feature tensor

        Returns:
            Tuple of:
                - embeddings: [num_nodes, 32] structural node representations
                - logits: [num_nodes, 2] class prediction logits
        """
        # --- Layer 1 ---
        h1_conv = self.conv1(x, edge_index, edge_attr=edge_attr)
        h1_skip = self.skip1(x)
        h1 = self.norm1(F.elu(h1_conv + h1_skip))
        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        # --- Layer 2 (Structural Embedding) ---
        h2_conv = self.conv2(h1, edge_index, edge_attr=edge_attr)
        h2_skip = self.skip2(h1)
        embeddings = self.norm2(F.elu(h2_conv + h2_skip))

        # --- Auxiliary Classifier Head ---
        logits = self.classifier(embeddings)

        return embeddings, logits
