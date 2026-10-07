"""Single-frame ACT (Action Chunking with Transformers) for Push-T.

Inputs are already ImageNet-normalized images and standardized positions/actions.
Normalization statistics and action denormalization belong to the data/control layer.
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


def sinusoidal_encoding(length: int, dimension: int) -> Tensor:
    """Fixed positions, with paired sine/cosine frequencies."""

    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(torch.arange(0, dimension, 2, dtype=torch.float32) * (-math.log(10000.0) / dimension))
    encoding = torch.empty(length, dimension)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(positions * frequencies)
    return encoding


class VisualEncoder(nn.Module):
    """ResNet-18 (Residual Network) spatial grid: 96x96 -> 9 tokens."""

    def __init__(self, d_model=256, pretrained=True):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        # Remove both global average pooling and the classification layer.
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        self.projection = nn.Conv2d(512, d_model, kernel_size=1)
        self.d_model = d_model

    def forward(self, x):

        features = self.projection(self.feature_extractor(x))
        height, width = features.shape[-2:]
        # Concatenated row/column encodings preserve 2D spatial identity.
        row = sinusoidal_encoding(height, self.d_model // 2)
        column = sinusoidal_encoding(width, self.d_model // 2)
        positions = torch.cat(
            [
                row[:, None, :].expand(-1, width, -1),
                column[None, :, :].expand(height, -1, -1),
            ],
            dim=-1,
        ).reshape(1, height * width, self.d_model)
        return features.flatten(2).transpose(1, 2) + positions.to(features)


class StateEncoder(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, d_model),
            nn.ReLU(),
        )

    def forward(self, x):

        return self.state_encoder(x).unsqueeze(1)


class LatentEncoder(nn.Module):
    """Conditional variational posterior q(z | position, valid expert actions)."""

    def __init__(self, d_model=256, nhead=8, num_layers=4, latent_dim=32, chunk_size=16):
        super().__init__()

        self.chunk_size = chunk_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position_projection = nn.Linear(2, d_model)
        self.action_projection = nn.Linear(2, d_model)
        self.encoder = PolicyEncoder(d_model, nhead, num_layers)
        self.posterior_projection = nn.Linear(d_model, 2 * latent_dim)
        self.register_buffer("time_encoding", sinusoidal_encoding(chunk_size + 2, d_model))

    def forward(self, position, actions, valid_mask):

        # Sanitize before projection: even NaN padding cannot enter attention.
        safe_actions = actions.masked_fill(~valid_mask.unsqueeze(-1), 0)
        tokens = torch.cat(
            [
                self.cls_token.expand(actions.shape[0], -1, -1),
                self.position_projection(position).unsqueeze(1),
                self.action_projection(safe_actions),
            ],
            dim=1,
        )
        tokens = tokens + self.time_encoding.to(tokens)
        prefix_mask = valid_mask.new_zeros((actions.shape[0], 2))
        padding_mask = torch.cat([prefix_mask, ~valid_mask], dim=1)
        summary = self.encoder(tokens, padding_mask)[:, 0]
        mu, logvar = self.posterior_projection(summary).chunk(2, dim=-1)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return z, mu, logvar


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model=256, nhead=8):
        super().__init__()

        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.attn_output = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x, key_padding_mask=None):
        # x: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.size()
        qkv = self.qkv(x).view(batch_size, seq_len, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        if key_padding_mask is not None:

            attn_weights = attn_weights.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        x = self.norm1(x + self.attn_output(attn_output))
        return self.norm2(x + self.ffn(x))


class PolicyEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=4):
        super().__init__()

        self.transformer_blocks = nn.ModuleList([TransformerEncoderBlock(d_model, nhead) for _ in range(num_layers)])

    def forward(self, x, key_padding_mask=None):
        for block in self.transformer_blocks:
            x = block(x, key_padding_mask)
        return x


class PolicyDecoder(nn.Module):
    """单层策略解码器：[B, S, D] 场景表示 -> [B, K, D] 动作表示。

    使用可学习动作 token 作为初始内容；依次进行自注意力、交叉注意力
    和逐 token 前馈变换，每个子层采用残差连接后层归一化。
    此处不做二维动作投影，也不施加自回归因果掩码。
    """

    def __init__(self, d_model=256, nhead=8, num_action_queries=16):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.num_action_queries = num_action_queries

        # One learned token for each action in the predicted chunk.
        self.action_embeddings = nn.Embedding(num_action_queries, d_model)
        self.action_self_attn_qkv = nn.Linear(d_model, d_model * 3)
        self.action_self_attn_output = nn.Linear(d_model, d_model)

        # Action tokens supply queries; encoder memory supplies keys and values.
        self.decoder_query = nn.Linear(d_model, d_model)
        self.decoder_kv = nn.Linear(d_model, d_model * 2)
        self.cross_attn_output = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def _split_heads(self, tensor):
        # [B, L, D] -> [B, H, L, D/H]
        batch_size, seq_len, _ = tensor.shape
        return tensor.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)

    def _merge_heads(self, tensor):
        # [B, H, L, D/H] -> [B, L, D]
        batch_size, _, seq_len, _ = tensor.shape
        return tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, x):
        # x: [B, S, D], output: [B, 16, D] by default.

        batch_size = x.shape[0]
        action_tokens = self.action_embeddings.weight.unsqueeze(0).expand(batch_size, -1, -1)

        # Self-attention among the 16 learned action tokens.
        self_q, self_k, self_v = self.action_self_attn_qkv(action_tokens).chunk(3, dim=-1)
        self_q = self._split_heads(self_q)
        self_k = self._split_heads(self_k)
        self_v = self._split_heads(self_v)
        self_attention = torch.softmax(
            torch.matmul(self_q, self_k.transpose(-2, -1)) / (self.head_dim**0.5),
            dim=-1,
        )
        self_attn_output = self.action_self_attn_output(self._merge_heads(torch.matmul(self_attention, self_v)))
        action_tokens = self.norm1(action_tokens + self_attn_output)

        # Cross-attention: Q comes from action tokens; K and V come from x.
        query = self._split_heads(self.decoder_query(action_tokens))
        memory_k, memory_v = self.decoder_kv(x).chunk(2, dim=-1)
        memory_k = self._split_heads(memory_k)
        memory_v = self._split_heads(memory_v)
        cross_attention = torch.softmax(
            torch.matmul(query, memory_k.transpose(-2, -1)) / (self.head_dim**0.5),
            dim=-1,
        )
        decoded_actions = torch.matmul(cross_attention, memory_v)
        cross_attn_output = self.cross_attn_output(self._merge_heads(decoded_actions))
        action_tokens = self.norm2(action_tokens + cross_attn_output)
        # Feed-forward network acts independently on each of the K tokens.
        return self.norm3(action_tokens + self.ffn(action_tokens))


class ActionPolicy(nn.Module):
    """Training returns a dict with actions/mu/logvar; evaluation returns actions.

    Train: forward(image, position, action_chunk, action_valid_mask).
    Deploy: eval(); forward(image, position), with a fixed zero latent.
    Outputs remain standardized; callers denormalize and clip for the environment.
    """

    def __init__(
        self,
        pretrained=True,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_latent_layers=4,
        latent_dim=32,
        chunk_size=16,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.chunk_size = chunk_size
        self.visual_encoder = VisualEncoder(d_model, pretrained)
        self.state_encoder = StateEncoder(d_model)
        self.latent_encoder = LatentEncoder(d_model, nhead, num_latent_layers, latent_dim, chunk_size)
        self.latent_projection = nn.Linear(latent_dim, d_model)
        self.token_type_embeddings = nn.Embedding(3, d_model)
        self.policy_encoder = PolicyEncoder(d_model, nhead, num_encoder_layers)
        self.policy_decoder = PolicyDecoder(d_model, nhead, chunk_size)
        self.action_head = nn.Linear(d_model, 2)

    def forward(self, observation_image, agent_position, action_chunk=None, action_valid_mask=None):

        if self.training:
            z, mu, logvar = self.latent_encoder(agent_position, action_chunk, action_valid_mask)
        else:

            z = agent_position.new_zeros((agent_position.shape[0], self.latent_dim))

        types = self.token_type_embeddings.weight
        visual = self.visual_encoder(observation_image) + types[0]
        state = self.state_encoder(agent_position) + types[1]
        latent = self.latent_projection(z).unsqueeze(1) + types[2]
        memory = self.policy_encoder(torch.cat([visual, state, latent], dim=1))
        actions = self.action_head(self.policy_decoder(memory))
        if self.training:
            return {"actions": actions, "mu": mu, "logvar": logvar}
        return actions
