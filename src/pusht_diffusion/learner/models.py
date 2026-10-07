"""USER OWNED. FiLM, residual blocks, and the U-Net body were moved here.

B=batch, T=two observations, H=16 action times. Environment time t and
noise time k are distinct. See docs/learning.md for independent milestones.
"""
from __future__ import annotations
import torch
from torch import Tensor, nn
import math

def diffusion_step_embedding(k: Tensor, dim: int = 128) -> Tensor:
    # Sinusoidal embedding of diffusion step k [B] -> [B,dim].
    i = torch.arange(dim // 2, device=k.device, dtype=torch.float32)
    frequencies = torch.exp(-math.log(10000.0) * i / (dim // 2 - 1))
    angels = k.float()[:, None] * frequencies[None, :]
    return torch.cat([torch.sin(angels), torch.cos(angels)], dim=-1)  # [B,dim]

class VisualEncoder(nn.Module):
    """Encode two normalized Push-T images with a shared ResNet-18.

    Images [B, 2, 3, 96, 96] -> per-frame features [B, 2, 256].
    BatchNorm running statistics stay fixed while backbone weights can train.
    """

    def __init__(self, projection_dim: int = 256, *, initialize_pretrained: bool = False) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1 if initialize_pretrained else None
        backbone = resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.projection = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(512 * 3 * 3, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.Mish(),
        )
        self.train(True)

    def train(self, mode: bool = True) -> VisualEncoder:
        super().train(mode)
        for module in self.backbone.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 5 or images.shape[1:] != (2, 3, 96, 96):
            raise ValueError("images must have shape [B, 2, 3, 96, 96]")
        batch = images.shape[0]
        frames = images.reshape(batch * 2, 3, 96, 96)
        features = self.backbone(frames)
        if features.shape[1:] != (512, 3, 3):
            raise ValueError("ResNet-18 must produce [2B, 512, 3, 3]")
        return self.projection(features).reshape(batch, 2, -1)

class FiLM(nn.Module):
    #input: observation(B, 512 + 4) concat diffusion step(B, 128)
    #output: (B, 2 * C_out) split into gamma(B, C_out), beta(B, C_out)
    # One Linear layer to generate gamma and beta from the conditioning input
    def __init__(self, out_channels, cond_dim):
        super().__init__()
        self.linear = nn.Linear(cond_dim, 2 * out_channels)
    def forward(self, x, cond):
        cond = nn.SiLU()(cond)
        s, beta = self.linear(cond).chunk(2, dim=1)  # Each: (B, C_out)
        s = s.unsqueeze(-1)  # (B, C_out, 1), broadcast over action time
        beta = beta.unsqueeze(-1)
        return (s + 1) * x + beta



class ConditionalResidualBlock(nn.Module):
    # input(B, C_in, T) output(B, C_out, T)
    # conv1d(kernel_size=5, padding=2)
    # groupnorm(num_groups=8)
    # Mish
    # FiLM
    # conv1d(kernel_size=5, padding=2)
    # groupnorm(num_groups=8)
    # Mish
    # add shortcut
    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.mish1 = nn.Mish()
        self.film = FiLM(out_channels, cond_dim)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.mish2 = nn.Mish()
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, cond):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.mish1(out)
        out = self.film(out, cond)
        out = self.conv2(out)
        out = self.gn2(out)
        out = self.mish2(out)
        out += residual
        return out

class UnetDiffusionModel(nn.Module):
    def __init__(self, cond_dim):
        super().__init__()
        self.resblock1 = ConditionalResidualBlock(in_channels=2, out_channels=128, cond_dim=cond_dim)
        self.resblock2 = ConditionalResidualBlock(in_channels=128, out_channels=128, cond_dim=cond_dim)
        self.resblock3 = ConditionalResidualBlock(in_channels=128, out_channels=256, cond_dim=cond_dim)
        self.resblock4 = ConditionalResidualBlock(in_channels=256, out_channels=256, cond_dim=cond_dim)
        self.resblock5 = ConditionalResidualBlock(in_channels=256, out_channels=512, cond_dim=cond_dim)
        self.resblock6 = ConditionalResidualBlock(in_channels=512, out_channels=512, cond_dim=cond_dim)
        self.resblock7 = ConditionalResidualBlock(in_channels=512, out_channels=256, cond_dim=cond_dim)
        self.resblock8 = ConditionalResidualBlock(in_channels=256, out_channels=256, cond_dim=cond_dim)
        self.resblock9 = ConditionalResidualBlock(in_channels=256, out_channels=128, cond_dim=cond_dim)
        self.resblock10 = ConditionalResidualBlock(in_channels=128, out_channels=128, cond_dim=cond_dim)
        self.final_conv = nn.Conv1d(in_channels=128, out_channels=2, kernel_size=1)
        self.downsample1 = nn.Conv1d(kernel_size=4, stride=2, in_channels=128, out_channels=128, padding=1)
        self.downsample2 = nn.Conv1d(kernel_size=4, stride=2, in_channels=256, out_channels=256, padding=1)
        self.upsample1 = nn.ConvTranspose1d(kernel_size=4, stride=2, in_channels=512, out_channels=256, padding=1)
        self.upsample2 = nn.ConvTranspose1d(kernel_size=4, stride=2, in_channels=256, out_channels=128, padding=1)

    def forward(self, x, cond):
        # noisy_action_chunk: (B, 16, 2)
        x = x.transpose(1, 2)  # (B, 2, 16)
        x = self.resblock1(x, cond)  # (B, 128, 16)
        skip16 = self.resblock2(x, cond)  # (B, 128, 16)
        x = self.downsample1(skip16)  # (B, 128, 8)
        x = self.resblock3(x, cond)  # (B, 256, 8)
        skip8 = self.resblock4(x, cond)  # (B, 256, 8)
        x = self.downsample2(skip8)  # (B, 256, 4)
        x = self.resblock5(x, cond)  # (B, 512, 4)
        x = self.resblock6(x, cond)  # (B, 512, 4)
        x = self.upsample1(x)  # (B, 256, 8)
        x = torch.cat([x, skip8], dim=1)  # (B, 512, 8)
        x = self.resblock7(x, cond)  # (B, 256, 8)
        x = self.resblock8(x, cond)  # (B, 256, 8)
        x = self.upsample2(x)  # (B, 128, 16)
        x = torch.cat([x, skip16], dim=1)  #(B, 256, 16)
        x = self.resblock9(x, cond)  # (B, 128, 16)
        x = self.resblock10(x, cond)  # (B, 128, 16)
        x = self.final_conv(x)  # (B, 2, 16)
        x = x.transpose(1, 2)  # (B, 16, 2)
        return x

class DiffusionPolicy(nn.Module):
    def __init__(self, cond_dim):
        super().__init__()
        self.unet = UnetDiffusionModel(cond_dim=cond_dim)
        self.visual_encoder = VisualEncoder(projection_dim=256, initialize_pretrained=True)
        
    def forward(self, x: Tensor, images: Tensor, position: Tensor, k: Tensor) -> Tensor:
        cond = torch.concat([self.visual_encoder(images), position], dim=-1).flatten(1)  # [B, 516]
        cond = torch.concat([cond, diffusion_step_embedding(k, dim=128)], dim=-1)  # [B, 516 + 128]
        return self.unet(x, cond)

