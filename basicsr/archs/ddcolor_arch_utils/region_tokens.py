import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor


class RegionTokenSpec:
    """轻量占位：后续如果需要基于人脸/语义掩码的 Region tokens 再细化实现。

    目前用户阶段主要使用“dense/grid” 无掩码模式，所以这里只保留
    配置接口，避免脚本 import 失败。
    """

    def __init__(self, num_regions: int = 6, include_area_frac: bool = True):
        self.num_regions = int(num_regions)
        self.include_area_frac = bool(include_area_frac)


class MultiScaleGridTokenConditioner(nn.Module):
    """多尺度 Pyramid Grid conditioner（无掩码）。

    - 输入：参考图 encoder 的多尺度特征 ref_feats: List[Tensor] ，
      每个张量形状为 (B, C_k, H_k, W_k)，通常对应 norm1/norm2/norm3。
    - 输出：
      - cond_tokens_per_scale: List[Tensor]，长度 = num_scales，
        第 k 个元素形状为 (S_k, B, hidden_dim)
      - cond_pos_per_scale: List[Tensor]，同样形状，用作 K/V 的 position embedding。

    设计要点：
    - 每个尺度使用自适应池化到给定 grid_size[k]，形成不同比例、不同长度的 token。
    - 先用 1×1 conv 把通道统一投影到 hidden_dim，再 flatten 成序列。
    - cond_pos 由三部分组成（可开关）：
      1) token 内容本身（与旧实现兼容：token 既是内容又是 K 的偏置）
      2) scale embedding：显式标识尺度 level，帮助区分不同尺度的 token
      3) 2D spatial position embedding：显式标识 grid token 的 (x,y) 空间位置
    """

    def __init__(
        self,
        num_scales: int = 3,
        hidden_dim: int = 256,
        grid_sizes: Sequence[int] = (16, 8, 4),
        use_scale_embed: bool = True,
        use_spatial_pos: bool = True,
        spatial_pos_type: str = "learnable",  # "learnable" | "sincos"
    ) -> None:
        super().__init__()
        self.num_scales = int(num_scales)
        self.hidden_dim = int(hidden_dim)
        self.use_scale_embed = bool(use_scale_embed)
        self.use_spatial_pos = bool(use_spatial_pos)
        self.spatial_pos_type = str(spatial_pos_type)

        grid_sizes = list(grid_sizes)
        if len(grid_sizes) < self.num_scales:
            # 不足时用最后一个补齐，避免配置错误直接崩掉
            grid_sizes = list(grid_sizes) + [grid_sizes[-1]] * (self.num_scales - len(grid_sizes))
        self.grid_sizes = [max(1, int(g)) for g in grid_sizes[: self.num_scales]]

        # 按尺度的 1×1 conv，在第一次 forward 时懒初始化（根据 ref_feats 的 C_k）
        self.input_proj: Optional[nn.ModuleList] = None

        # scale embedding：每个尺度一个 learnable embedding
        if self.use_scale_embed:
            self.scale_embed = nn.Parameter(torch.empty(self.num_scales, self.hidden_dim))
            nn.init.normal_(self.scale_embed, std=0.02)
        else:
            self.register_parameter("scale_embed", None)

        # 2D spatial position embedding（可学习版本：按尺度存成 (1, C, g, g)，forward 时插值到 (gh, gw)）
        if self.use_spatial_pos and self.spatial_pos_type == "learnable":
            emb = []
            for g in self.grid_sizes:
                p = nn.Parameter(torch.empty(1, self.hidden_dim, g, g))
                nn.init.normal_(p, std=0.02)
                emb.append(p)
            self.spatial_pos_embed = nn.ParameterList(emb)
        else:
            # sincos 版本不需要参数；关闭时也无需参数
            self.spatial_pos_embed = None

    @staticmethod
    def _sincos_1d_pos_embed(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """标准 1D sin-cos 位置编码，返回 (length, dim)。dim 必须为偶数。"""
        if dim % 2 != 0:
            raise ValueError(f"sincos pos embed requires even dim, got dim={dim}")
        positions = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)  # (L, 1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / dim)
        )  # (dim/2,)
        angles = positions * div_term.unsqueeze(0)  # (L, dim/2)
        emb = torch.zeros((length, dim), device=device, dtype=dtype)
        emb[:, 0::2] = torch.sin(angles)
        emb[:, 1::2] = torch.cos(angles)
        return emb

    def _get_2d_sincos_pos_embed(self, gh: int, gw: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """2D sin-cos 位置编码，返回 (S=gh*gw, C=hidden_dim)。"""
        # 把 hidden_dim 拆成 h/w 两份，各自做 1D 编码再拼接
        c = self.hidden_dim
        if c % 2 != 0:
            raise ValueError(f"2D sincos pos embed expects hidden_dim even, got hidden_dim={c}")
        c_half = c // 2
        if c_half % 2 != 0:
            # 1D 编码要求 dim 为偶数；hidden_dim=256 时 c_half=128 满足
            raise ValueError(f"2D sincos split expects hidden_dim/2 even, got hidden_dim={c}")

        emb_h = self._sincos_1d_pos_embed(gh, c_half, device=device, dtype=dtype)  # (gh, c/2)
        emb_w = self._sincos_1d_pos_embed(gw, c_half, device=device, dtype=dtype)  # (gw, c/2)

        # meshgrid -> (gh, gw, c)
        emb_h = emb_h[:, None, :].expand(gh, gw, c_half)
        emb_w = emb_w[None, :, :].expand(gh, gw, c_half)
        emb = torch.cat([emb_h, emb_w], dim=-1)  # (gh, gw, c)
        return emb.reshape(gh * gw, c)

    def _build_input_proj(self, ref_feats: List[Tensor]) -> None:
        assert len(ref_feats) >= self.num_scales
        projs = []
        for k in range(self.num_scales):
            c_in = int(ref_feats[k].shape[1])
            conv = nn.Conv2d(c_in, self.hidden_dim, kernel_size=1)
            nn.init.kaiming_uniform_(conv.weight, a=1.0)
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0.0)
            projs.append(conv)
        self.input_proj = nn.ModuleList(projs)

    def forward(
        self,
        ref_feats: List[Tensor],
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Args:
            ref_feats: list of tensors, length >= num_scales,
                       each of shape (B, C_k, H_k, W_k)
        Returns:
            cond_tokens_per_scale, cond_pos_per_scale
        """
        assert isinstance(ref_feats, (list, tuple)) and len(ref_feats) >= self.num_scales
        # 约定使用列表前 num_scales 个尺度（与 infer_style_transfer 中一致）
        feats = list(ref_feats[: self.num_scales])

        if self.input_proj is None:
            self._build_input_proj(feats)
        assert self.input_proj is not None

        cond_tokens_per_scale: List[Tensor] = []
        cond_pos_per_scale: List[Tensor] = []

        for level, feat in enumerate(feats):
            # feat: (B, C, H, W)
            B, _, H, W = feat.shape

            proj = self.input_proj[level](feat)  # (B, hidden_dim, H, W)

            # 对应尺度的 grid 大小，若特征图太小自动 clamp
            g = self.grid_sizes[level]
            gh = min(g, H)
            gw = min(g, W)

            pooled = F.adaptive_avg_pool2d(proj, (gh, gw))  # (B, hidden_dim, gh, gw)
            B, C, gh, gw = pooled.shape

            # (B, C, gh, gw) -> (S_k, B, C)
            tokens = pooled.flatten(2).permute(2, 0, 1).contiguous()

            # cond_pos：token 内容 + scale embedding + 2D spatial pos（均为可选）
            pos = tokens

            if self.use_scale_embed and self.scale_embed is not None:
                # (1, 1, C) broadcast to (S_k, B, C)
                scale = self.scale_embed[level].view(1, 1, self.hidden_dim).to(device=tokens.device, dtype=tokens.dtype)
                pos = pos + scale

            if self.use_spatial_pos:
                if self.spatial_pos_type == "learnable":
                    if self.spatial_pos_embed is None:
                        raise RuntimeError("spatial_pos_type='learnable' but spatial_pos_embed is None")
                    base = self.spatial_pos_embed[level].to(device=tokens.device, dtype=tokens.dtype)  # (1,C,g,g)
                    if base.shape[-2] != gh or base.shape[-1] != gw:
                        base = F.interpolate(base, size=(gh, gw), mode="bicubic", align_corners=False)
                    # (1,C,gh,gw) -> (S_k,1,C)
                    spatial = base.flatten(2).permute(2, 0, 1).contiguous()
                elif self.spatial_pos_type in ("sincos", "sin", "sine", "sinusoidal"):
                    spatial = self._get_2d_sincos_pos_embed(
                        gh, gw, device=tokens.device, dtype=tokens.dtype
                    ).unsqueeze(1)  # (S_k,1,C)
                else:
                    raise ValueError(f"Unknown spatial_pos_type={self.spatial_pos_type!r}")
                pos = pos + spatial

            cond_tokens_per_scale.append(tokens)
            cond_pos_per_scale.append(pos)

        return cond_tokens_per_scale, cond_pos_per_scale


class MultiScaleDenseTokenConditioner(nn.Module):
    """兼容现有脚本的 dense/grid conditioner。

    - 对外接口与 `scripts/infer_style_transfer.py` 中使用的一致：
      `MultiScaleDenseTokenConditioner(num_scales, hidden_dim, grid_size)`
    - 内部使用 Pyramid Grid：从单一 `grid_size` 推导出多尺度
      `[grid_size, grid_size/2, grid_size/4, ...]`。
    - forward 返回展平成一条长序列的 `(cond_tokens, cond_pos)`，形状均为
      `(S_all, B, hidden_dim)`，方便旧接口直接使用。
    """

    def __init__(
        self,
        num_scales: int = 3,
        hidden_dim: int = 256,
        grid_size: int = 16,
        use_scale_embed: bool = True,
        use_spatial_pos: bool = True,
        spatial_pos_type: str = "learnable",
    ) -> None:
        super().__init__()
        self.num_scales = int(num_scales)
        self.hidden_dim = int(hidden_dim)
        self.grid_size = int(grid_size)
        self.use_scale_embed = bool(use_scale_embed)
        self.use_spatial_pos = bool(use_spatial_pos)
        self.spatial_pos_type = str(spatial_pos_type)

        # 从一个 base grid_size 派生多尺度 pyramid grid
        derived: List[int] = []
        base = max(1, self.grid_size)
        for k in range(self.num_scales):
            g = max(1, base // (2 ** k))
            derived.append(g)

        self.grid_conditioner = MultiScaleGridTokenConditioner(
            num_scales=self.num_scales,
            hidden_dim=self.hidden_dim,
            grid_sizes=derived,
            use_scale_embed=self.use_scale_embed,
            use_spatial_pos=self.use_spatial_pos,
            spatial_pos_type=self.spatial_pos_type,
        )

    def forward(
        self,
        ref_feats: List[Tensor],
        masks: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        # 当前 dense/grid 模式不依赖人脸/语义掩码，忽略 masks。
        cond_tokens_per_scale, cond_pos_per_scale = self.grid_conditioner(ref_feats)

        # 在序列维拼接，兼容原来的 (S, B, C) 接口
        cond_tokens = torch.cat(cond_tokens_per_scale, dim=0)
        cond_pos = torch.cat(cond_pos_per_scale, dim=0)
        return cond_tokens, cond_pos


class MultiScaleRegionTokenConditioner(nn.Module):
    """占位实现：目前阶段不再依赖 face_parse / 语义分割。

    若后续需要重新启用基于人脸/区域的条件 tokens，可在此类中
    按 cond_ddcolor.md 中的“方法 2”补全 masked pooling 逻辑。
    现在 forward 若被误用会直接报错，避免悄悄走错分支。
    """

    def __init__(
        self,
        spec: Optional[RegionTokenSpec] = None,
        num_scales: int = 3,
        hidden_dim: int = 256,
        include_area_frac: bool = True,
    ) -> None:
        super().__init__()
        self.spec = spec or RegionTokenSpec()
        self.num_scales = int(num_scales)
        self.hidden_dim = int(hidden_dim)
        self.include_area_frac = bool(include_area_frac)

    def forward(
        self,
        ref_feats: List[Tensor],
        masks: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError(
            "MultiScaleRegionTokenConditioner 当前阶段未启用（已按需求移除 face_parse / 区域掩码路径）。"
        )

