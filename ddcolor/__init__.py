import os
from typing import Type, Optional

import torch
from torch import nn

# Re-export the training-time DDColor architecture
from basicsr.archs.ddcolor_arch import DDColor as _DDColorImpl


DDColor = _DDColorImpl


def _load_basicsr_checkpoint(
    model: nn.Module,
    model_path: str,
    map_location: str = "cpu",
    param_key: Optional[str] = "params_ema",
    strict: bool = True,
) -> nn.Module:
    """Load a BasicSR-style generator checkpoint (net_g_*.pth) into `model`.

    This mirrors the logic in `basicsr.models.base_model.BaseModel.load_network`,
    but in a lightweight, script-friendly form.
    """
    if not model_path or not os.path.isfile(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    ckpt = torch.load(model_path, map_location=map_location)

    if isinstance(ckpt, dict) and any(k in ckpt for k in ["params_ema", "params"]):
        # Prefer EMA params if available, otherwise fall back to 'params'
        if param_key is None:
            key = "params_ema" if "params_ema" in ckpt else "params"
        else:
            key = param_key
            if key not in ckpt:
                key = "params_ema" if "params_ema" in ckpt else "params"
        state = ckpt[key]
    else:
        # Raw state_dict
        state = ckpt

    # Strip potential DistributedDataParallel "module." prefix
    cleaned = {}
    for k, v in state.items():
        if k.startswith("module."):
            cleaned[k[7:]] = v
        else:
            cleaned[k] = v

    model.load_state_dict(cleaned, strict=strict)
    return model


def build_ddcolor_model(
    arch_cls: Type[nn.Module],
    *,
    model_path: Optional[str] = None,
    input_size: int = 512,
    model_size: str = "large",
    decoder_type: str = "MultiScaleColorDecoder",
    device: Optional[torch.device] = None,
    num_queries: int = 256,
    num_scales: int = 3,
    dec_layers: int = 9,
) -> nn.Module:
    """Factory used by all inference scripts to construct a DDColor model.

    This is a local re-implementation of the original `ddcolor` pip package API,
    wired to the training-time architecture in `basicsr.archs.ddcolor_arch.DDColor`
    and to BasicSR-style `net_g_*.pth` checkpoints.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_size = str(model_size).lower()
    if model_size not in {"tiny", "large"}:
        raise ValueError(f"Unsupported model_size: {model_size!r}")

    # Map logical size -> encoder backbone; this mirrors the training configs,
    # where we use ConvNeXt-Large for the 'large' variant.
    if model_size == "large":
        encoder_name = "convnext-l"
        nf = 512
    else:
        # Tiny variant reuses the ConvNeXt-Tiny backbone by convention.
        encoder_name = "convnext-t"
        nf = 384

    # Our cond-B dense-gated training config uses:
    # - last_norm = Spectral
    # - num_output_channels = 2 (ab in Lab space)
    # - do_normalize = False (we feed gray RGB and output ab directly)
    model: nn.Module = arch_cls(
        encoder_name=encoder_name,
        decoder_name=decoder_type,
        num_input_channels=3,
        input_size=(input_size, input_size),
        nf=nf,
        num_output_channels=2,
        last_norm="Spectral",
        do_normalize=False,
        num_queries=num_queries,
        num_scales=num_scales,
        dec_layers=dec_layers,
        use_cond_gate=True,
        cond_gate_init=-1.5,
        encoder_from_pretrain=False,
    )

    model.to(device)
    model.eval()

    if model_path is not None:
        model = _load_basicsr_checkpoint(
            model,
            model_path=model_path,
            map_location="cpu",
            param_key="params_ema",
            strict=False,
        )
        model.to(device)
        model.eval()

    return model


__all__ = [
    "DDColor",
    "build_ddcolor_model",
]

