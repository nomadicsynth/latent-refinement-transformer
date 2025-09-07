"""NeonVisionProcessor: Hugging Face style image -> token/embedding processor.

Supports encoder types defined in `image_encoders.py`:
  - raw: flatten 28x28 grayscale pixels into token IDs (0..255) (MNIST style)
  - conv: `ConvStemEncoder` -> embeddings
  - patch: `PatchEmbeddingEncoder` -> embeddings
  - wavelet: `WaveletEncoder` (requires pywt) -> embeddings
  - siren: `SIRENImplicitEncoder` (coordinate implicit features) -> embeddings

Returned dict keys:
  raw mode -> {"input_ids": LongTensor[B, 785 or 784?], "attention_mask": LongTensor, ...}
  embedding modes -> {"inputs_embeds": FloatTensor[B, N, hidden], "attention_mask": LongTensor[B,N]}

This processor intentionally does NOT add BOS/label tokens; those are added later in the training pipeline.

Serialization: `save_pretrained(output_dir)` saves a JSON config plus (optionally) encoder weights under `encoder.pt` for learned encoders.
`from_pretrained(path)` restores processor & encoder weights.

Example:
```python
from models.neon_vision_processor import NeonVisionProcessor
proc = NeonVisionProcessor(encoder_type="conv", image_size=28, hidden_size=256)
imgs = torch.randn(8,1,28,28)
out = proc(imgs)
# pass out["inputs_embeds"] to model via Trainer data_collator
```
"""
from __future__ import annotations
import json, os, math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Union, List, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from PIL import Image as PILImage  # type: ignore

import torch
import torch.nn as nn

from .image_encoders import (
    ConvStemEncoder,
    PatchEmbeddingEncoder,
    WaveletEncoder,
    SIRENImplicitEncoder,
)

try:  # optional dependency for HF mixin style (not strictly required)
    from transformers.image_processing_utils import ImageProcessingMixin
except Exception:  # fallback minimal mixin
    class ImageProcessingMixin:  # type: ignore
        def save_pretrained(self, save_directory: str, **kwargs):  # pragma: no cover
            raise NotImplementedError


ENCODER_TYPES = {"raw", "conv", "patch", "wavelet", "siren"}


@dataclass
class NeonVisionConfig:
    encoder_type: str = "raw"
    image_size: int = 28
    in_channels: int = 1
    patch_size: int = 4
    hidden_size: int = 256  # output dim for embeddings
    conv_hidden: int = 64    # internal hidden channels for conv stem
    wavelet: str = "haar"
    wavelet_levels: int = 2
    siren_hidden: int = 128
    siren_layers: int = 3
    normalize: bool = True
    mean: float = 0.1307  # MNIST default
    std: float = 0.3081
    pad_to_square: bool = False  # future use

    def to_dict(self):  # stable order
        return {
            "encoder_type": self.encoder_type,
            "image_size": self.image_size,
            "in_channels": self.in_channels,
            "patch_size": self.patch_size,
            "hidden_size": self.hidden_size,
            "conv_hidden": self.conv_hidden,
            "wavelet": self.wavelet,
            "wavelet_levels": self.wavelet_levels,
            "siren_hidden": self.siren_hidden,
            "siren_layers": self.siren_layers,
            "normalize": self.normalize,
            "mean": self.mean,
            "std": self.std,
            "pad_to_square": self.pad_to_square,
        }


class NeonVisionProcessor(ImageProcessingMixin):
    def __init__(self, *, config: Optional[NeonVisionConfig] = None, **kwargs):
        if config is None:
            config = NeonVisionConfig(**kwargs)
        else:
            # allow override via kwargs
            for k, v in kwargs.items():
                if hasattr(config, k):
                    setattr(config, k, v)
        if config.encoder_type not in ENCODER_TYPES:
            raise ValueError(f"encoder_type must be one of {ENCODER_TYPES}")
        self.config = config
        self.encoder: Optional[nn.Module] = None
        if config.encoder_type != "raw":
            self._build_encoder()

    # ---------------- Encoder Construction ------------------
    def _build_encoder(self):
        et = self.config.encoder_type
        c = self.config
        if et == "conv":
            self.encoder = ConvStemEncoder(in_ch=c.in_channels, hidden=c.conv_hidden, out_dim=c.hidden_size)
        elif et == "patch":
            self.encoder = PatchEmbeddingEncoder(img_size=c.image_size, patch=c.patch_size, in_ch=c.in_channels, out_dim=c.hidden_size)
        elif et == "wavelet":
            self.encoder = WaveletEncoder(wave=c.wavelet, levels=c.wavelet_levels, out_dim=c.hidden_size)
        elif et == "siren":
            self.encoder = SIRENImplicitEncoder(img_size=c.image_size, hidden=c.siren_hidden, layers=c.siren_layers, out_dim=c.hidden_size)
        else:
            self.encoder = None

    # ---------------- Public API ------------------
    def __call__(self, images: Union[torch.Tensor, List[torch.Tensor]], return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
        tensor = self._to_tensor(images)
        if self.config.normalize:
            tensor = (tensor - self.config.mean) / (self.config.std + 1e-8)
        if self.config.encoder_type == "raw":
            return self._encode_raw(tensor)
        else:
            return self._encode_embeds(tensor)

    # ---------------- Tensor Prep ------------------
    def _to_tensor(self, images) -> torch.Tensor:
        if isinstance(images, torch.Tensor):
            t = images
            if t.dim() == 3:  # (C,H,W) -> (1,C,H,W)
                t = t.unsqueeze(0)
        else:
            # list of tensors or PIL images
            arr = []
            for im in images if isinstance(images, (list, tuple)) else [images]:
                if 'PIL' in str(type(im)):
                    import torchvision.transforms as T  # local import
                    to_t = T.ToTensor()
                    arr.append(to_t(im))
                elif isinstance(im, torch.Tensor):
                    arr.append(im)
                else:
                    raise TypeError("Unsupported image element type")
            t = torch.stack(arr, dim=0)
        # Expect (B,C,H,W)
        if t.size(1) != self.config.in_channels:
            raise ValueError(f"Expected in_channels={self.config.in_channels}, got {t.size(1)}")
        return t

    # ---------------- Raw token path ------------------
    def _encode_raw(self, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        # t: (B,1,H,W) -> flatten to uint8 0..255 tokens
        B = t.size(0)
        if t.dtype != torch.float32:
            t = t.float()
        # Assume values ~ normalized; rescale to 0..255 after clamp to [-4,4]
        v = torch.clamp(t, -4, 4)
        v = (v - v.min()) / (v.max() - v.min() + 1e-8)
        tok = (v * 255).to(torch.long).view(B, -1)  # (B,H*W)
        attn = torch.ones_like(tok)
        return {"input_ids": tok, "attention_mask": attn}

    # ---------------- Embedding path ------------------
    def _encode_embeds(self, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.encoder is None:
            self._build_encoder()
        assert self.encoder is not None
        emb = self.encoder(t)  # (B,N,D)
        attn = torch.ones(emb.size(0), emb.size(1), dtype=torch.long, device=emb.device)
        return {"inputs_embeds": emb, "attention_mask": attn}

    # ---------------- Serialization ------------------
    def to_dict(self):
        return self.config.to_dict()

    def save_pretrained(self, save_directory: str, save_encoder_weights: bool = True):
        os.makedirs(save_directory, exist_ok=True)
        cfg_path = os.path.join(save_directory, "neon_vision_config.json")
        with open(cfg_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        if save_encoder_weights and self.encoder is not None:
            torch.save(self.encoder.state_dict(), os.path.join(save_directory, "encoder.pt"))

    @classmethod
    def from_pretrained(cls, directory: str, map_location: Optional[str] = None):
        cfg_path = os.path.join(directory, "neon_vision_config.json")
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"Missing config at {cfg_path}")
        with open(cfg_path, "r") as f:
            data = json.load(f)
        config = NeonVisionConfig(**data)
        proc = cls(config=config)
        enc_path = os.path.join(directory, "encoder.pt")
        if os.path.isfile(enc_path) and proc.encoder is not None:
            state = torch.load(enc_path, map_location=map_location or 'cpu')
            proc.encoder.load_state_dict(state, strict=False)
        return proc

    # HF compatibility naming
    def to_json_string(self):  # pragma: no cover
        return json.dumps(self.config.to_dict(), indent=2) + "\n"


__all__ = ["NeonVisionProcessor", "NeonVisionConfig"]
