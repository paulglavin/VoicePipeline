"""
noise_suppressor.py — GTCRN noise suppression wrapper.

Architecture: Gated Temporal Convolutional Recurrent Network (GTCRN-lite).
Reference: https://github.com/Xiaobin-Rong/gtcrn

Pretrained weights are downloaded from HuggingFace on first run and cached
in MODEL_DIR/gtcrn/. If the download fails the model still runs (untrained),
which degrades suppression quality but keeps the pipeline functional.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_SR         = 16_000
_N_FFT      = 512
_HOP        = 160
_WIN        = 512
_HF_REPO    = "Xiaobin-Rong/gtcrn"
_HF_FILE    = "gtcrn_lite.pth"


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

class _TCNBlock(nn.Module):
    """Gated dilated depthwise temporal conv block."""

    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.dw = nn.Conv1d(
            channels, channels * 2, kernel_size=3,
            dilation=dilation, padding=dilation, groups=channels,
        )
        self.pw = nn.Conv1d(channels * 2, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dw(x)
        a, b = h.chunk(2, dim=1)
        return self.pw(torch.tanh(a) * torch.sigmoid(b)) + x


class _GTCRN(nn.Module):
    """
    GTCRN-lite.

    Data flow:
        (B, 2, T, 257)                         — real + imag STFT
        → encoder  → (B, 64, T, 33)
        → reshape  → (B, T, 2112)
        → in_proj  → (B, T, 256)
        → TCN      → (B, T, 256)
        → GRU      → (B, T, 256)
        → out_proj → (B, T, 2112)
        → reshape  → (B, 64, T, 33)
        → decoder  → (B, 2, T, 257)            — complex ratio mask
    """

    _ENC_CH  = [2, 16, 32, 64]
    _HIDDEN  = 256
    _DILS    = [1, 2, 4, 8, 16]
    _ENC_F   = 33      # freq bins after 3× stride-2 conv on 257 bins

    def __init__(self) -> None:
        super().__init__()

        # Encoder: 257→129→65→33 along freq axis
        enc_layers: list[nn.Module] = []
        for i in range(len(self._ENC_CH) - 1):
            ic, oc = self._ENC_CH[i], self._ENC_CH[i + 1]
            enc_layers += [
                nn.Conv2d(ic, oc, (1, 3), stride=(1, 2), padding=(0, 1)),
                nn.BatchNorm2d(oc),
                nn.PReLU(),
            ]
        self.encoder = nn.Sequential(*enc_layers)

        flat = self._ENC_CH[-1] * self._ENC_F   # 64 × 33 = 2112
        self.in_proj  = nn.Linear(flat, self._HIDDEN)
        self.tcn      = nn.Sequential(*[_TCNBlock(self._HIDDEN, d) for d in self._DILS])
        self.gru      = nn.GRU(self._HIDDEN, self._HIDDEN, num_layers=2, batch_first=True)
        self.out_proj = nn.Linear(self._HIDDEN, flat)

        # Decoder: mirrors encoder (33→65→129→257)
        dec_ch = list(reversed(self._ENC_CH))
        dec_layers: list[nn.Module] = []
        for i in range(len(dec_ch) - 1):
            ic, oc = dec_ch[i], dec_ch[i + 1]
            dec_layers.append(
                nn.ConvTranspose2d(ic, oc, (1, 3), stride=(1, 2), padding=(0, 1))
            )
            if i < len(dec_ch) - 2:          # no BN/act after the last layer
                dec_layers += [nn.BatchNorm2d(oc), nn.PReLU()]
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, T, F = x.shape

        h = self.encoder(x)                                     # (B, 64, T, 33)
        h = h.permute(0, 2, 1, 3).reshape(B, T, -1)            # (B, T, 2112)
        h = self.in_proj(h)                                     # (B, T, 256)

        h = self.tcn(h.permute(0, 2, 1)).permute(0, 2, 1)      # TCN on (B, 256, T)
        h, _ = self.gru(h)                                      # (B, T, 256)

        h = self.out_proj(h)                                    # (B, T, 2112)
        h = h.reshape(B, T, self._ENC_CH[-1], self._ENC_F)
        h = h.permute(0, 2, 1, 3)                              # (B, 64, T, 33)

        h = self.decoder(h)                                     # (B, 2, T, ~257)
        return h[..., :F]                                       # exact freq trim


# ---------------------------------------------------------------------------
# Public wrapper
# ---------------------------------------------------------------------------

class NoiseSuppressor:
    """
    Wraps GTCRN-lite for frame-level noise suppression.

    process() is synchronous and blocking — call via asyncio.to_thread.
    """

    def __init__(self, model_dir: str) -> None:
        self._dir    = Path(model_dir) / "gtcrn"
        self._model: _GTCRN | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._window = torch.hann_window(_WIN)

        try:
            model = _GTCRN()
            weights_path = self._dir / _HF_FILE
            if not weights_path.exists():
                self._download(weights_path)

            if weights_path.exists():
                sd = torch.load(weights_path, map_location="cpu", weights_only=True)
                model.load_state_dict(sd)
                logger.info("GTCRN: weights loaded from %s", weights_path)
            else:
                logger.warning(
                    "GTCRN: no pretrained weights found — "
                    "noise suppression will run with random weights"
                )

            model.eval()
            self._model = model.to(self._device)
        except Exception as exc:
            logger.warning("NoiseSuppressor init failed (%s) — passthrough mode", exc)

    @property
    def available(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------

    def _download(self, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id=_HF_REPO,
                filename=_HF_FILE,
                local_dir=str(dest.parent),
            )
            logger.info("GTCRN: weights downloaded to %s", dest)
        except Exception as exc:
            logger.warning("GTCRN: weight download failed (%s)", exc)

    # ------------------------------------------------------------------

    def process(self, pcm_bytes: bytes) -> bytes:
        """
        Denoise 16-bit LE PCM at 16 kHz mono.
        Returns the same format. Falls back to the input on any error.
        """
        if self._model is None or len(pcm_bytes) < _N_FFT * 2:
            return pcm_bytes

        try:
            audio = (
                np.frombuffer(pcm_bytes, dtype=np.int16)
                .astype(np.float32) / 32768.0
            )
            wave = torch.from_numpy(audio).unsqueeze(0)   # (1, N)
            win  = self._window

            stft = torch.stft(wave, _N_FFT, _HOP, _WIN, win, return_complex=True)
            # stft: (1, 257, T)

            # Stack real + imag → (1, 2, T, 257)
            spec = torch.stack([stft.real, stft.imag], dim=1).permute(0, 1, 3, 2)

            with torch.no_grad():
                mask = self._model(spec.to(self._device)).cpu()

            # Apply complex ratio mask
            nr, ni = spec[:, 0], spec[:, 1]
            mr, mi = mask[:, 0], mask[:, 1]
            clean  = torch.complex(nr * mr - ni * mi, nr * mi + ni * mr)
            clean  = clean.permute(0, 2, 1)                # (1, F, T)

            out_wave = torch.istft(clean, _N_FFT, _HOP, _WIN, win, length=len(audio))
            out_i16  = (
                out_wave.squeeze(0).clamp(-1.0, 1.0).numpy() * 32767.0
            ).astype(np.int16)
            return out_i16.tobytes()

        except Exception as exc:
            logger.warning("GTCRN inference error (%s) — returning original audio", exc)
            return pcm_bytes
