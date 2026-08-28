"""PSNR/SSIM formulas copied faithfully from the challenge-adapted baseline."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _create_window(window_size: int, channel: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    gauss = torch.exp(-(coords**2) / (2 * (1.5**2)))
    gauss = gauss / gauss.sum()
    kernel_2d = torch.outer(gauss, gauss).unsqueeze(0).unsqueeze(0)
    return kernel_2d.expand(channel, 1, window_size, window_size).contiguous()


def masked_psnr(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Baseline-compatible masked PSNR for [N,C,H,W] inputs in [0,1]."""

    mask3 = mask.repeat(1, pred.shape[1], 1, 1)
    mse = (((pred - gt) ** 2) * mask3).sum() / mask3.sum().clamp_min(1.0)
    return -10.0 * torch.log10(mse + eps)


def masked_ssim(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    window_size: int = 11,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Baseline-compatible SSIM map with strict masked averaging.

    Like the adapted Endo-4DGS ``metrics.py``, the Gaussian statistics are not
    mask-normalized; the final SSIM map alone is averaged through the mask.
    """

    channel = pred.shape[1]
    window = _create_window(window_size, channel, pred.device, pred.dtype)
    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(gt, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(gt * gt, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * gt, window, padding=window_size // 2, groups=channel) - mu1_mu2
    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + eps
    )
    mask3 = mask.repeat(1, channel, 1, 1)
    return (ssim_map * mask3).sum() / mask3.sum().clamp_min(1.0)
