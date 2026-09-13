"""Source-geometry confidence for Method 3 M3-C.

All signals are computed exclusively from the organized Endoscope2/L depth
geometry.  This module does not read images, masks, calibration files, or
target-view information.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SurfaceConfidenceMaps:
    """Confidence components on the organized source image grid."""

    combined: torch.Tensor  # [H,W], product of the three components
    depth: torch.Tensor  # [H,W], distance from the discontinuity threshold
    tangent: torch.Tensor  # [H,W], local tangent-pair stability
    viewing: torch.Tensor  # [H,W], fronto-parallel viewing reliability


def estimate_surface_confidence(
    points_source: torch.Tensor,
    tangent_u: torch.Tensor,
    tangent_v: torch.Tensor,
    normals_source: torch.Tensor,
    maximum_depth_delta: torch.Tensor,
    allowed_depth_delta: torch.Tensor,
    surface_valid: torch.Tensor,
    epsilon: float = 1e-8,
) -> SurfaceConfidenceMaps:
    """Compute conservative, deterministic source-geometry confidence.

    Definitions:

    ``C_depth = clamp(1 - max_neighbour_delta / allowed_delta, 0, 1)``

    ``C_tangent = |du x dv| / (|du| |dv| + eps)``

    ``C_viewing = |n . normalize(-X_source)|``

    The absolute viewing dot product makes the calculation independent of the
    arbitrary normal orientation.  Invalid surface neighbourhoods receive
    zero confidence and are handled by the exact MV1A bilinear fallback.
    """

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    allowed_safe = allowed_depth_delta.clamp_min(epsilon)
    depth_confidence = (1.0 - maximum_depth_delta / allowed_safe).clamp(0.0, 1.0)

    tangent_cross = torch.linalg.cross(tangent_u, tangent_v, dim=-1)
    tangent_area = torch.linalg.vector_norm(tangent_cross, dim=-1)
    tangent_scale = (
        torch.linalg.vector_norm(tangent_u, dim=-1)
        * torch.linalg.vector_norm(tangent_v, dim=-1)
    )
    tangent_confidence = (tangent_area / tangent_scale.clamp_min(epsilon)).clamp(0.0, 1.0)

    direction_to_source_camera = -points_source / torch.linalg.vector_norm(
        points_source, dim=-1, keepdim=True
    ).clamp_min(epsilon)
    viewing_confidence = (
        (normals_source * direction_to_source_camera).sum(dim=-1).abs().clamp(0.0, 1.0)
    )

    valid = surface_valid.to(dtype=torch.bool)
    depth_confidence = torch.where(valid, depth_confidence, torch.zeros_like(depth_confidence))
    tangent_confidence = torch.where(
        valid, tangent_confidence, torch.zeros_like(tangent_confidence)
    )
    viewing_confidence = torch.where(
        valid, viewing_confidence, torch.zeros_like(viewing_confidence)
    )

    depth_confidence = torch.nan_to_num(depth_confidence, nan=0.0, posinf=0.0, neginf=0.0)
    tangent_confidence = torch.nan_to_num(
        tangent_confidence, nan=0.0, posinf=0.0, neginf=0.0
    )
    viewing_confidence = torch.nan_to_num(
        viewing_confidence, nan=0.0, posinf=0.0, neginf=0.0
    )
    combined = depth_confidence * tangent_confidence * viewing_confidence
    combined = torch.where(valid, combined.clamp(0.0, 1.0), torch.zeros_like(combined))
    return SurfaceConfidenceMaps(
        combined=combined,
        depth=depth_confidence,
        tangent=tangent_confidence,
        viewing=viewing_confidence,
    )
