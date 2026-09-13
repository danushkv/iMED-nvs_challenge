#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import faulthandler
faulthandler.enable()
import numpy as np
import random
import os, sys
import torch
import cv2
from random import randint
import traceback
from utils.loss_utils import l1_loss, l2_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, build_rotation
import uuid
from tqdm import tqdm
import torch.nn.functional as F
from utils.image_utils import psnr, ssim
from utils.loss_utils import mae_loss
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from torch.utils.data import DataLoader
from utils.timer import Timer
from utils.loader_utils import FineSampler, get_stamp_list
from utils.scene_utils import render_training_image
from utils.loss_utils import GradL1Loss, confidence_loss, TV_loss
from utils.graphics_utils import get_pseudo_normal
from time import time
import copy
import open3d as o3d
import csv


to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


# ===== METHOD2: METRIC DEPTH =====
def _method2_depth_valid_mask(pred_depth, gt_depth, tissue_mask):
    """Return [B,1,H,W] validity for metric source-depth supervision."""

    if tissue_mask is None:
        tissue_valid = torch.ones_like(pred_depth, dtype=torch.bool)
    else:
        tissue_valid = tissue_mask
        if tissue_valid.ndim == pred_depth.ndim - 1:
            tissue_valid = tissue_valid.unsqueeze(1)
        if tissue_valid.shape != pred_depth.shape:
            tissue_valid = tissue_valid.expand_as(pred_depth)
        tissue_valid = tissue_valid.bool()
    return (
        tissue_valid
        & torch.isfinite(pred_depth)
        & torch.isfinite(gt_depth)
        & (pred_depth > 0)
        & (gt_depth > 0)
    )


def _method2_metric_depth_loss(pred_depth, gt_depth, valid_mask, mode, huber_beta):
    """Compute an unweighted strict-valid metric depth loss in millimetres."""

    pred_valid = pred_depth[valid_mask]
    gt_valid = gt_depth[valid_mask]
    if pred_valid.numel() == 0:
        # Preserve a differentiable zero when no source depth is valid.
        return pred_depth.sum() * 0.0
    if mode == "metric_l1":
        return torch.abs(pred_valid - gt_valid).mean()
    if mode == "metric_huber":
        if huber_beta <= 0:
            raise ValueError("depth_huber_beta must be positive")
        return F.smooth_l1_loss(
            pred_valid,
            gt_valid,
            beta=huber_beta,
            reduction="mean",
        )
    raise ValueError(f"Unsupported metric depth mode: {mode}")


@torch.no_grad()
def _method2_depth_diagnostic_stats(pred_depth, gt_depth, valid_mask):
    abs_error = torch.abs(pred_depth - gt_depth)
    valid_error = abs_error[valid_mask]
    valid_gt = gt_depth[valid_mask]
    if valid_error.numel() == 0:
        return {
            "mean_abs_depth_error_mm": 0.0,
            "median_abs_depth_error_mm": 0.0,
            "mean_relative_depth_error": 0.0,
            "valid_depth_fraction": 0.0,
        }
    return {
        "mean_abs_depth_error_mm": valid_error.mean().item(),
        "median_abs_depth_error_mm": valid_error.median().item(),
        "mean_relative_depth_error": (
            valid_error / (torch.abs(valid_gt) + 1e-6)
        ).mean().item(),
        "valid_depth_fraction": valid_mask.float().mean().item(),
    }


@torch.no_grad()
def _method2_save_depth_debug_images(
    model_path,
    stage,
    iteration,
    pred_depth,
    gt_depth,
    valid_mask,
    rendered_rgb,
    gt_rgb,
):
    """Save source-only B1 diagnostics with shared metric-depth scaling."""

    output_dir = os.path.join(model_path, "depth_diagnostics", stage)
    os.makedirs(output_dir, exist_ok=True)
    prefix = os.path.join(output_dir, f"{iteration:06d}")

    pred = pred_depth[0, 0].detach().float().cpu().numpy()
    gt = gt_depth[0, 0].detach().float().cpu().numpy()
    valid = valid_mask[0, 0].detach().cpu().numpy().astype(bool)
    abs_error = np.abs(pred - gt)

    if valid.any():
        depth_scale = max(float(np.max(gt[valid])), 1e-6)
        error_scale = max(float(np.max(abs_error[valid])), 1e-6)
    else:
        depth_scale = 1.0
        error_scale = 1.0

    def save_gray(path, values):
        image = np.clip(values * 255.0, 0.0, 255.0).astype(np.uint8)
        cv2.imwrite(path, image)

    save_gray(prefix + "_gt_depth.png", np.where(valid, gt / depth_scale, 0.0))
    save_gray(prefix + "_rendered_depth.png", np.where(valid, pred / depth_scale, 0.0))
    save_gray(prefix + "_abs_depth_error.png", np.where(valid, abs_error / error_scale, 0.0))
    save_gray(prefix + "_valid_mask.png", valid.astype(np.float32))

    for suffix, tensor in (("rgb_gt", gt_rgb[0]), ("rgb_render", rendered_rgb[0])):
        rgb = tensor.detach().float().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
        bgr = np.ascontiguousarray(rgb[..., ::-1] * 255.0).astype(np.uint8)
        cv2.imwrite(prefix + f"_{suffix}.png", bgr)


def _method2_append_depth_diagnostics(model_path, row):
    path = os.path.join(model_path, "depth_diagnostics.csv")
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _method2_scalar(value):
    if torch.is_tensor(value):
        return value.detach().item()
    return float(value)


# ===== METHOD2: TOOL MASK =====
def _method2_effective_tissue_mask(tissue_mask, dilation_pixels):
    """Dilate excluded source-tool pixels by a radius measured in pixels.

    The input and output use the baseline convention: True/1 means tissue is
    valid for training, while False/0 means an excluded tool pixel.
    """

    if tissue_mask is None:
        return None
    if dilation_pixels < 0:
        raise ValueError("tool_mask_dilation must be non-negative")

    tissue = tissue_mask.bool()
    if dilation_pixels == 0:
        return tissue

    original_ndim = tissue.ndim
    if original_ndim == 2:
        tissue_4d = tissue.unsqueeze(0).unsqueeze(0)
    elif original_ndim == 3:
        tissue_4d = tissue.unsqueeze(1)
    elif original_ndim == 4 and tissue.shape[1] == 1:
        tissue_4d = tissue
    else:
        raise ValueError(f"Unsupported tissue-mask shape: {tuple(tissue.shape)}")

    radius = int(dilation_pixels)
    kernel_size = 2 * radius + 1
    dilated_tool = F.max_pool2d(
        (~tissue_4d).float(),
        kernel_size=kernel_size,
        stride=1,
        padding=radius,
    ) > 0
    effective_tissue = ~dilated_tool

    if original_ndim == 2:
        return effective_tissue[0, 0]
    if original_ndim == 3:
        return effective_tissue[:, 0]
    return effective_tissue


def _method2_masked_l1_valid_mean(network_output, gt, mask):
    """Mean absolute error over finite valid elements, not the whole image."""

    valid = mask.bool()
    if valid.ndim == network_output.ndim - 1:
        valid = valid.unsqueeze(1)
    if valid.shape != network_output.shape:
        valid = valid.expand_as(network_output)
    valid = valid & torch.isfinite(network_output) & torch.isfinite(gt)
    if not valid.any():
        return network_output.sum() * 0.0
    return torch.abs(network_output[valid] - gt[valid]).mean()


@torch.no_grad()
def _method2_save_tool_mask_debug(
    model_path,
    stage,
    base_tissue_mask,
    effective_tissue_mask,
):
    """Save source-only B2 mask diagnostics for the first sampled frame."""

    output_dir = os.path.join(model_path, "tool_mask_diagnostics")
    os.makedirs(output_dir, exist_ok=True)
    base = base_tissue_mask[0].detach().bool().cpu().numpy()
    effective = effective_tissue_mask[0].detach().bool().cpu().numpy()
    newly_excluded = base & ~effective
    for suffix, values in (
        ("base_tissue", base),
        ("effective_tissue", effective),
        ("newly_excluded_boundary", newly_excluded),
    ):
        cv2.imwrite(
            os.path.join(output_dir, f"{stage}_{suffix}.png"),
            values.astype(np.uint8) * 255,
        )


# ===== METHOD2: APPEARANCE CORRECTION =====
class _Method2PerFrameAppearance(torch.nn.Module):
    """Source-training-only affine RGB correction indexed by frame uid."""

    def __init__(self, frame_count):
        super().__init__()
        self.rgb_scale = torch.nn.Parameter(torch.ones(frame_count, 3))
        self.rgb_bias = torch.nn.Parameter(torch.zeros(frame_count, 3))

    def forward(self, rendered_rgb, frame_uid):
        if frame_uid < 0 or frame_uid >= self.rgb_scale.shape[0]:
            raise IndexError(
                f"Appearance frame uid {frame_uid} outside "
                f"[0, {self.rgb_scale.shape[0]})"
            )
        scale = self.rgb_scale[frame_uid]
        bias = self.rgb_bias[frame_uid]
        corrected = rendered_rgb * scale[:, None, None] + bias[:, None, None]
        return corrected, scale, bias


def _method2_append_appearance_diagnostics(model_path, row):
    path = os.path.join(model_path, "appearance_diagnostics.csv")
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


@torch.no_grad()
def _method2_save_appearance_parameters(model_path, appearance_model, hyper):
    """Persist source-frame diagnostics; render.py intentionally ignores it."""

    scale = appearance_model.rgb_scale.detach().float().cpu()
    bias = appearance_model.rgb_bias.detach().float().cpu()
    torch.save(
        {
            "rgb_scale": scale,
            "rgb_bias": bias,
            "training_only": True,
            "appearance_lr": getattr(hyper, "appearance_lr", 0.001),
            "appearance_reg_weight": getattr(hyper, "appearance_reg_weight", 0.01),
        },
        os.path.join(model_path, "appearance_correction_training_only.pth"),
    )
    with open(
        os.path.join(model_path, "appearance_parameters.csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        fieldnames = [
            "frame_uid",
            "scale_r",
            "scale_g",
            "scale_b",
            "bias_r",
            "bias_g",
            "bias_b",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for frame_uid in range(scale.shape[0]):
            writer.writerow(
                {
                    "frame_uid": frame_uid,
                    "scale_r": scale[frame_uid, 0].item(),
                    "scale_g": scale[frame_uid, 1].item(),
                    "scale_b": scale[frame_uid, 2].item(),
                    "bias_r": bias[frame_uid, 0].item(),
                    "bias_g": bias[frame_uid, 1].item(),
                    "bias_b": bias[frame_uid, 2].item(),
                }
            )



def scene_reconstruction(mp, opt, hyper, pipe, testing_iterations, saving_iterations, 
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, stage, tb_writer, train_iter, timer,
                         appearance_model=None, appearance_optimizer=None):
    first_iter = 0
    use_depth = pipe.use_depth
    use_smooth = pipe.use_smooth
    use_normal = pipe.use_normal
    use_confidence = pipe.use_confidence
    configured_depth_loss = getattr(hyper, "depth_loss", "normalized")
    if configured_depth_loss not in {"normalized", "metric_l1", "metric_huber"}:
        raise ValueError(f"Unsupported depth_loss: {configured_depth_loss}")
    tool_aware_loss = getattr(hyper, "tool_aware_loss", False)
    tool_mask_dilation = getattr(hyper, "tool_mask_dilation", 0)
    if tool_mask_dilation < 0:
        raise ValueError("tool_mask_dilation must be non-negative")
    use_appearance = appearance_model is not None
    print('Init with pretrain:', mp.use_pretrain)
    print('Use depth l1:', use_depth)
    print('Depth loss mode:', configured_depth_loss)
    print('Tool-aware strict valid mean:', tool_aware_loss)
    print('Tool-mask dilation radius:', tool_mask_dilation)
    print('Training-only appearance correction:', use_appearance)
    print('Use smooth:', use_smooth)
    print('Use normal:', use_normal)
    print('Use confidence:', use_confidence)
    
    gaussians.training_setup(opt)
    if checkpoint:
        # breakpoint()
        if stage == "coarse" and stage not in checkpoint:
            print("start from fine stage, skip coarse stage.")
            # process is in the coarse stage, but start from fine stage
            return
        if stage in checkpoint: 
            (model_params, first_iter) = torch.load(checkpoint)
            gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if mp.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    viewpoint_stack = None
    final_iter = train_iter
    
    progress_bar = tqdm(range(first_iter, final_iter), desc="Training")
    first_iter += 1
    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()

    if not viewpoint_stack and not opt.dataloader:
        # dnerf's branch
        viewpoint_stack = [i for i in train_cams]
        temp_list = copy.deepcopy(viewpoint_stack)
    # op dataloader: False
    batch_size = opt.batch_size
    if opt.dataloader:
        viewpoint_stack = scene.getTrainCameras()
        if opt.custom_sampler is not None:
            sampler = FineSampler(viewpoint_stack)
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,sampler=sampler,num_workers=32,collate_fn=list)
            random_loader = False
        else:
            viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=batch_size,shuffle=True,num_workers=32,collate_fn=list)
            random_loader = True
        loader = iter(viewpoint_stack_loader)
    
    
    # dynerf, zerostamp_init
    # breakpoint()
    if stage == "coarse" and opt.zerostamp_init:
        load_in_memory = True
        # batch_size = 4
        temp_list = get_stamp_list(viewpoint_stack,0)
        viewpoint_stack = temp_list.copy()
    else:
        load_in_memory = False 
    
    count = 0
    for iteration in range(first_iter, final_iter+1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer, ts = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer, stage=stage, \
                        cam_type=scene.dataset_type)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, mp.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # dynerf's branch
        if opt.dataloader and not load_in_memory:
            try:
                viewpoint_cams = next(loader)
            except StopIteration:
                print("reset dataloader into random dataloader.")
                if not random_loader:
                    viewpoint_stack_loader = DataLoader(viewpoint_stack, batch_size=opt.batch_size,shuffle=True,num_workers=32,collate_fn=list)
                    random_loader = True
                loader = iter(viewpoint_stack_loader)
        else:
            idx = 0
            viewpoint_cams = []

            while idx < batch_size :    
                viewpoint_cam = viewpoint_stack.pop(randint(0,len(viewpoint_stack)-1))
                if not viewpoint_stack :
                    viewpoint_stack =  temp_list.copy()
                viewpoint_cams.append(viewpoint_cam)
                idx +=1
            if len(viewpoint_cams) == 0:
                continue
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        images = []
        gt_images = []
        appearance_scales = []
        appearance_biases = []
        # read depth
        gt_depths = []
        depths = []
        masks = []
        radii_list = []
        visibility_filter_list = []
        viewspace_point_tensor_list = []
        gs_normal = []
        if use_confidence:
            confidences = []
        # ranking_loss = EdgeguidedRankingLoss(point_pairs=5000, alpha=0.8)
        # pc_loss = PearsonCorrCoef().cuda()
        # silog_loss = SILogLoss()
        grad_loss = GradL1Loss()
        avg_filter = torch.nn.AvgPool2d(2)
        jump = False
        # SSI_loss = ScaleAndShiftInvariantLoss()
        for viewpoint_cam in viewpoint_cams:
            # pc = viewpoint_cam.pc.cuda()
            # src_pcd = o3d.t.geometry.PointCloud(o3d.core.Tensor(gaussians.get_xyz.detach().cpu().numpy(), o3d.core.float32))
            # tgt_pcd = o3d.t.geometry.PointCloud(o3d.core.Tensor(pc.detach().cpu().numpy(), o3d.core.float32))
            # result = o3d.t.pipelines.registration.icp(src_pcd, tgt_pcd, 5)
            # corr_set = result.correspondence_set.numpy()
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, stage=stage, \
                cam_type=scene.dataset_type, iteration=count)
            image, viewspace_point_tensor, radii, depth = \
                render_pkg["render"], render_pkg["viewspace_points"], \
                render_pkg["radii"], render_pkg['depth']
            if use_appearance:
                image, appearance_scale, appearance_bias = appearance_model(
                    image,
                    int(viewpoint_cam.uid),
                )
                appearance_scales.append(appearance_scale)
                appearance_biases.append(appearance_bias)
            visibility_filter = radii>0
            gt_depth = viewpoint_cam.depth.cuda()
            mask = viewpoint_cam.mask

            if scene.dataset_type!="PanopticSports":
                gt_image = viewpoint_cam.original_image.cuda()
            else:
                gt_image  = viewpoint_cam['image'].cuda()

            if mask is not None:
                mask = mask.cuda()
                masks.append(mask.unsqueeze(0))
            
            images.append(image.unsqueeze(0))
            dep_mask = torch.logical_and(gt_depth > 0, depth > 0)
            gt_depth = gt_depth * dep_mask
            depth = depth * dep_mask
            depths.append(depth.unsqueeze(0))
            if use_normal:
                gs_normal.append(render_pkg['normal'].unsqueeze(0))
            if use_confidence:
                confidences.append(render_pkg['confidence'].unsqueeze(0))
                
            gt_depths.append(gt_depth.unsqueeze(0))
            gt_images.append(gt_image.unsqueeze(0))
            radii_list.append(radii.unsqueeze(0))
            visibility_filter_list.append(visibility_filter.unsqueeze(0))
            viewspace_point_tensor_list.append(viewspace_point_tensor)

        radii = torch.cat(radii_list,0).max(dim=0).values
        visibility_filter = torch.cat(visibility_filter_list).any(dim=0)
        
        if len(masks) != 0:
            base_mask_tensor = torch.cat(masks, 0).bool()
            mask_tensor = _method2_effective_tissue_mask(
                base_mask_tensor,
                tool_mask_dilation,
            )
        else:
            base_mask_tensor = None
            mask_tensor = None

        if iteration == 1 and mask_tensor is not None and (
            tool_aware_loss or tool_mask_dilation > 0
        ):
            _method2_save_tool_mask_debug(
                mp.model_path,
                stage,
                base_mask_tensor,
                mask_tensor,
            )
            print(
                "[METHOD2][TOOL]",
                {
                    "stage": stage,
                    "strict_valid_mean": bool(tool_aware_loss),
                    "dilation_radius_pixels": int(tool_mask_dilation),
                    "base_tissue_fraction": base_mask_tensor.float().mean().item(),
                    "effective_tissue_fraction": mask_tensor.float().mean().item(),
                },
            )
        
        image_tensor = torch.cat(images,0) * mask_tensor
        depth_tensor = torch.cat(depths, 0) * mask_tensor
        gt_image_tensor = torch.cat(gt_images,0) * mask_tensor
        gt_depth_tensor = torch.cat(gt_depths, 0) * mask_tensor
        
        if use_normal:
            gs_normal = torch.cat(gs_normal, 0) * mask_tensor
        if use_confidence:
            confidences = torch.cat(confidences, 0) * mask_tensor
        
        # Loss
        if tool_aware_loss:
            Ll1 = _method2_masked_l1_valid_mean(
                image_tensor,
                gt_image_tensor,
                mask_tensor,
            )
        else:
            # Exact B0 reduction: masked errors are averaged over the complete
            # image tensor rather than divided by the number of tissue pixels.
            Ll1 = l1_loss(image_tensor, gt_image_tensor, mask_tensor.unsqueeze(0))
        psnr_ = psnr(image_tensor, gt_image_tensor).mean().double()
        # norm
        if use_depth:
            # ===== METHOD2: METRIC DEPTH =====
            depth_loss_mode = configured_depth_loss
            # Do not alter hyper.depth_weight here: B0 also uses it for the
            # gradient and TV terms. B1 sweeps only primary depth supervision.
            depth_weight = getattr(
                hyper,
                "primary_depth_weight",
                hyper.depth_weight,
            )
            if depth_loss_mode == "normalized":
                pred_depth_normalized = depth_tensor / (depth_tensor.max() + 1e-6)
                gt_depth_normalized = gt_depth_tensor / (gt_depth_tensor.max() + 1e-6)
                if tool_aware_loss:
                    normalized_depth_valid = _method2_depth_valid_mask(
                        depth_tensor,
                        gt_depth_tensor,
                        mask_tensor,
                    )
                    depth_loss_unweighted = _method2_masked_l1_valid_mean(
                        pred_depth_normalized,
                        gt_depth_normalized,
                        normalized_depth_valid,
                    )
                else:
                    # Exact B0 equation. Predicted and GT depth are
                    # independently max-normalized before masked L1.
                    depth_loss_unweighted = l1_loss(
                        pred_depth_normalized,
                        gt_depth_normalized,
                        mask=mask_tensor.unsqueeze(0),
                    )
                metric_depth_valid = None
            elif depth_loss_mode in {"metric_l1", "metric_huber"}:
                metric_depth_valid = _method2_depth_valid_mask(
                    depth_tensor,
                    gt_depth_tensor,
                    mask_tensor,
                )
                depth_loss_unweighted = _method2_metric_depth_loss(
                    depth_tensor,
                    gt_depth_tensor,
                    metric_depth_valid,
                    mode=depth_loss_mode,
                    huber_beta=getattr(hyper, "depth_huber_beta", 5.0),
                )
            else:
                raise ValueError(f"Unsupported depth_loss: {depth_loss_mode}")
            depth_loss = depth_loss_unweighted * depth_weight
            loss = Ll1 + depth_loss 
        else:
            loss = Ll1

        if use_appearance:
            appearance_scale_tensor = torch.stack(appearance_scales, dim=0)
            appearance_bias_tensor = torch.stack(appearance_biases, dim=0)
            appearance_regularization = (
                ((appearance_scale_tensor - 1.0) ** 2).mean()
                + (appearance_bias_tensor ** 2).mean()
            )
            appearance_loss = (
                appearance_regularization
                * getattr(hyper, "appearance_reg_weight", 0.01)
            )
            loss += appearance_loss
        
        if use_smooth:
            grad_weight=hyper.depth_weight
            sm_loss = (grad_loss(depth_tensor, gt_depth_tensor, mask=mask_tensor)) * grad_weight
            loss += sm_loss
            
        if use_normal:
            normal_weight = hyper.normal_weight
            pseudo_normal=get_pseudo_normal(gt_depth_tensor, mask_tensor.unsqueeze(0))
            pseudo_normal = F.interpolate(pseudo_normal, gs_normal.shape[2:4])
            normal_loss = mae_loss(gs_normal, pseudo_normal, mask_tensor)*normal_weight
            loss += normal_loss
            
        if use_confidence:
            un_img_weight = hyper.un_img_weight
            un_dep_weight = hyper.un_dep_weight
            confidence_loss_img = confidence_loss(gt_image_tensor, image_tensor, \
                confidences, mask_tensor.unsqueeze(0))*un_img_weight
            confidence_loss_dep = confidence_loss(gt_depth_tensor/gt_depth_tensor.max(), \
                depth_tensor/depth_tensor.max(), confidences, mask_tensor.unsqueeze(0))*un_dep_weight
            loss += confidence_loss_img
            loss += confidence_loss_dep
            
        if stage == "fine" and hyper.time_smoothness_weight != 0:
            tv_loss = gaussians.compute_regulation(hyper.time_smoothness_weight, \
                hyper.l1_time_planes, hyper.plane_tv_weight) + \
                    +(TV_loss(depth_tensor)+TV_loss(image_tensor))*hyper.depth_weight
            loss += tv_loss
            
        if opt.lambda_dssim != 0:
            ssim_loss = ssim(image_tensor, gt_image_tensor)
            loss += opt.lambda_dssim * (1.0-ssim_loss)

        # ===== METHOD2: METRIC DEPTH DIAGNOSTICS =====
        diagnostics_interval = getattr(hyper, "depth_diagnostics_interval", 0)
        if use_depth and diagnostics_interval > 0 and (
            iteration == 1 or iteration % diagnostics_interval == 0
        ):
            if metric_depth_valid is None:
                metric_depth_valid = _method2_depth_valid_mask(
                    depth_tensor,
                    gt_depth_tensor,
                    mask_tensor,
                )
            depth_stats = _method2_depth_diagnostic_stats(
                depth_tensor,
                gt_depth_tensor,
                metric_depth_valid,
            )
            diagnostic_row = {
                "stage": stage,
                "iteration": iteration,
                "depth_loss_mode": depth_loss_mode,
                "depth_weight": depth_weight,
                "auxiliary_depth_weight": hyper.depth_weight,
                "depth_huber_beta_mm": getattr(hyper, "depth_huber_beta", 5.0),
                "L_rgb": _method2_scalar(Ll1),
                "L_depth_unweighted": _method2_scalar(depth_loss_unweighted),
                "L_depth_weighted": _method2_scalar(depth_loss),
                "L_smooth": _method2_scalar(sm_loss) if use_smooth else 0.0,
                "L_normal": _method2_scalar(normal_loss) if use_normal else 0.0,
                "L_confidence_image": (
                    _method2_scalar(confidence_loss_img) if use_confidence else 0.0
                ),
                "L_confidence_depth": (
                    _method2_scalar(confidence_loss_dep) if use_confidence else 0.0
                ),
                "L_ssim_weighted": (
                    _method2_scalar(opt.lambda_dssim * (1.0 - ssim_loss))
                    if opt.lambda_dssim != 0
                    else 0.0
                ),
                "L_total": _method2_scalar(loss),
                **depth_stats,
            }
            _method2_append_depth_diagnostics(mp.model_path, diagnostic_row)
            _method2_save_depth_debug_images(
                mp.model_path,
                stage,
                iteration,
                depth_tensor,
                gt_depth_tensor,
                metric_depth_valid,
                image_tensor,
                gt_image_tensor,
            )
            print("[METHOD2][DEPTH]", diagnostic_row)

        # ===== METHOD2: APPEARANCE DIAGNOSTICS =====
        appearance_diagnostics_interval = getattr(
            hyper,
            "appearance_diagnostics_interval",
            0,
        )
        if use_appearance and appearance_diagnostics_interval > 0 and (
            iteration == 1 or iteration % appearance_diagnostics_interval == 0
        ):
            scale_mean = appearance_scale_tensor.detach().mean(dim=0)
            bias_mean = appearance_bias_tensor.detach().mean(dim=0)
            appearance_row = {
                "stage": stage,
                "iteration": iteration,
                "appearance_lr": getattr(hyper, "appearance_lr", 0.001),
                "appearance_reg_weight": getattr(
                    hyper,
                    "appearance_reg_weight",
                    0.01,
                ),
                "L_rgb_corrected": _method2_scalar(Ll1),
                "L_appearance_regularization": _method2_scalar(
                    appearance_regularization
                ),
                "L_appearance_weighted": _method2_scalar(appearance_loss),
                "sample_scale_r": scale_mean[0].item(),
                "sample_scale_g": scale_mean[1].item(),
                "sample_scale_b": scale_mean[2].item(),
                "sample_bias_r": bias_mean[0].item(),
                "sample_bias_g": bias_mean[1].item(),
                "sample_bias_b": bias_mean[2].item(),
            }
            _method2_append_appearance_diagnostics(mp.model_path, appearance_row)
            print("[METHOD2][APPEARANCE]", appearance_row)
            
        loss.backward()
            
        if torch.isnan(loss).any():
            print("loss is nan,end training, reexecv program now.")
            loss_dict = {"Loss": f"{Ll1.item():.{4}f}",
                        "psnr": f"{psnr_:.{2}f}"}
            if stage == "fine" and hyper.time_smoothness_weight != 0:
                loss_dict['tv_loss'] = f"{tv_loss:.{4}f}"
            if use_depth:
                loss_dict['depth'] = f"{depth_loss:.{4}f}"
            if use_smooth:
                loss_dict["Smooth"] = f"{sm_loss:.{4}f}"
            if opt.lambda_dssim != 0:
                loss_dict["ssim"] = f"{ssim_loss:.{4}f}"
            if use_normal:
                loss_dict["Norm"] = f"{normal_loss:.{4}f}"
            if use_confidence:
                loss_dict["Un_img"] = f"{confidence_loss_img:.{4}f}"
                loss_dict["Un_dep"] = f"{confidence_loss_dep:.{4}f}"
            print(loss_dict)
            
            os.execv(sys.executable, [sys.executable] + sys.argv)
        viewspace_point_tensor_grad = torch.zeros_like(viewspace_point_tensor)
        for idx in range(0, len(viewspace_point_tensor_list)):
            if jump:
                viewspace_point_tensor_grad = viewspace_point_tensor_grad
            else:
                viewspace_point_tensor_grad = viewspace_point_tensor_grad + viewspace_point_tensor_list[idx].grad
        iter_end.record()
        torch.cuda.synchronize()

        with torch.no_grad():
            # Progress bar
            total_point = gaussians._xyz.shape[0]
            if iteration % 10 == 0:
                string_dict = {"Loss": f"{Ll1.item():.{4}f}",
                                        "psnr": f"{psnr_:.{2}f}"}
                if stage == 'fine':
                    string_dict['tv'] = f"{tv_loss:.{4}f}"
                if use_depth:
                    string_dict["Dep"] = f"{depth_loss:.{4}f}"
                    if depth_loss_mode != "normalized":
                        string_dict["DepRaw"] = f"{depth_loss_unweighted:.{4}f}"
                if use_smooth:
                    string_dict["Sm"] = f"{sm_loss:.{4}f}"
                if use_normal:
                    string_dict["Norm"] = f"{normal_loss:.{4}f}"
                if use_confidence:
                    string_dict["Un_img"] = f"{confidence_loss_img:.{4}f}"
                    string_dict["Un_dep"] = f"{confidence_loss_dep:.{4}f}"
                if use_appearance:
                    string_dict["App"] = f"{appearance_loss:.{4}f}"
                    
                progress_bar.set_postfix(string_dict)
                    
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # timer.pause()
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), \
                testing_iterations, scene, render, [pipe, background], stage, scene.dataset_type)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, stage)
            if mp.render_process:
                if (iteration < 1000 and iteration % 10 == 9) \
                    or (iteration < 3000 and iteration % 50 == 49) \
                        or (iteration < 60000 and iteration %  100 == 99) :
                        render_training_image(scene, gaussians, [test_cams[iteration%len(test_cams)]], \
                            render, pipe, background, stage+"test", iteration,timer.get_elapsed_time(),scene.dataset_type)
            timer.start()
            
            # Densification
            if iteration < opt.densify_until_iter :
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                if stage == "coarse":
                    opacity_threshold = opt.opacity_threshold_coarse
                    densify_threshold = opt.densify_grad_threshold_coarse
                else:    
                    opacity_threshold = opt.opacity_threshold_fine_init - iteration*(opt.opacity_threshold_fine_init - \
                        opt.opacity_threshold_fine_after)/(opt.densify_until_iter)  
                    densify_threshold = opt.densify_grad_threshold_fine_init - iteration*(opt.densify_grad_threshold_fine_init \
                        - opt.densify_grad_threshold_after)/(opt.densify_until_iter )  
                
                if  iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<360000:
                    # print('Densify')
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold, 5, 5, scene.model_path, iteration, stage)
                    # gaussians.densify(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                
                if  iteration > opt.pruning_from_iter and iteration % opt.pruning_interval == 0 and gaussians.get_xyz.shape[0]>200000:
                    # print('Prune')
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None

                    gaussians.prune(densify_threshold, opacity_threshold, scene.cameras_extent, size_threshold)
                    
                # if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 :
                if iteration % opt.densification_interval == 0 and gaussians.get_xyz.shape[0]<360000 and opt.add_point:
                    # print('Grow')
                    gaussians.grow(5,5,scene.model_path,iteration,stage)
                    # torch.cuda.empty_cache()
                if iteration % opt.opacity_reset_interval == 0:
                    # print("reset opacity")
                    gaussians.reset_opacity()
                    
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                if appearance_optimizer is not None:
                    appearance_optimizer.step()
                    appearance_optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" +f"_{stage}_" + str(iteration) + ".pth")

def training(model_param, hyper, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, expname):
    # first_iter = 0
    tb_writer = prepare_output_and_logger(expname)
    gaussians = GaussianModel(model_param.sh_degree, hyper)
    model_param.model_path = args.model_path
    timer = Timer()
    scene = Scene(model_param, gaussians, load_coarse=None)
    appearance_model = None
    appearance_optimizer = None
    if getattr(hyper, "appearance_correction", False):
        if checkpoint:
            print(
                "[METHOD2][APPEARANCE] Warning: source appearance parameters "
                "are initialized from identity when resuming a Gaussian checkpoint."
            )
        appearance_model = _Method2PerFrameAppearance(
            len(scene.getTrainCameras())
        ).cuda()
        appearance_optimizer = torch.optim.Adam(
            appearance_model.parameters(),
            lr=getattr(hyper, "appearance_lr", 0.001),
        )
    timer.start()
    scene_reconstruction(model_param, opt, hyper, pipe, testing_iterations, saving_iterations,
                             checkpoint_iterations, checkpoint, debug_from,
                             gaussians, scene, "coarse", tb_writer,
                             opt.coarse_iterations, timer, appearance_model,
                             appearance_optimizer)
    scene_reconstruction(model_param, opt, hyper, pipe, testing_iterations, saving_iterations,
                         checkpoint_iterations, checkpoint, debug_from,
                         gaussians, scene, "fine", tb_writer, opt.iterations,
                         timer, appearance_model, appearance_optimizer)
    if appearance_model is not None:
        _method2_save_appearance_parameters(
            model_param.model_path,
            appearance_model,
            hyper,
        )

def prepare_output_and_logger(expname):    
    if not args.model_path:
        unique_str = expname

        args.model_path = os.path.join("./output/", unique_str)
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, stage, dataset_type):
    if tb_writer:
        tb_writer.add_scalar(f'{stage}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{stage}/train_loss_patchestotal_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{stage}/iter_time', elapsed, iteration)
        
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        # 
        validation_configs = ({'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(10, 5000, 299)]},)#,
        use_imed_overlap_eval = "imed" in args.source_path.lower()

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                lpips_score_test = 0.0
                ssim_test = 0.0
                overlap_mask = None

                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians,stage=stage, cam_type=dataset_type, *renderArgs)["render"], 0.0, 1.0)
                    if dataset_type == "PanopticSports":
                        gt_image = torch.clamp(viewpoint["image"].to("cuda"), 0.0, 1.0)
                    else:
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    mask = viewpoint.mask
                    if mask is not None:
                        mask = mask.cuda()
                    if use_imed_overlap_eval:
                        if overlap_mask is None:
                            overlap_mask = (image.sum(dim=0) > 0).float()
                        mask = overlap_mask if mask is None else (mask.float() * overlap_mask).clamp(0.0, 1.0)

                    if mask is not None:
                        image = image * mask
                        gt_image = gt_image * mask
                    
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])  
                lpips_score_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])  

                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} lpips_score {}".format(iteration, config['name'], l1_test, psnr_test, ssim_test, lpips_score_test))
                if tb_writer:
                    tb_writer.add_scalar(stage + "/"+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(stage+"/"+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(stage+"/"+config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(stage+"/"+config['name'] + '/loss_viewpoint - lpips_score', lpips_score_test, iteration)


        if tb_writer:
            tb_writer.add_scalar(f'{stage}/total_points', scene.gaussians.get_xyz.shape[0], iteration)
            tb_writer.add_scalar(f'{stage}/deformation_rate', scene.gaussians._deformation_table.sum()/scene.gaussians.get_xyz.shape[0], iteration)
        
        torch.cuda.empty_cache()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    setup_seed(6666)
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[500*i for i in range(100)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1000, 3000, 4000, 5000, 6000, 7_000, 9000, 10000, 12000, 14000, 20000, 30_000, 45000, 60000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--expname", type=str, default = "")
    parser.add_argument("--configs", type=str, default = "")
    
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), args.test_iterations,
            args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.expname)
    # All done
    print("\nTraining complete.")
