"""Faithful reproduction of ICTIS 2025 UDDF (UGDIP + UQN + YOLOv8n + MC-Dropout).

Paper reference:
  Z. Gong, C. Qian, and Z. Chen, "Uncertainty-Driven YOLO for Object Detection
  in Adverse Weather," in Proc. IEEE 8th Int. Conf. on Transportation
  Information and Safety (ICTIS 2025), Granada, Spain, Jul. 2025, pp. 1882-1888,
  doi: 10.1109/ICTIS68762.2025.11214987.

Components implemented:
  1. Eq. (3) atmospheric scattering synthetic fog generator (`apply_paper_eq3_fog`).
  2. `VisionEncoder` (5 conv layers 3->64->...->1024 -> GAP -> 128-d latent) +
     `SevenGatingBlocks` (WB, Gamma, Identity, Defogging, Sharpening, Tone, Contrast, Eq. 1).
  3. `UQN`: Reparameterized pixel-wise Gaussian (mu_p, sigma_p) for aleatoric uncertainty.
  4. `UDDFModel`: End-to-end UGDIP + YOLOv8n with Eq. (2) uncertainty-regularized loss
     and 10-pass MC-Dropout stochastic clustering for epistemic uncertainty filtering.
  5. `UDDFPersonDetector`: Drop-in `.detect(bgr, depth)` adapter for `AerialVictimTracker`.
"""
from pathlib import Path
from types import SimpleNamespace
import copy
import json
import math
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.nms import non_max_suppression

ROOT = Path(__file__).resolve().parents[1]
INPUT_HW = (480, 640)  # Exact 4:3 UAV frame size and multiple of 32 (zero letterbox padding)


def apply_paper_eq3_fog(bgr, level_i=4, A=0.5):
    """Apply synthetic fog using exact Eq. (3) from Gong et al. (ICTIS 2025, p. 1885).

    I(x) = J(x) * exp(-beta * d(x)) + A * (1 - exp(-beta * d(x)))
    where d(x) = max(0, -0.04 * rho + sqrt(max(row, col))), beta = 0.01 * i + 0.05.
    """
    img = np.asarray(bgr, dtype=np.float32) / 255.0
    h, w = img.shape[:2]
    ys = np.arange(h, dtype=np.float32)[:, None] - (h / 2.0)
    xs = np.arange(w, dtype=np.float32)[None, :] - (w / 2.0)
    rho = np.sqrt(ys * ys + xs * xs)
    d = np.maximum(0.0, -0.04 * rho + math.sqrt(float(max(h, w))))
    beta = 0.01 * float(level_i) + 0.05
    t = np.exp(-beta * d)[:, :, None]
    out = img * t + float(A) * (1.0 - t)
    return np.clip(np.round(out * 255.0), 0.0, 255.0).astype(np.uint8)


class VisionEncoder(nn.Module):
    """5-layer CNN encoder (channels 64->1024) + GAP + FC -> 128-d latent embedding."""

    def __init__(self, latent_dim=128):
        super().__init__()
        channels = (3, 64, 128, 256, 512, 1024)
        layers = []
        for idx in range(5):
            layers.append(nn.Conv2d(channels[idx], channels[idx + 1], kernel_size=3, stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(channels[idx + 1]))
            layers.append(nn.LeakyReLU(0.1, inplace=True))
            if idx < 4:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, latent_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        # Downsample inside encoder so 5-stage 3x3 convs up to 1024 channels run fast on CPU.
        small = F.interpolate(x, size=(128, 128), mode="bilinear", align_corners=False)
        feat = self.pool(self.features(small))
        return self.fc(feat)


def _zero_init_linear(in_dim, out_dim, bias_val=0.0):
    layer = nn.Linear(in_dim, out_dim)
    nn.init.zeros_(layer.weight)
    nn.init.constant_(layer.bias, bias_val)
    return layer


def _minmax_norm(u):
    u_min = u.amin(dim=(1, 2, 3), keepdim=True)
    u_max = u.amax(dim=(1, 2, 3), keepdim=True)
    span = u_max - u_min
    normed = (u - u_min) / (span + 1e-5)
    return torch.where(span >= 1e-3, normed, u)


class SevenGatingBlocks(nn.Module):
    """Seven gated differentiable image-processing blocks (Eq. 1 of ICTIS 2025)."""

    NAMES = ("WB", "G", "I", "DF", "S", "T", "C")

    def __init__(self, latent_dim=128):
        super().__init__()
        self.gate_fc = nn.Linear(latent_dim, 7)
        nn.init.zeros_(self.gate_fc.weight)
        # Bias strongly toward Identity (GB_I, index 2) at step 0 so warm-started YOLOv8n is preserved.
        with torch.no_grad():
            self.gate_fc.bias.copy_(torch.tensor([0.0, 0.0, 2.2, 0.0, 0.0, 0.0, 0.0]))

        self.fc_wb = _zero_init_linear(latent_dim, 3)
        self.fc_g = _zero_init_linear(latent_dim, 1)
        self.fc_df = _zero_init_linear(latent_dim, 1)
        self.fc_s = _zero_init_linear(latent_dim, 1)
        self.fc_t = _zero_init_linear(latent_dim, 1)
        self.fc_c = _zero_init_linear(latent_dim, 1)

    def forward(self, x, latent):
        b = x.shape[0]
        weights = F.softmax(self.gate_fc(latent), dim=-1)  # (B, 7)

        # 1. GB_WB: White Balance
        gains = (0.7 + 0.6 * torch.sigmoid(self.fc_wb(latent))).view(b, 3, 1, 1)
        f_wb = torch.clamp(gains * x, 0.0, 1.0)

        # 2. GB_G: Gamma Correction
        gamma = (0.5 + 1.5 * torch.sigmoid(self.fc_g(latent))).view(b, 1, 1, 1)
        f_g = torch.pow(torch.clamp(x, 1e-4, 1.0), gamma)

        # 3. GB_I: Identity
        f_i = x

        # 4. GB_DF: Differentiable Dark-Channel Defogging
        dark_min_c = torch.amin(x, dim=1, keepdim=True)
        dark_channel = -F.max_pool2d(-dark_min_c, kernel_size=7, stride=1, padding=3)
        a_est = torch.clamp(dark_channel.amax(dim=(2, 3), keepdim=True), 0.35, 0.98)
        omega = (0.25 + 0.65 * torch.sigmoid(self.fc_df(latent))).view(b, 1, 1, 1)
        trans = torch.clamp(1.0 - omega * dark_channel / a_est, 0.20, 1.0)
        f_df = torch.clamp((x - a_est) / trans + a_est, 0.0, 1.0)

        # 5. GB_S: Unsharp-Mask Sharpening
        alpha = (1.2 * torch.sigmoid(self.fc_s(latent))).view(b, 1, 1, 1)
        blurred = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        f_s = torch.clamp(x + alpha * (x - blurred), 0.0, 1.0)

        # 6. GB_T: Smooth S-Curve Tone Adjustment
        tau = (0.6 + 1.8 * torch.sigmoid(self.fc_t(latent))).view(b, 1, 1, 1)
        half_tanh = torch.tanh(0.5 * tau)
        f_t = torch.clamp((torch.tanh(tau * (x - 0.5)) + half_tanh) / (2.0 * half_tanh + 1e-6), 0.0, 1.0)

        # 7. GB_C: Contrast Adjustment
        c_val = (-0.3 + 1.0 * torch.sigmoid(self.fc_c(latent))).view(b, 1, 1, 1)
        x_mean = x.mean(dim=(1, 2, 3), keepdim=True)
        f_c = torch.clamp((1.0 + c_val) * (x - x_mean) + x_mean, 0.0, 1.0)

        ops = (f_wb, f_g, f_i, f_df, f_s, f_t, f_c)
        z0 = torch.zeros_like(x)
        for idx, op in enumerate(ops):
            z0 = z0 + weights[:, idx].view(b, 1, 1, 1) * _minmax_norm(op)
        return torch.clamp(z0, 0.0, 1.0), weights


class UQN(nn.Module):
    """Reparameterized aleatoric uncertainty quantification network (Fig. 2 of ICTIS 2025)."""

    def __init__(self, hidden_ch=32):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, hidden_ch, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.head_mu = nn.Conv2d(hidden_ch, 3, kernel_size=3, padding=1)
        nn.init.zeros_(self.head_mu.weight)
        nn.init.zeros_(self.head_mu.bias)

        self.head_logvar = nn.Conv2d(hidden_ch, 3, kernel_size=3, padding=1)
        nn.init.zeros_(self.head_logvar.weight)
        nn.init.constant_(self.head_logvar.bias, -4.5)

    def forward(self, z0, n_samples=None):
        feat = self.backbone(z0)
        mu = torch.clamp(z0 + 0.25 * torch.tanh(self.head_mu(feat)), 0.0, 1.0)
        raw_logvar = self.head_logvar(feat)
        logvar = torch.clamp(raw_logvar, -6.5, -1.0)
        sigma = torch.exp(0.5 * logvar)
        # Analytical predicted spatial uncertainty map (computed from unclamped logvar, stretched p02..p98).
        with torch.no_grad():
            s_raw = torch.exp(0.5 * torch.clamp(raw_logvar, -12.0, -1.0)).mean(dim=1)  # (B, H, W)
            flat = s_raw.flatten(start_dim=1)
            q02 = torch.quantile(flat, 0.02, dim=1).view(-1, 1, 1)
            q98 = torch.quantile(flat, 0.98, dim=1).view(-1, 1, 1)
            self.last_sigma_map = torch.clamp((s_raw - q02) / (q98 - q02 + 1e-6), 0.0, 1.0)

        k_samples = (4 if self.training else 6) if n_samples is None else int(n_samples)
        samples = torch.stack(
            [torch.clamp(mu + torch.randn_like(sigma) * sigma, 0.0, 1.0) for _ in range(k_samples)],
            dim=0,
        )  # (K, B, 3, H, W)
        var_map = samples.var(dim=0, unbiased=False).mean(dim=1)  # (B, H, W)
        v_max = var_map.amax(dim=(1, 2), keepdim=True)
        u_ale_map = var_map / (v_max + 1e-6)

        if self.training:
            enhanced = samples[0]
            unc_loss = torch.mean(
                0.5 * torch.exp(-logvar) * (z0.detach() - mu).pow(2) + 0.5 * (logvar + 6.5) * 0.15
            )
        else:
            enhanced = mu
            unc_loss = z0.new_tensor(0.0)
        return enhanced, u_ale_map, unc_loss


class UGDIP(nn.Module):
    """Uncertainty-Guided Degradation Image Processing front-end (Encoder + 7 GBs + UQN)."""

    def __init__(self, latent_dim=128):
        super().__init__()
        self.encoder = VisionEncoder(latent_dim=latent_dim)
        self.gating = SevenGatingBlocks(latent_dim=latent_dim)
        self.uqn = UQN()

    def forward(self, x, n_samples=None):
        latent = self.encoder(x)
        z0, gate_weights = self.gating(x, latent)
        enhanced, u_ale_map, unc_loss = self.uqn(z0, n_samples=n_samples)
        return enhanced, u_ale_map, unc_loss, gate_weights


class UDDFModel(nn.Module):
    """End-to-end UDDF: UGDIP + YOLOv8n with backbone/head MC-Dropout hooks."""

    def __init__(self, yolov8_weights, dropout_p=0.2, lambda_unc=0.1):
        super().__init__()
        self.ugdip = UGDIP()
        yolo_wrap = YOLO(str(yolov8_weights))
        self.detector_net = copy.deepcopy(yolo_wrap.model)
        for p in self.detector_net.parameters():
            p.requires_grad_(True)
        self.detector_net.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        self.criterion = v8DetectionLoss(self.detector_net)
        self.lambda_unc = float(lambda_unc)
        self.dropout_p = float(dropout_p)
        self.mc_dropout_active = False

        # Learnable combination logits for total uncertainty = w_ale * U_ale + w_epi * U_epi
        self.unc_logits = nn.Parameter(torch.zeros(2))
        self._hook_handles = []
        self._register_mc_dropout_hooks()

    def _register_mc_dropout_hooks(self):
        def hook_fn(_module, _inp, output):
            if self.mc_dropout_active and isinstance(output, torch.Tensor):
                return F.dropout2d(output, p=self.dropout_p, training=True)
            return output

        for idx in (4, 6, 8):
            if idx < len(self.detector_net.model):
                self._hook_handles.append(self.detector_net.model[idx].register_forward_hook(hook_fn))

        detect_head = self.detector_net.model[-1]
        for branch_list_name in ("cv2", "cv3"):
            branch_list = getattr(detect_head, branch_list_name, None)
            if branch_list is None:
                continue
            for branch in branch_list:
                # Apply dropout before final 1x1 prediction conv inside Detect head.
                target_layer = branch[-2] if hasattr(branch, "__len__") and len(branch) >= 2 else branch
                self._hook_handles.append(target_layer.register_forward_hook(hook_fn))

    def set_detector_bn_eval(self):
        for m in self.detector_net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward_train(self, rgb_01, batch_targets):
        self.ugdip.train()
        self.detector_net.train()
        self.set_detector_bn_eval()
        self.mc_dropout_active = True

        enhanced, u_ale_map, unc_loss, gate_weights = self.ugdip(rgb_01)
        preds = self.detector_net(enhanced)
        loss_vec, loss_items = self.criterion(preds, batch_targets)
        b_size = max(1, rgb_01.shape[0])
        loss_obj = loss_vec.sum() / float(b_size)
        loss_total = loss_obj + self.lambda_unc * unc_loss
        return loss_total, dict(
            loss_total=float(loss_total.detach()),
            loss_obj=float(loss_obj.detach()),
            loss_unc=float(unc_loss.detach()),
            box_loss=float(loss_items["box_loss"].detach()),
            cls_loss=float(loss_items["cls_loss"].detach()),
            dfl_loss=float(loss_items["dfl_loss"].detach()),
            gate_mean=gate_weights.detach().mean(dim=0).cpu().tolist(),
        )

    @staticmethod
    def _box_iou(box_a, box_b):
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
        area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 1e-6 else 0.0

    @torch.no_grad()
    def predict_bgr(self, bgr, conf_thr=0.45, mc_passes=10, max_unc=0.72, iou_nms=0.45):
        """Run UGDIP once + T-pass MC-Dropout clustering on a BGR image."""
        h_orig, w_orig = bgr.shape[:2]
        in_h, in_w = INPUT_HW
        if (h_orig, w_orig) != (in_h, in_w):
            resized = cv2.resize(bgr, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = bgr
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        self.ugdip.eval()
        self.detector_net.eval()

        enhanced, u_ale_map, _, gate_w = self.ugdip(x, n_samples=6)
        ale_np = u_ale_map[0].cpu().numpy()
        sigma_np = (
            self.ugdip.uqn.last_sigma_map[0].cpu().numpy()
            if getattr(self.ugdip.uqn, "last_sigma_map", None) is not None
            else ale_np
        )
        enh_rgb = np.clip(np.round(enhanced[0].permute(1, 2, 0).cpu().numpy() * 255.0), 0.0, 255.0).astype(np.uint8)
        enh_bgr = cv2.cvtColor(enh_rgb, cv2.COLOR_RGB2BGR)

        sx = float(w_orig) / float(in_w)
        sy = float(h_orig) / float(in_h)
        unc_w = F.softmax(self.unc_logits, dim=0).cpu().numpy()
        w_ale, w_epi = float(unc_w[0]), float(unc_w[1])

        def _box_ale_unc(x1, y1, x2, y2):
            ix1 = max(0, min(in_w - 1, int(math.floor(x1))))
            iy1 = max(0, min(in_h - 1, int(math.floor(y1))))
            ix2 = max(ix1 + 1, min(in_w, int(math.ceil(x2))))
            iy2 = max(iy1 + 1, min(in_h, int(math.ceil(y2))))
            return float(ale_np[iy1:iy2, ix1:ix2].mean())

        if int(mc_passes) <= 1:
            self.mc_dropout_active = False
            raw = self.detector_net(enhanced)
            pred = raw[0] if isinstance(raw, (tuple, list)) else raw
            dets = non_max_suppression(pred, conf_thres=float(conf_thr), iou_thres=float(iou_nms))[0]
            results = []
            for row in dets.cpu().numpy():
                bx1, by1, bx2, by2, sc, _cls = map(float, row[:6])
                u_a = _box_ale_unc(bx1, by1, bx2, by2)
                results.append(dict(
                    bbox=[bx1 * sx, by1 * sy, bx2 * sx, by2 * sy],
                    score=sc,
                    u_ale=u_a,
                    u_epi=0.0,
                    u_total=u_a,
                    views_mc=1,
                ))
            return results, dict(enhanced_bgr=enh_bgr, aleatoric_map=ale_np, sigma_map=sigma_np, gate_weights=gate_w[0].cpu().tolist())

        # T > 1 stochastic MC-Dropout forward passes & IoU clustering across passes.
        t_passes = int(mc_passes)
        self.mc_dropout_active = True
        pass_conf = min(0.28, max(0.18, float(conf_thr) * 0.65))
        clusters = []  # each: dict(boxes=[], scores=[])

        try:
            for _ in range(t_passes):
                raw = self.detector_net(enhanced)
                pred = raw[0] if isinstance(raw, (tuple, list)) else raw
                dets = non_max_suppression(pred, conf_thres=pass_conf, iou_thres=float(iou_nms))[0]
                if len(dets) == 0:
                    continue
                used_clusters = set()
                for row in dets.cpu().numpy():
                    box = [float(row[0]), float(row[1]), float(row[2]), float(row[3])]
                    sc = float(row[4])
                    best_c, best_iou = None, 0.45
                    for ci, cl in enumerate(clusters):
                        if ci in used_clusters:
                            continue
                        mean_box = np.mean(cl["boxes"], axis=0).tolist()
                        iou = self._box_iou(box, mean_box)
                        if iou >= best_iou:
                            best_iou = iou
                            best_c = ci
                    if best_c is None:
                        clusters.append(dict(boxes=[box], scores=[sc]))
                        used_clusters.add(len(clusters) - 1)
                    else:
                        clusters[best_c]["boxes"].append(box)
                        clusters[best_c]["scores"].append(sc)
                        used_clusters.add(best_c)
        finally:
            self.mc_dropout_active = False

        min_hits = max(2, int(math.ceil(0.3 * t_passes)))
        pre_kept = []
        for cl in clusters:
            m = len(cl["boxes"])
            if m < min_hits:
                continue
            boxes_arr = np.asarray(cl["boxes"], dtype=float)
            scores_arr = np.asarray(cl["scores"], dtype=float)
            mean_box = boxes_arr.mean(axis=0)
            p_hit = float(scores_arr.mean())
            support = float(m) / float(t_passes)
            # Effective detection confidence scales hit confidence with stochastic consistency.
            p_eff = p_hit * (0.35 + 0.65 * support)

            box_std_norm = float(np.sqrt(np.var(boxes_arr, axis=0).mean()) / 24.0)
            all_scores = np.zeros(t_passes, dtype=float)
            all_scores[:m] = scores_arr
            prob_var = float(np.var(all_scores))
            p_clip = float(np.clip(p_eff, 1e-5, 1.0 - 1e-5))
            entropy = float(-(p_clip * math.log(p_clip) + (1.0 - p_clip) * math.log(1.0 - p_clip)) / math.log(2.0))
            u_epi = float(np.clip(0.40 * box_std_norm + 0.30 * (1.0 - support) + 0.20 * entropy + 0.10 * (prob_var * 4.0), 0.0, 1.0))

            u_ale = _box_ale_unc(*mean_box)
            u_tot = float(np.clip(w_ale * u_ale + w_epi * u_epi, 0.0, 1.0))

            if p_eff < float(conf_thr) or u_tot > float(max_unc):
                continue
            pre_kept.append(dict(
                bbox=[float(mean_box[0] * sx), float(mean_box[1] * sy), float(mean_box[2] * sx), float(mean_box[3] * sy)],
                score=float(p_eff),
                u_ale=float(u_ale),
                u_epi=float(u_epi),
                u_total=float(u_tot),
                views_mc=int(m),
            ))

        # Final NMS across aggregated clusters
        if pre_kept:
            b_cv = [[r["bbox"][0], r["bbox"][1], r["bbox"][2] - r["bbox"][0], r["bbox"][3] - r["bbox"][1]] for r in pre_kept]
            s_cv = [r["score"] for r in pre_kept]
            idx_keep = np.asarray(cv2.dnn.NMSBoxes(b_cv, s_cv, float(conf_thr), float(iou_nms))).reshape(-1)
            pre_kept = [pre_kept[i] for i in idx_keep]

        return pre_kept, dict(enhanced_bgr=enh_bgr, aleatoric_map=ale_np, sigma_map=sigma_np, gate_weights=gate_w[0].cpu().tolist())


class UDDFPersonDetector:
    """Adapter matching `PersonDetector.detect(bgr, depth)` for `AerialVictimTracker`."""

    def __init__(self, checkpoint_path, conf_thr=0.45, mc_passes=10, max_unc=0.72):
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        yolov8_base = ckpt.get("yolov8_base") or str(ROOT / "models/yolov8n.pt")
        self.model = UDDFModel(yolov8_base, dropout_p=ckpt.get("dropout_p", 0.2), lambda_unc=ckpt.get("lambda_unc", 0.1))
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.conf_thr = float(conf_thr)
        self.mc_passes = int(mc_passes)
        self.max_unc = float(max_unc)

    def detect(self, bgr, depth=None):
        preds, _ = self.model.predict_bgr(
            bgr, conf_thr=self.conf_thr, mc_passes=self.mc_passes, max_unc=self.max_unc,
        )
        out = []
        for item in preds:
            x1, y1, x2, y2 = item["bbox"]
            out.append(([int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))], float(item["score"]), None))
        return out


class UDDFVictimDataset(Dataset):
    """Loads YOLO-format dataset (`images/<split>`, `labels/<split>`) with online fog/flip/jitter."""

    def __init__(self, dataset_root, split="train", augment=True, fog_prob=0.25, seed=71):
        self.root = Path(dataset_root)
        self.split = split
        self.augment = augment
        self.fog_prob = float(fog_prob)
        self.rng = random.Random(seed)
        self.images = sorted((self.root / "images" / split).glob("*.jpg"))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        lbl_path = self.root / "labels" / self.split / f"{img_path.stem}.txt"
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise FileNotFoundError(img_path)
        if bgr.shape[:2] != INPUT_HW:
            bgr = cv2.resize(bgr, (INPUT_HW[1], INPUT_HW[0]), interpolation=cv2.INTER_LINEAR)

        boxes = []
        if lbl_path.is_file():
            for line in lbl_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    _, cx, cy, bw, bh = map(float, parts)
                    boxes.append([cx, cy, bw, bh])
        boxes_arr = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)

        if self.augment:
            if self.rng.random() < self.fog_prob:
                bgr = apply_paper_eq3_fog(bgr, level_i=self.rng.randint(0, 6), A=0.5)
            if self.rng.random() < 0.5:
                bgr = np.ascontiguousarray(np.fliplr(bgr))
                if len(boxes_arr):
                    boxes_arr[:, 0] = 1.0 - boxes_arr[:, 0]
            if self.rng.random() < 0.5:
                bgr = np.ascontiguousarray(np.flipud(bgr))
                if len(boxes_arr):
                    boxes_arr[:, 1] = 1.0 - boxes_arr[:, 1]
            # Mild brightness/contrast jitter matching Section IV-A [0.8, 1.2]
            alpha = self.rng.uniform(0.85, 1.18)
            beta = self.rng.uniform(-12.0, 12.0)
            bgr = np.clip(bgr.astype(np.float32) * alpha + beta, 0.0, 255.0).astype(np.uint8)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return img_t, torch.from_numpy(boxes_arr)


def uddf_collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    b_idxs = []
    bboxes = []
    for i, (_, box_t) in enumerate(batch):
        if len(box_t):
            b_idxs.append(torch.full((len(box_t),), float(i), dtype=torch.float32))
            bboxes.append(box_t)
    if bboxes:
        batch_idx = torch.cat(b_idxs, dim=0)
        bbox_cat = torch.cat(bboxes, dim=0)
        cls_cat = torch.zeros((len(bbox_cat), 1), dtype=torch.float32)
    else:
        batch_idx = torch.zeros((0,), dtype=torch.float32)
        bbox_cat = torch.zeros((0, 4), dtype=torch.float32)
        cls_cat = torch.zeros((0, 1), dtype=torch.float32)
    return imgs, dict(img=imgs, batch_idx=batch_idx, cls=cls_cat, bboxes=bbox_cat)


def _xywh_norm_to_xyxy_px(boxes_norm, w, h):
    out = []
    for cx, cy, bw, bh in boxes_norm:
        out.append([(cx - 0.5 * bw) * w, (cy - 0.5 * bh) * h, (cx + 0.5 * bw) * w, (cy + 0.5 * bh) * h])
    return out


def compute_box_metrics(pred_list, gt_list, iou_thr=0.5):
    """Compute Precision, Recall, F1, AP@50, and Pearson correlation of uncertainty vs (1 - IoU)."""
    all_scores = []
    all_tp = []
    total_gt = sum(len(g) for g in gt_list)
    unc_vals = []
    err_vals = []

    flat = []
    for img_i, preds in enumerate(pred_list):
        for p in preds:
            flat.append((img_i, p))
    flat.sort(key=lambda item: -item[1]["score"])

    matched_gt = [set() for _ in range(len(gt_list))]
    for img_i, pred in flat:
        pbox = pred["bbox"]
        gts = gt_list[img_i]
        best_j, best_iou = None, 0.0
        for j, gbox in enumerate(gts):
            if j in matched_gt[img_i]:
                continue
            iou = UDDFModel._box_iou(pbox, gbox)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        # Best IoU against ANY gt for uncertainty-error audit
        any_iou = max((UDDFModel._box_iou(pbox, g) for g in gts), default=0.0)
        if "u_total" in pred:
            unc_vals.append(float(pred["u_total"]))
            err_vals.append(float(1.0 - any_iou))

        all_scores.append(float(pred["score"]))
        if best_j is not None and best_iou >= iou_thr:
            matched_gt[img_i].add(best_j)
            all_tp.append(1)
        else:
            all_tp.append(0)

    tp_cnt = int(sum(all_tp))
    fp_cnt = len(all_tp) - tp_cnt
    prec = tp_cnt / len(all_tp) if all_tp else 0.0
    rec = tp_cnt / total_gt if total_gt else 0.0
    f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    # Standard interpolated AP@50
    ap50 = 0.0
    if all_tp and total_gt > 0:
        tp_cum = np.cumsum(all_tp, dtype=float)
        fp_cum = np.cumsum([1 - x for x in all_tp], dtype=float)
        recalls = tp_cum / float(total_gt)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([1.0], precisions, [0.0]))
        for k in range(len(mpre) - 2, -1, -1):
            mpre[k] = max(mpre[k], mpre[k + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap50 = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

    unc_corr = None
    if len(unc_vals) >= 4 and np.std(unc_vals) > 1e-6 and np.std(err_vals) > 1e-6:
        unc_corr = float(np.corrcoef(unc_vals, err_vals)[0, 1])

    return dict(
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1=round(f1, 4),
        ap50=round(ap50, 4),
        tp=tp_cnt,
        fp=fp_cnt,
        gt=total_gt,
        unc_err_corr=round(unc_corr, 4) if unc_corr is not None else None,
    )


def evaluate_uddf_on_val(model, dataset_root, conf_thr=0.45, mc_passes=1, max_unc=0.80):
    val_ds = UDDFVictimDataset(dataset_root, split="val", augment=False)
    preds, gts = [], []
    for img_path in val_ds.images:
        bgr = cv2.imread(str(img_path))
        h, w = bgr.shape[:2]
        lbl_path = val_ds.root / "labels/val" / f"{img_path.stem}.txt"
        gt_norm = []
        if lbl_path.is_file():
            for line in lbl_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    gt_norm.append(list(map(float, parts[1:])))
        gts.append(_xywh_norm_to_xyxy_px(gt_norm, w, h))
        det_items, _ = model.predict_bgr(bgr, conf_thr=conf_thr, mc_passes=mc_passes, max_unc=max_unc)
        preds.append(det_items)
    return compute_box_metrics(preds, gts), preds, gts


def calibrate_uddf_on_val(model, dataset_root, mc_passes=10):
    """Calibrate (conf_thr, max_unc) strictly on Scene C val (zero Scene D access)."""
    val_ds = UDDFVictimDataset(dataset_root, split="val", augment=False)
    raw_preds, gts = [], []
    for img_path in val_ds.images:
        bgr = cv2.imread(str(img_path))
        h, w = bgr.shape[:2]
        lbl_path = val_ds.root / "labels/val" / f"{img_path.stem}.txt"
        gt_norm = []
        if lbl_path.is_file():
            for line in lbl_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    gt_norm.append(list(map(float, parts[1:])))
        gts.append(_xywh_norm_to_xyxy_px(gt_norm, w, h))
        det_items, _ = model.predict_bgr(bgr, conf_thr=0.22, mc_passes=mc_passes, max_unc=1.0)
        raw_preds.append(det_items)

    best = None
    for conf in (0.35, 0.40, 0.45, 0.50, 0.55):
        for max_u in (0.55, 0.65, 0.75, 0.85, 1.00):
            filtered = [[p for p in ps if p["score"] >= conf and p["u_total"] <= max_u] for ps in raw_preds]
            m = compute_box_metrics(filtered, gts)
            # Prefer precision >= 0.85 while maximizing F1 (fall back to highest F1 if none hits 0.85)
            key = (1 if m["precision"] >= 0.85 else 0, m["f1"], m["ap50"])
            if best is None or key > best[0]:
                best = (key, conf, max_u, m)
    return best[1], best[2], best[3]


def train_uddf(dataset_root, yolov8_init, output_ckpt, epochs=7, batch_size=8, lr=2.5e-4):
    """End-to-end joint training of UGDIP + YOLOv8n with Eq. (2) uncertainty loss."""
    torch.manual_seed(71)
    np.random.seed(71)
    random.seed(71)

    ds_train = UDDFVictimDataset(dataset_root, split="train", augment=True, fog_prob=0.25, seed=71)
    loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, collate_fn=uddf_collate_fn)

    model = UDDFModel(yolov8_init, dropout_p=0.2, lambda_unc=0.1)
    optimizer = torch.optim.AdamW([
        {"params": model.ugdip.parameters(), "lr": lr * 2.0, "weight_decay": 1e-4},
        {"params": [model.unc_logits], "lr": lr * 2.0, "weight_decay": 0.0},
        {"params": model.detector_net.parameters(), "lr": lr, "weight_decay": 5e-4},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)

    out_path = Path(output_ckpt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    history = []
    best_score = -1.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        stats_accum = []
        for imgs, batch in loader:
            optimizer.zero_grad()
            loss, info = model.forward_train(imgs, batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"UDDF training produced non-finite loss at epoch {epoch}: {info}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            stats_accum.append(info)
        scheduler.step()

        mean_loss = float(np.mean([s["loss_total"] for s in stats_accum]))
        mean_unc = float(np.mean([s["loss_unc"] for s in stats_accum]))
        mean_gate = np.mean([s["gate_mean"] for s in stats_accum], axis=0).round(3).tolist()
        val_m, _, _ = evaluate_uddf_on_val(model, dataset_root, conf_thr=0.45, mc_passes=1, max_unc=1.0)
        elapsed = time.time() - t0
        rec = dict(epoch=epoch, elapsed_s=round(elapsed, 1), loss_total=round(mean_loss, 4),
                   loss_unc=round(mean_unc, 4), gate_weights=mean_gate, val_c=val_m)
        history.append(rec)
        print(
            f"[UDDF Train] Epoch {epoch}/{epochs} ({elapsed:.1f}s) loss={mean_loss:.3f} "
            f"(unc={mean_unc:.3f}) | Scene C val P={val_m['precision']:.3f} R={val_m['recall']:.3f} "
            f"F1={val_m['f1']:.3f} AP50={val_m['ap50']:.3f} | gate={mean_gate}",
            flush=True,
        )
        score = val_m["f1"] + 0.5 * val_m["ap50"]
        if score >= best_score:
            best_score = score
            torch.save(dict(
                model_state=copy.deepcopy(model.state_dict()),
                yolov8_base=str(yolov8_init),
                dropout_p=0.2,
                lambda_unc=0.1,
                best_epoch=epoch,
                val_c_t1=val_m,
                history=history,
            ), out_path)

    # Reload best checkpoint and calibrate MC-Dropout T=10 thresholds strictly on Scene C val
    best_ckpt = torch.load(str(out_path), map_location="cpu", weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])
    conf_cal, max_unc_cal, val_t10 = calibrate_uddf_on_val(model, dataset_root, mc_passes=10)
    best_ckpt["calibrated_conf_thr"] = conf_cal
    best_ckpt["calibrated_max_unc"] = max_unc_cal
    best_ckpt["val_c_t10"] = val_t10
    torch.save(best_ckpt, out_path)
    print(
        f"[UDDF Calibrated on Scene C] conf_thr={conf_cal}, max_unc={max_unc_cal} -> "
        f"P={val_t10['precision']:.3f}, R={val_t10['recall']:.3f}, AP50={val_t10['ap50']:.3f}, "
        f"Unc-Err Corr={val_t10['unc_err_corr']}",
        flush=True,
    )
    return out_path, best_ckpt
