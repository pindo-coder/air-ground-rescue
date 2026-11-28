from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
tools_dir = str(ROOT / "tools")
if tools_dir not in sys.path:
    sys.path.insert(0, tools_dir)

from reproduce_uddf_ictis2025 import UGDIP, UDDFModel, apply_paper_eq3_fog


class TestUddfReproduction(unittest.TestCase):
    def test_paper_eq3_fog_monotonically_reduces_contrast(self):
        rng = np.random.default_rng(7)
        bgr = rng.integers(10, 245, size=(120, 160, 3), dtype=np.uint8)
        f0 = apply_paper_eq3_fog(bgr, level_i=0, A=0.5)
        f5 = apply_paper_eq3_fog(bgr, level_i=5, A=0.5)
        f9 = apply_paper_eq3_fog(bgr, level_i=9, A=0.5)
        self.assertEqual(f0.shape, bgr.shape)
        self.assertEqual(f9.dtype, np.uint8)
        self.assertGreater(float(np.std(f0)), float(np.std(f5)))
        self.assertGreater(float(np.std(f5)), float(np.std(f9)))

    def test_ugdip_shape_range_and_gradients(self):
        ugdip = UGDIP()
        ugdip.train()
        x = torch.rand(2, 3, 240, 320, requires_grad=True)
        enh, u_ale, unc_loss, gate_w = ugdip(x, n_samples=3)
        self.assertEqual(tuple(enh.shape), (2, 3, 240, 320))
        self.assertEqual(tuple(u_ale.shape), (2, 240, 320))
        self.assertTrue(bool((enh >= 0.0).all() and (enh <= 1.0).all()))
        self.assertTrue(torch.allclose(gate_w.sum(dim=-1), torch.ones(2), atol=1e-4))
        total = enh.mean() + unc_loss
        total.backward()
        self.assertIsNotNone(ugdip.encoder.fc[1].weight.grad)
        self.assertIsNotNone(ugdip.gating.gate_fc.weight.grad)
        self.assertIsNotNone(ugdip.uqn.head_logvar.weight.grad)

    def test_uddf_model_train_and_mc_dropout_inference(self):
        yolov8_w = ROOT / "models/yolov8n.pt"
        if not yolov8_w.is_file():
            self.skipTest("models/yolov8n.pt not present")
        model = UDDFModel(yolov8_w, dropout_p=0.2, lambda_unc=0.1)
        imgs = torch.rand(2, 3, 480, 640)
        batch = dict(
            img=imgs,
            batch_idx=torch.tensor([0.0, 1.0], dtype=torch.float32),
            cls=torch.tensor([[0.0], [0.0]], dtype=torch.float32),
            bboxes=torch.tensor([[0.5, 0.5, 0.06, 0.06], [0.4, 0.4, 0.05, 0.05]], dtype=torch.float32),
        )
        loss, info = model.forward_train(imgs, batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("loss_unc", info)
        loss.backward()

        bgr = np.random.default_rng(11).integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
        preds, meta = model.predict_bgr(bgr, conf_thr=0.01, mc_passes=3, max_unc=1.0)
        self.assertEqual(meta["enhanced_bgr"].shape, bgr.shape)
        self.assertEqual(meta["aleatoric_map"].shape, (480, 640))
        for item in preds:
            self.assertEqual(len(item["bbox"]), 4)
            self.assertGreaterEqual(item["u_total"], 0.0)
            self.assertLessEqual(item["u_total"], 1.0)


if __name__ == "__main__":
    unittest.main()
