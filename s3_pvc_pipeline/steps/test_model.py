# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test step: evaluate the trained model on test data and log test metrics."""

import base64
import io
import random
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import LightningModule
from zenml import log_metadata, step
from zenml.types import HTMLString

from steps.dataloader import get_dataloader
from steps.utils import (
    FASHION_MNIST_IDX_TO_CLASS_NAMES,
    N_SAMPLE_IMAGES,
    plot_sample_predictions,
)


def _evaluate_model(
    model: LightningModule, test_loader: Any
) -> Tuple[Dict[str, float], Dict[int, Dict[str, int]]]:
    """Run model on test_loader. Return overall metrics and per-class counts."""
    model.eval()
    loss_fn = getattr(model, "loss_fn", torch.nn.CrossEntropyLoss())
    total_loss = 0.0
    total_correct = 0
    total = 0
    per_class: Dict[int, Dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            if next(model.parameters()).is_cuda:
                x, y = x.cuda(), y.cuda()
            logits = model(x)
            loss = loss_fn(logits, y)
            preds = logits.argmax(1)
            total_loss += loss.item() * x.size(0)
            total_correct += (preds == y).sum().item()
            total += x.size(0)
            for label, pred in zip(y.cpu().tolist(), preds.cpu().tolist()):
                per_class[label]["total"] += 1
                if label == pred:
                    per_class[label]["correct"] += 1
    overall = {
        "test_loss": total_loss / total if total else 0.0,
        "test_accuracy": 100.0 * total_correct / total if total else 0.0,
    }
    return overall, dict(per_class)


def _build_html_report(
    overall_metrics: Dict[str, float],
    per_class: Dict[int, Dict[str, int]],
    n_test: int,
    sample_image_b64: str,
) -> str:
    """Build an HTML report with metrics summary, per-class table, and sample predictions."""
    acc = overall_metrics["test_accuracy"]
    loss = overall_metrics["test_loss"]

    # Per-class accuracy rows
    class_rows = ""
    for idx in sorted(per_class.keys()):
        name = FASHION_MNIST_IDX_TO_CLASS_NAMES.get(idx, str(idx))
        c = per_class[idx]
        cls_acc = 100.0 * c["correct"] / c["total"] if c["total"] else 0.0
        bar_width = cls_acc
        bar_color = "#22c55e" if cls_acc >= 80 else "#eab308" if cls_acc >= 60 else "#ef4444"
        class_rows += f"""
        <tr>
          <td style="padding:6px 12px;border-bottom:1px solid #e5e7eb;font-weight:500;">{name}</td>
          <td style="padding:6px 12px;border-bottom:1px solid #e5e7eb;text-align:right;">{c['correct']}/{c['total']}</td>
          <td style="padding:6px 12px;border-bottom:1px solid #e5e7eb;text-align:right;">{cls_acc:.1f}%</td>
          <td style="padding:6px 12px;border-bottom:1px solid #e5e7eb;width:120px;">
            <div style="background:#f3f4f6;border-radius:4px;height:14px;overflow:hidden;">
              <div style="background:{bar_color};height:100%;width:{bar_width:.0f}%;border-radius:4px;"></div>
            </div>
          </td>
        </tr>"""

    return f"""
    <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;max-width:800px;margin:0 auto;color:#1f2937;">

      <h2 style="margin:0 0 16px 0;font-size:20px;font-weight:600;">Test Evaluation Report</h2>

      <div style="display:flex;gap:16px;margin-bottom:24px;flex-wrap:wrap;">
        <div style="flex:1;min-width:140px;background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:16px;text-align:center;">
          <div style="font-size:12px;color:#166534;text-transform:uppercase;letter-spacing:0.05em;">Accuracy</div>
          <div style="font-size:28px;font-weight:700;color:#15803d;">{acc:.2f}%</div>
        </div>
        <div style="flex:1;min-width:140px;background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:16px;text-align:center;">
          <div style="font-size:12px;color:#1e40af;text-transform:uppercase;letter-spacing:0.05em;">Loss</div>
          <div style="font-size:28px;font-weight:700;color:#1d4ed8;">{loss:.4f}</div>
        </div>
        <div style="flex:1;min-width:140px;background:#f5f3ff;border:1px solid #ddd6fe;border-radius:8px;padding:16px;text-align:center;">
          <div style="font-size:12px;color:#5b21b6;text-transform:uppercase;letter-spacing:0.05em;">Samples</div>
          <div style="font-size:28px;font-weight:700;color:#6d28d9;">{n_test:,}</div>
        </div>
      </div>

      <h3 style="margin:0 0 8px 0;font-size:16px;font-weight:600;">Per-class Accuracy</h3>
      <table style="width:100%;border-collapse:collapse;margin-bottom:24px;font-size:14px;">
        <thead>
          <tr style="background:#f9fafb;">
            <th style="padding:8px 12px;text-align:left;border-bottom:2px solid #d1d5db;">Class</th>
            <th style="padding:8px 12px;text-align:right;border-bottom:2px solid #d1d5db;">Correct</th>
            <th style="padding:8px 12px;text-align:right;border-bottom:2px solid #d1d5db;">Accuracy</th>
            <th style="padding:8px 12px;text-align:left;border-bottom:2px solid #d1d5db;"></th>
          </tr>
        </thead>
        <tbody>{class_rows}
        </tbody>
      </table>

      <h3 style="margin:0 0 8px 0;font-size:16px;font-weight:600;">Sample Predictions</h3>
      <div style="text-align:center;">
        <img src="data:image/png;base64,{sample_image_b64}" style="max-width:100%;height:auto;border-radius:8px;border:1px solid #e5e7eb;">
      </div>
    </div>"""


@step
def test_model(
    trained_model: LightningModule,
    processed_test_dir: Path,
    seed: int,
) -> Annotated[HTMLString, "test_report_visualization"]:
    """Evaluate the trained model on the test set and log metrics.

    Returns an HTML report with metrics, per-class accuracy, and sample
    predictions for the ZenML dashboard.

    Args:
        trained_model: The trained model from the train_model step.
        processed_test_dir: Path to the processed test data directory on PVC.
        seed: Random seed for reproducibility.

    Returns:
        HTML visualization with test report and sample predictions.
    """
    print("\nTesting model on test set")
    random.seed(seed)

    test_loader = get_dataloader(
        preprocessed_data_dir=processed_test_dir,
        batch_size=256,
        shuffle=False,
    )
    n_test = len(test_loader.dataset)  # type: ignore[arg-type]

    overall_metrics, per_class = _evaluate_model(trained_model, test_loader)
    test_result = {
        "n_test_samples": n_test,
        "test_loss": overall_metrics["test_loss"],
        "test_accuracy": overall_metrics["test_accuracy"],
    }

    print("  Test result")
    print("  ----------")
    print(f"    test_loss:     {overall_metrics['test_loss']:.4f}")
    print(f"    test_accuracy: {overall_metrics['test_accuracy']:.2f}%")
    print(f"    n_test_samples: {n_test}")
    print("  Test step complete\n")

    log_metadata(metadata=test_result)  # type: ignore[arg-type]

    # Randomly select samples for the visualization
    dataset = test_loader.dataset
    n_sample = min(N_SAMPLE_IMAGES, n_test)
    indices = random.sample(range(n_test), n_sample)
    images_list = []
    actual_list = []
    for idx in indices:
        x, y = dataset[idx]
        images_list.append(x.numpy())
        actual_list.append(y)
    images_np = np.stack(images_list, axis=0)
    actual_np = np.array(actual_list, dtype=np.int64)

    trained_model.eval()
    with torch.no_grad():
        batch = torch.from_numpy(images_np)
        logits = trained_model(batch)
        pred_np = logits.argmax(1).cpu().numpy()

    fig = plot_sample_predictions(images_np, actual_np, pred_np)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    html = _build_html_report(overall_metrics, per_class, n_test, image_b64)
    return HTMLString(html)
