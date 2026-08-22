"""评测回归入口：加载标注 → 跑系统 → 算指标 → 输出对比。

CI 集成点：改 prompt / 图结构后，运行本脚本跑评测集，指标回归（见 docs / GitHub Actions）。
当前无真实数据集时，用 MOCK_ANNOTATIONS 跑通流程；真实数据接入后替换 dataset 加载器即可。
"""

from __future__ import annotations

import json

from eval.dataset.annotations import MOCK_ANNOTATIONS, Annotation, Prediction
from eval.metrics.perception import compute_perception_metrics


def load_annotations(path: str | None = None) -> list[Annotation]:
    """从 JSON 加载标注；未提供时用样例。"""
    if path is None:
        return MOCK_ANNOTATIONS
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [Annotation(**item) for item in data]


def run_eval(predictions: list[Prediction], annotations: list[Annotation] | None = None) -> dict:
    annotations = annotations or load_annotations()
    metrics = compute_perception_metrics(annotations, predictions)
    return {"metrics": metrics, "annotations": len(annotations), "predictions": len(predictions)}


if __name__ == "__main__":
    # 无真实预测时，跑通样例指标（全 FP），验证链路
    sample_preds = [
        Prediction(clip_id="cam_001.mp4", rule_id="rule_helmet", timestamp_seconds=3.0),
    ]
    result = run_eval(sample_preds)
    print(json.dumps(result, ensure_ascii=False, indent=2))
