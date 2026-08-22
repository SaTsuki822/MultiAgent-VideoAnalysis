"""感知层指标：Recall / Precision / F1 / 误报率。

匹配规则：系统告警的时间戳落在标注区间内（±容差）视为命中该标注（TP）。
纯 Python 实现，无 sklearn 依赖，便于理解与面试讲解。
"""

from __future__ import annotations

from eval.dataset.annotations import Annotation, Prediction


def compute_perception_metrics(
    annotations: list[Annotation],
    predictions: list[Prediction],
    tolerance_seconds: float = 1.0,
) -> dict:
    """按 (clip_id, rule_id) 分组匹配，返回 TP/FP/FN 及衍生指标。"""
    # 建立 (clip_id, rule_id) -> [annotations] 索引
    ann_by_key: dict[tuple[str, str], list[Annotation]] = {}
    for a in annotations:
        ann_by_key.setdefault((a.clip_id, a.rule_id), []).append(a)

    matched_ann: set[int] = set()
    tp = fp = 0
    for i, p in enumerate(predictions):
        anns = ann_by_key.get((p.clip_id, p.rule_id), [])
        hit = any(
            a.start_seconds - tolerance_seconds <= p.timestamp_seconds <= a.end_seconds + tolerance_seconds
            and id(a) not in matched_ann
            for a in anns
        )
        if hit:
            tp += 1
            # 标记首个匹配到的标注为已命中（简化：一个预测命中一个标注）
            for a in anns:
                if a.start_seconds - tolerance_seconds <= p.timestamp_seconds <= a.end_seconds + tolerance_seconds and id(a) not in matched_ann:
                    matched_ann.add(id(a))
                    break
        else:
            fp += 1

    fn = len(annotations) - len(matched_ann)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    false_positive_rate = fp / len(predictions) if predictions else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
    }
