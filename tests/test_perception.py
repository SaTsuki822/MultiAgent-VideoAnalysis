"""测试感知层指标：Recall / Precision / F1 / 误报率。"""

from eval.dataset.annotations import Annotation, Prediction
from eval.metrics.perception import compute_perception_metrics


def test_perfect_prediction():
    anns = [Annotation(clip_id="c", rule_id="r", start_seconds=1.0, end_seconds=4.0, severity="high")]
    preds = [Prediction(clip_id="c", rule_id="r", timestamp_seconds=2.0)]
    m = compute_perception_metrics(anns, preds)
    assert m["tp"] == 1
    assert m["recall"] == 1.0
    assert m["precision"] == 1.0
    assert m["f1"] == 1.0


def test_miss_and_false_positive():
    anns = [Annotation(clip_id="c", rule_id="r", start_seconds=1.0, end_seconds=4.0, severity="high")]
    preds = [Prediction(clip_id="c", rule_id="r", timestamp_seconds=10.0)]  # 不在区间
    m = compute_perception_metrics(anns, preds)
    assert m["fp"] == 1
    assert m["fn"] == 1
    assert m["recall"] == 0.0
    assert m["false_positive_rate"] == 1.0


def test_empty_predictions():
    anns = [Annotation(clip_id="c", rule_id="r", start_seconds=1.0, end_seconds=4.0, severity="high")]
    m = compute_perception_metrics(anns, [])
    assert m["recall"] == 0.0
    assert m["false_positive_rate"] == 0.0
