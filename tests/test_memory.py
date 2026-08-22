"""测试误报记忆：积累生效、防污染白名单。"""

from agents.config import Settings
from agents.llm import MockLLMClient
from agents.memory.embedding import HashingEmbedder
from agents.memory.false_positive import lookup_similar_false_positive, remember_false_positive, should_suppress
from agents.memory.vector_store import InMemoryVectorStore
from agents.models import Alarm


def make_alarm(severity: str = "medium") -> Alarm:
    return Alarm(id="a1", camera_id="cam_001", rule_id="rule_helmet", rule_name="未佩戴安全帽", severity=severity, confidence=0.9)


def _ctx():
    return Settings(), InMemoryVectorStore(), HashingEmbedder(dim=256), MockLLMClient()


def test_activation_requires_two_remembers():
    settings, store, emb, llm = _ctx()
    alarm = make_alarm()
    # 第一次记住：occurrence_count=1，不足以抑制
    remember_false_positive(alarm, "光照误判", store, emb, llm, settings)
    hits = lookup_similar_false_positive(alarm, store, emb, llm, settings)
    assert should_suppress(alarm, hits, settings)[0] is False
    # 第二次记住：occurrence_count=2，达到 activation_count，抑制生效
    remember_false_positive(alarm, "光照误判", store, emb, llm, settings)
    hits = lookup_similar_false_positive(alarm, store, emb, llm, settings)
    suppress, reason = should_suppress(alarm, hits, settings)
    assert suppress is True
    assert "count=2" in reason


def test_high_severity_never_suppressed():
    settings, store, emb, llm = _ctx()
    alarm = make_alarm(severity="high")
    for _ in range(3):
        remember_false_positive(alarm, "x", store, emb, llm, settings)
    hits = lookup_similar_false_positive(alarm, store, emb, llm, settings)
    suppress, reason = should_suppress(alarm, hits, settings)
    assert suppress is False
    assert "whitelist" in reason


def test_unrelated_alarm_not_suppressed():
    settings, store, emb, llm = _ctx()
    src = make_alarm()
    for _ in range(2):
        remember_false_positive(src, "x", store, emb, llm, settings)
    # 不同规则/摄像头的告警：检索键不同，不应被抑制
    other = Alarm(id="a2", camera_id="cam_999", rule_id="rule_fire", rule_name="明火", severity="medium", confidence=0.9)
    hits = lookup_similar_false_positive(other, store, emb, llm, settings)
    assert should_suppress(other, hits, settings)[0] is False
