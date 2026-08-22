"""测试 reducer：merge_by_id 的去重合并语义。"""

from dataclasses import dataclass

from agents.state import merge_by_id


@dataclass
class Item:
    id: str
    value: str


def test_merge_by_id_dedups_and_overwrites():
    a = Item("1", "a")
    b = Item("2", "b")
    c = Item("1", "new")  # 同 id，新值应覆盖旧值
    merged = merge_by_id([a, b], [c])
    ids = [i.id for i in merged]
    assert ids == ["1", "2"]  # 顺序保持首次出现
    assert next(i for i in merged if i.id == "1").value == "new"


def test_merge_by_id_empty():
    assert merge_by_id([], []) == []
    a = Item("1", "a")
    assert merge_by_id([], [a]) == [a]
