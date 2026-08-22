"""HITL 告警复核台（Streamlit，可选依赖）。

展示关键帧拼图 + 模型证据 + 置信度，人工选择确认/误报/改级。
诚实标注：这是交互式复核台的简化版，展示 HITL 交互形态；
关键帧拼图（Pillow 2×2）与真实告警流接入见 Phase 5 完整版。
运行：pip install streamlit && streamlit run frontend/streamlit_review.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from agents.models import Alarm, FrameEvidence

st.set_page_config(page_title="GuardEye 告警复核台", layout="wide")
st.title("GuardEye 告警复核台")


# mock 待复核数据（真实接入后从 pending_review 读取）
@st.cache_data
def mock_pending() -> list[Alarm]:
    return [
        Alarm(
            id="alarm_demo1",
            camera_id="cam_001",
            rule_id="rule_helmet",
            rule_name="未佩戴安全帽",
            severity="high",
            confidence=0.87,
            evidence=[
                FrameEvidence(frame_index=20, timestamp_seconds=2.0, description="人员未佩戴安全帽"),
                FrameEvidence(frame_index=30, timestamp_seconds=3.0, description="同上"),
            ],
        )
    ]


for alarm in mock_pending():
    with st.expander(f"[{alarm.severity}] {alarm.rule_name} @ {alarm.camera_id}（置信度 {alarm.confidence:.2f}）", expanded=True):
        st.write("**证据帧**")
        for e in alarm.evidence:
            st.write(f"- 帧 {e.frame_index}（t={e.timestamp_seconds:.1f}s）：{e.description}")

        col1, col2, col3 = st.columns(3)
        if col1.button("确认", key=f"confirm_{alarm.id}"):
            st.success(f"{alarm.id} 已确认，将进入事件记忆")
        if col2.button("误报", key=f"fp_{alarm.id}"):
            comment = st.text_input("误报原因（用于签名抽取）", key=f"comment_{alarm.id}")
            st.info(f"{alarm.id} 标记为误报，签名将写入误报记忆")
        if col3.button("修改级别", key=f"sev_{alarm.id}"):
            st.warning(f"{alarm.id} 级别待修改")
