"""
分层时空行为图模型最小原型

目标：
1. 接收动态多源传感器事件
2. 构建“传感器-动作-行为-高级行为”的图结构
3. 先识别基础动作，再组合成简单行为，再推理高级行为
4. 输出行为、置信度和依据

这个文件不是最终机器学习模型，而是一个可运行的建模骨架。
后续可以把 BasicActionInferencer / SimpleBehaviorInferencer /
HighLevelBehaviorInferencer 替换成真正的深度模型或 GNN。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from math import exp
from typing import Any, Dict, Iterable, List, Optional, Tuple


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + exp(-x))


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass
class SensorEvent:
    """统一后的传感器事件。"""

    start: datetime
    end: datetime
    source: str
    sensor_type: str
    values: Dict[str, Any]
    confidence: float = 1.0
    location: Optional[str] = None


@dataclass
class GraphNode:
    """图节点：可以是传感器、基础动作、简单行为、高级行为或上下文。"""

    node_id: str
    node_type: str
    label: str
    weight: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    """图边：表示两个节点之间的证据关系。"""

    src: str
    dst: str
    relation: str
    weight: float
    reason: str


@dataclass
class InferenceResult:
    """每一层推理的输出。"""

    label: str
    layer: str
    start: datetime
    end: datetime
    confidence: float
    evidence: List[str]
    scores: Dict[str, float] = field(default_factory=dict)


class BehaviorGraph:
    """保存一次推理过程中的节点和边。"""

    def __init__(self) -> None:
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []

    def add_node(self, node: GraphNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, edge: GraphEdge) -> None:
        self.edges.append(edge)

    def add_sensor_event(self, event: SensorEvent, index: int) -> str:
        node_id = f"sensor:{index}:{event.source}:{event.sensor_type}"
        label = f"{event.source}.{event.sensor_type}"
        self.add_node(
            GraphNode(
                node_id=node_id,
                node_type="sensor",
                label=label,
                weight=event.confidence,
                meta={
                    "start": event.start,
                    "end": event.end,
                    "values": event.values,
                    "location": event.location,
                },
            )
        )
        return node_id


class FeatureExtractor:
    """把原始传感器事件转成更容易推理的上下文特征。"""

    def extract(self, events: Iterable[SensorEvent]) -> Dict[str, Any]:
        features: Dict[str, Any] = {
            "locations": [],
            "time_hour": None,
            "heart_rate_avg": None,
            "wrist_motion_intensity": None,
            "wrist_rotation_intensity": None,
            "steps": 0,
            "screen_active": None,
            "wifi_place": None,
            "duration_min": 0.0,
        }

        events = list(events)
        if not events:
            return features

        start = min(e.start for e in events)
        end = max(e.end for e in events)
        features["duration_min"] = (end - start).total_seconds() / 60.0
        features["time_hour"] = start.hour + start.minute / 60.0

        heart_rates: List[float] = []
        wrist_motion_values: List[float] = []
        wrist_rotation_values: List[float] = []

        for event in events:
            if event.location:
                features["locations"].append(event.location)

            if event.sensor_type == "heart_rate":
                heart_rates.append(float(event.values.get("bpm", 0)))

            if event.sensor_type == "accelerometer":
                wrist_motion_values.append(float(event.values.get("intensity", 0)))

            if event.sensor_type == "gyroscope":
                wrist_rotation_values.append(float(event.values.get("rotation", 0)))

            if event.sensor_type == "pedometer":
                features["steps"] += int(event.values.get("steps", 0))

            if event.sensor_type == "screen":
                features["screen_active"] = bool(event.values.get("active", False))

            if event.sensor_type == "wifi":
                features["wifi_place"] = event.values.get("place")

        if heart_rates:
            features["heart_rate_avg"] = sum(heart_rates) / len(heart_rates)

        if wrist_motion_values:
            features["wrist_motion_intensity"] = sum(wrist_motion_values) / len(wrist_motion_values)

        if wrist_rotation_values:
            features["wrist_rotation_intensity"] = sum(wrist_rotation_values) / len(wrist_rotation_values)

        return features


class BasicActionInferencer:
    """第一层：基础动作识别。"""

    def infer(self, features: Dict[str, Any], start: datetime, end: datetime) -> List[InferenceResult]:
        motion = features.get("wrist_motion_intensity") or 0.0
        rotation = features.get("wrist_rotation_intensity") or 0.0
        steps = features.get("steps") or 0

        candidates: List[InferenceResult] = []

        if steps > 300 and motion > 0.45:
            confidence = clamp(0.45 + steps / 2000 + motion * 0.3)
            candidates.append(
                InferenceResult(
                    label="步行/移动",
                    layer="基础动作",
                    start=start,
                    end=end,
                    confidence=confidence,
                    evidence=[f"步数={steps}", f"手腕运动强度={motion:.2f}"],
                )
            )

        if motion < 0.2 and steps < 50:
            confidence = clamp(0.8 - motion)
            candidates.append(
                InferenceResult(
                    label="静止",
                    layer="基础动作",
                    start=start,
                    end=end,
                    confidence=confidence,
                    evidence=[f"步数较少={steps}", f"手腕运动强度低={motion:.2f}"],
                )
            )

        if rotation > 0.65 and motion > 0.55:
            confidence = clamp(0.35 + rotation * 0.45 + motion * 0.25)
            candidates.append(
                InferenceResult(
                    label="高频手腕挥动",
                    layer="基础动作",
                    start=start,
                    end=end,
                    confidence=confidence,
                    evidence=[f"手腕旋转强度={rotation:.2f}", f"手腕运动强度={motion:.2f}"],
                )
            )

        if not candidates:
            candidates.append(
                InferenceResult(
                    label="未知基础动作",
                    layer="基础动作",
                    start=start,
                    end=end,
                    confidence=0.35,
                    evidence=["基础传感器证据不足"],
                )
            )

        return sorted(candidates, key=lambda x: x.confidence, reverse=True)


class SimpleBehaviorInferencer:
    """第二层：由基础动作 + 时空上下文推理简单行为。"""

    def infer(
        self,
        features: Dict[str, Any],
        basic_actions: List[InferenceResult],
        start: datetime,
        end: datetime,
    ) -> List[InferenceResult]:
        locations = features.get("locations") or []
        location_text = " ".join(locations)
        hour = features.get("time_hour") or 0.0
        heart_rate = features.get("heart_rate_avg") or 0.0
        duration_min = features.get("duration_min") or 0.0

        action_map = {a.label: a for a in basic_actions}
        scores: Dict[str, Tuple[float, List[str]]] = {}

        def add_score(label: str, score: float, reason: str) -> None:
            old_score, old_reasons = scores.get(label, (0.0, []))
            scores[label] = (old_score + score, old_reasons + [reason])

        if "高频手腕挥动" in action_map:
            add_score("球类/挥拍运动", action_map["高频手腕挥动"].confidence * 0.45, "基础动作显示高频手腕挥动")
            if "羽毛球" in location_text:
                add_score("羽毛球", 0.28, "高频挥拍与羽毛球场景同时出现")
            if "篮球" in location_text:
                add_score("篮球", 0.18, "高频手腕动作与篮球场景同时出现")

        if heart_rate >= 110:
            add_score("球类/挥拍运动", 0.2, f"心率较高={heart_rate:.0f}")
            add_score("运动", 0.25, f"心率较高={heart_rate:.0f}")
            if "羽毛球" in location_text:
                add_score("羽毛球", 0.12, f"羽毛球场景下心率较高={heart_rate:.0f}")
            if "篮球" in location_text:
                add_score("篮球", 0.12, f"篮球场景下心率较高={heart_rate:.0f}")

        if "羽毛球" in location_text:
            add_score("羽毛球", 0.5, "地点包含羽毛球场/羽毛球馆")
            add_score("球类/挥拍运动", 0.25, "地点支持球类运动判断")

        if "篮球" in location_text:
            add_score("篮球", 0.5, "地点包含篮球场")
            add_score("球类/挥拍运动", 0.2, "地点支持球类运动判断")

        if "餐厅" in location_text and 11 <= hour <= 14 and "静止" in action_map:
            add_score("吃饭", 0.65, "午餐时段在餐厅且运动较少")

        if "公司" in location_text or features.get("wifi_place") == "office":
            if "静止" in action_map and features.get("screen_active"):
                add_score("办公", 0.55, "公司/办公网络环境且屏幕活跃")

        if duration_min >= 30:
            for label in list(scores.keys()):
                add_score(label, 0.08, f"持续时间较长={duration_min:.0f}分钟")

        results: List[InferenceResult] = []
        for label, (score, evidence) in scores.items():
            confidence = clamp(sigmoid(score * 2.4 - 0.8))
            results.append(
                InferenceResult(
                    label=label,
                    layer="简单行为",
                    start=start,
                    end=end,
                    confidence=confidence,
                    evidence=evidence,
                    scores={label: score},
                )
            )

        if not results:
            results.append(
                InferenceResult(
                    label="未知简单行为",
                    layer="简单行为",
                    start=start,
                    end=end,
                    confidence=0.30,
                    evidence=["基础动作和上下文不足以推断简单行为"],
                )
            )

        return sorted(results, key=lambda x: x.confidence, reverse=True)


class HighLevelBehaviorInferencer:
    """第三层：由简单行为 + 时间上下文推理高级行为。"""

    def infer(
        self,
        features: Dict[str, Any],
        simple_behaviors: List[InferenceResult],
        start: datetime,
        end: datetime,
    ) -> List[InferenceResult]:
        hour = features.get("time_hour") or 0.0
        results: List[InferenceResult] = []

        top = simple_behaviors[0]

        if top.label in {"羽毛球", "篮球", "球类/挥拍运动", "运动"}:
            confidence = clamp(top.confidence * 0.9 + 0.08)
            if top.label in {"羽毛球", "篮球"}:
                label = f"晚间{top.label}运动" if hour >= 17 else f"{top.label}运动"
            else:
                label = "晚间运动" if hour >= 17 else "运动活动"
            results.append(
                InferenceResult(
                    label=label,
                    layer="高级行为",
                    start=start,
                    end=end,
                    confidence=confidence,
                    evidence=[f"简单行为={top.label}", *top.evidence],
                )
            )

        if top.label == "吃饭":
            results.append(
                InferenceResult(
                    label="午餐" if 11 <= hour <= 14 else "用餐",
                    layer="高级行为",
                    start=start,
                    end=end,
                    confidence=top.confidence,
                    evidence=[f"简单行为={top.label}", *top.evidence],
                )
            )

        if top.label == "办公":
            results.append(
                InferenceResult(
                    label="工作",
                    layer="高级行为",
                    start=start,
                    end=end,
                    confidence=top.confidence,
                    evidence=[f"简单行为={top.label}", *top.evidence],
                )
            )

        if not results:
            results.append(
                InferenceResult(
                    label="未知高级行为",
                    layer="高级行为",
                    start=start,
                    end=end,
                    confidence=0.28,
                    evidence=["简单行为置信度不足或缺少高层语义规则"],
                )
            )

        return sorted(results, key=lambda x: x.confidence, reverse=True)


class HierarchicalSpatioTemporalActionGraph:
    """分层时空行为图推理主类。"""

    def __init__(self) -> None:
        self.feature_extractor = FeatureExtractor()
        self.basic_inferencer = BasicActionInferencer()
        self.simple_inferencer = SimpleBehaviorInferencer()
        self.high_level_inferencer = HighLevelBehaviorInferencer()

    def infer(self, events: List[SensorEvent]) -> Dict[str, Any]:
        if not events:
            raise ValueError("events 不能为空")

        start = min(e.start for e in events)
        end = max(e.end for e in events)

        graph = BehaviorGraph()
        sensor_node_ids = [graph.add_sensor_event(event, i) for i, event in enumerate(events)]

        features = self.feature_extractor.extract(events)

        basic_actions = self.basic_inferencer.infer(features, start, end)
        self._add_results_to_graph(graph, "basic", basic_actions, sensor_node_ids)

        simple_behaviors = self.simple_inferencer.infer(features, basic_actions, start, end)
        self._add_results_to_graph(graph, "simple", simple_behaviors, [f"basic:{r.label}" for r in basic_actions])

        high_level_behaviors = self.high_level_inferencer.infer(features, simple_behaviors, start, end)
        self._add_results_to_graph(
            graph,
            "high",
            high_level_behaviors,
            [f"simple:{r.label}" for r in simple_behaviors],
        )

        return {
            "features": features,
            "basic_actions": basic_actions,
            "simple_behaviors": simple_behaviors,
            "high_level_behaviors": high_level_behaviors,
            "graph": graph,
        }

    def _add_results_to_graph(
        self,
        graph: BehaviorGraph,
        prefix: str,
        results: List[InferenceResult],
        source_nodes: List[str],
    ) -> None:
        for result in results:
            node_id = f"{prefix}:{result.label}"
            graph.add_node(
                GraphNode(
                    node_id=node_id,
                    node_type=result.layer,
                    label=result.label,
                    weight=result.confidence,
                    meta={"evidence": result.evidence},
                )
            )
            for source_node in source_nodes:
                if source_node in graph.nodes:
                    graph.add_edge(
                        GraphEdge(
                            src=source_node,
                            dst=node_id,
                            relation="supports",
                            weight=result.confidence,
                            reason="上层推理复用下层输出",
                        )
                    )


def print_result(title: str, results: List[InferenceResult]) -> None:
    print(f"\n{title}")
    for item in results:
        print(f"- {item.label} | 置信度={item.confidence:.2f}")
        for ev in item.evidence[:4]:
            print(f"  依据：{ev}")


def build_demo_events() -> List[SensorEvent]:
    """构造一段“晚上在羽毛球馆运动”的示例传感器数据。"""

    start = datetime(2026, 6, 26, 19, 0, 0)
    end = start + timedelta(minutes=70)

    return [
        SensorEvent(
            start=start,
            end=end,
            source="smart_band",
            sensor_type="accelerometer",
            values={"intensity": 0.78},
            location="星河羽毛球馆",
        ),
        SensorEvent(
            start=start,
            end=end,
            source="smart_band",
            sensor_type="gyroscope",
            values={"rotation": 0.88},
            location="星河羽毛球馆",
        ),
        SensorEvent(
            start=start,
            end=end,
            source="smart_band",
            sensor_type="heart_rate",
            values={"bpm": 132},
            location="星河羽毛球馆",
        ),
        SensorEvent(
            start=start,
            end=end,
            source="phone",
            sensor_type="gps",
            values={"lat": 31.2304, "lon": 121.4737},
            location="星河羽毛球馆",
        ),
        SensorEvent(
            start=start,
            end=end,
            source="smart_band",
            sensor_type="pedometer",
            values={"steps": 1800},
            location="星河羽毛球馆",
        ),
    ]


def main() -> None:
    model = HierarchicalSpatioTemporalActionGraph()
    output = model.infer(build_demo_events())

    print("分层时空行为图模型 Demo")
    print("=" * 32)
    print("抽取特征：")
    for key, value in output["features"].items():
        print(f"- {key}: {value}")

    print_result("第一层：基础动作", output["basic_actions"])
    print_result("第二层：简单行为", output["simple_behaviors"])
    print_result("第三层：高级行为", output["high_level_behaviors"])

    graph: BehaviorGraph = output["graph"]
    print("\n图结构摘要")
    print(f"- 节点数：{len(graph.nodes)}")
    print(f"- 边数：{len(graph.edges)}")

    final = output["high_level_behaviors"][0]
    print("\n最终输出")
    print(f"{final.start.strftime('%H:%M')}-{final.end.strftime('%H:%M')} | {final.label} | 置信度={final.confidence:.2f}")
    print("依据：" + "；".join(final.evidence[:4]))


if __name__ == "__main__":
    main()
