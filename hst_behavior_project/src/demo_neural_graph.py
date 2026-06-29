"""
分层时空行为图神经模型原型

这个版本比 hst_action_graph_demo.py 更接近“真正模型”：
1. 用 NumPy 实现可训练参数：Embedding、Linear、Graph Attention、分类头
2. 支持动态多源传感器事件输入
3. 使用图注意力传播传感器节点信息
4. 三层输出：基础动作 -> 简单行为 -> 高级行为
5. 下一层显式复用上一层的概率分布

说明：
- 当前文件提供的是模型结构和前向推理，不包含真实数据训练好的权重。
- 因为没有标注数据，默认输出采用“神经网络 logits + 先验 logits”融合。
- 后续拿到数据后，可以用 PyTorch/JAX 重写训练，或在本结构上补反向传播。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import cos, pi, sin, sqrt
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


def normalize_score(x: float, scale: float) -> float:
    return max(0.0, min(1.0, x / scale))


@dataclass
class SensorEvent:
    start: datetime
    end: datetime
    source: str
    sensor_type: str
    values: Dict[str, Any]
    confidence: float = 1.0
    location: Optional[str] = None


@dataclass
class Prediction:
    label: str
    confidence: float
    evidence: List[str]
    distribution: Dict[str, float]


class Vocab:
    def __init__(self, items: List[str]) -> None:
        self.items = ["<unk>"] + items
        self.index = {item: i for i, item in enumerate(self.items)}

    def id(self, item: Optional[str]) -> int:
        if item is None:
            return 0
        return self.index.get(item, 0)

    def __len__(self) -> int:
        return len(self.items)


class Linear:
    def __init__(self, in_dim: int, out_dim: int, rng: np.random.Generator) -> None:
        self.weight = rng.normal(0, 1 / sqrt(in_dim), size=(in_dim, out_dim))
        self.bias = np.zeros(out_dim)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x @ self.weight + self.bias


class GraphAttentionLayer:
    """轻量图注意力层：根据节点表示和邻接偏置进行消息传递。"""

    def __init__(self, dim: int, rng: np.random.Generator) -> None:
        self.q = Linear(dim, dim, rng)
        self.k = Linear(dim, dim, rng)
        self.v = Linear(dim, dim, rng)
        self.o = Linear(dim, dim, rng)
        self.dim = dim

    def __call__(self, node_states: np.ndarray, adjacency_bias: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        q = self.q(node_states)
        k = self.k(node_states)
        v = self.v(node_states)
        logits = (q @ k.T) / sqrt(self.dim) + adjacency_bias
        attn = softmax(logits, axis=-1)
        out = relu(self.o(attn @ v))
        return out, attn


class SensorGraphEncoder:
    """把传感器事件编码成图节点，再通过图注意力得到全局表示。"""

    def __init__(self, hidden_dim: int = 48, seed: int = 7) -> None:
        self.rng = np.random.default_rng(seed)
        self.hidden_dim = hidden_dim

        self.source_vocab = Vocab(["phone", "smart_band", "smart_watch", "smart_home"])
        self.sensor_vocab = Vocab([
            "accelerometer",
            "gyroscope",
            "heart_rate",
            "gps",
            "pedometer",
            "wifi",
            "screen",
            "door",
            "pir",
            "smart_plug",
        ])
        self.location_vocab = Vocab([
            "home",
            "office",
            "restaurant",
            "badminton_court",
            "basketball_court",
            "gym",
            "subway",
        ])

        self.source_emb = self.rng.normal(0, 0.2, size=(len(self.source_vocab), hidden_dim))
        self.sensor_emb = self.rng.normal(0, 0.2, size=(len(self.sensor_vocab), hidden_dim))
        self.location_emb = self.rng.normal(0, 0.2, size=(len(self.location_vocab), hidden_dim))

        self.numeric_proj = Linear(10, hidden_dim, self.rng)
        self.gat1 = GraphAttentionLayer(hidden_dim, self.rng)
        self.gat2 = GraphAttentionLayer(hidden_dim, self.rng)
        self.pool_gate = Linear(hidden_dim, 1, self.rng)

    def encode(self, events: List[SensorEvent]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        node_features = []
        for event in events:
            source_id = self.source_vocab.id(event.source)
            sensor_id = self.sensor_vocab.id(event.sensor_type)
            location_id = self.location_vocab.id(self._normalize_location(event.location))
            numeric = self._numeric_features(event)

            node = (
                self.source_emb[source_id]
                + self.sensor_emb[sensor_id]
                + self.location_emb[location_id]
                + self.numeric_proj(numeric)
            )
            node_features.append(np.tanh(node))

        h = np.vstack(node_features)
        adjacency_bias = self._build_adjacency_bias(events)
        h1, attn1 = self.gat1(h, adjacency_bias)
        h2, attn2 = self.gat2(h1 + h, adjacency_bias)

        pooled = self._attention_pool(h2)
        summary = self._summarize(events, attn2)
        return pooled, h2, summary

    def _attention_pool(self, node_states: np.ndarray) -> np.ndarray:
        gates = self.pool_gate(node_states).reshape(-1)
        weights = softmax(gates)
        return weights @ node_states

    def _numeric_features(self, event: SensorEvent) -> np.ndarray:
        duration_min = (event.end - event.start).total_seconds() / 60.0
        hour = event.start.hour + event.start.minute / 60.0
        values = event.values

        return np.array(
            [
                normalize_score(duration_min, 120),
                event.confidence,
                normalize_score(float(values.get("intensity", 0)), 1),
                normalize_score(float(values.get("rotation", 0)), 1),
                normalize_score(float(values.get("bpm", 0)), 200),
                normalize_score(float(values.get("steps", 0)), 3000),
                normalize_score(float(values.get("speed", 0)), 30),
                1.0 if values.get("active", False) else 0.0,
                sin(2 * pi * hour / 24),
                cos(2 * pi * hour / 24),
            ],
            dtype=float,
        )

    def _build_adjacency_bias(self, events: List[SensorEvent]) -> np.ndarray:
        n = len(events)
        bias = np.zeros((n, n), dtype=float)

        for i, a in enumerate(events):
            for j, b in enumerate(events):
                if i == j:
                    bias[i, j] += 0.2
                if a.location and b.location and a.location == b.location:
                    bias[i, j] += 0.35
                if a.source == b.source:
                    bias[i, j] += 0.15

                gap = abs((a.start - b.start).total_seconds()) / 60.0
                bias[i, j] += max(0.0, 0.4 - gap / 60.0)

        return bias

    def _normalize_location(self, location: Optional[str]) -> Optional[str]:
        if location is None:
            return None
        if "羽毛球" in location:
            return "badminton_court"
        if "篮球" in location:
            return "basketball_court"
        if "餐厅" in location or "饭店" in location:
            return "restaurant"
        if "公司" in location or "办公室" in location:
            return "office"
        if "家" in location:
            return "home"
        if "健身" in location:
            return "gym"
        if "地铁" in location:
            return "subway"
        return None

    def _summarize(self, events: List[SensorEvent], attn: np.ndarray) -> Dict[str, Any]:
        start = min(e.start for e in events)
        end = max(e.end for e in events)
        locations = [e.location for e in events if e.location]
        values_by_type: Dict[str, List[Dict[str, Any]]] = {}
        for event in events:
            values_by_type.setdefault(event.sensor_type, []).append(event.values)

        return {
            "start": start,
            "end": end,
            "duration_min": (end - start).total_seconds() / 60,
            "locations": locations,
            "hour": start.hour + start.minute / 60,
            "values_by_type": values_by_type,
            "attention": attn,
        }


class HierarchicalActionModel:
    """三层分层模型：基础动作、简单行为、高级行为。"""

    basic_labels = ["静止", "步行/移动", "高频手腕挥动", "未知基础动作"]
    simple_labels = ["羽毛球", "篮球", "球类/挥拍运动", "吃饭", "办公", "运动", "未知简单行为"]
    high_labels = ["晚间羽毛球运动", "篮球运动", "午餐", "工作", "运动活动", "未知高级行为"]

    def __init__(self, hidden_dim: int = 48, seed: int = 7, neural_weight: float = 0.35) -> None:
        self.rng = np.random.default_rng(seed)
        self.encoder = SensorGraphEncoder(hidden_dim=hidden_dim, seed=seed)
        self.neural_weight = neural_weight

        self.basic_head = Linear(hidden_dim, len(self.basic_labels), self.rng)
        self.basic_to_simple = Linear(hidden_dim + len(self.basic_labels), len(self.simple_labels), self.rng)
        self.simple_to_high = Linear(
            hidden_dim + len(self.basic_labels) + len(self.simple_labels),
            len(self.high_labels),
            self.rng,
        )

    def predict(self, events: List[SensorEvent]) -> Dict[str, Prediction]:
        pooled, _, summary = self.encoder.encode(events)

        basic_neural = self.basic_head(pooled)
        basic_prior, basic_evidence = self._basic_prior(summary)
        basic_prob = self._fuse_logits(basic_neural, basic_prior)

        simple_input = np.concatenate([pooled, basic_prob])
        simple_neural = self.basic_to_simple(simple_input)
        simple_prior, simple_evidence = self._simple_prior(summary, basic_prob)
        simple_prob = self._fuse_logits(simple_neural, simple_prior)

        high_input = np.concatenate([pooled, basic_prob, simple_prob])
        high_neural = self.simple_to_high(high_input)
        high_prior, high_evidence = self._high_prior(summary, simple_prob)
        high_prob = self._fuse_logits(high_neural, high_prior)

        return {
            "基础动作": self._to_prediction(self.basic_labels, basic_prob, basic_evidence),
            "简单行为": self._to_prediction(self.simple_labels, simple_prob, simple_evidence),
            "高级行为": self._to_prediction(self.high_labels, high_prob, high_evidence),
        }

    def _fuse_logits(self, neural_logits: np.ndarray, prior_logits: np.ndarray) -> np.ndarray:
        neural_prob = softmax(neural_logits)
        prior_prob = softmax(prior_logits)
        prob = self.neural_weight * neural_prob + (1 - self.neural_weight) * prior_prob
        return prob / prob.sum()

    def _to_prediction(self, labels: List[str], prob: np.ndarray, evidence_map: Dict[str, List[str]]) -> Prediction:
        idx = int(np.argmax(prob))
        label = labels[idx]
        return Prediction(
            label=label,
            confidence=float(prob[idx]),
            evidence=evidence_map.get(label, ["神经图模型综合判断"]),
            distribution={labels[i]: float(prob[i]) for i in range(len(labels))},
        )

    def _basic_prior(self, summary: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, List[str]]]:
        logits = np.zeros(len(self.basic_labels))
        evidence: Dict[str, List[str]] = {}

        values = summary["values_by_type"]
        acc = values.get("accelerometer", [{}])
        gyro = values.get("gyroscope", [{}])
        pedo = values.get("pedometer", [{}])

        intensity = max(float(v.get("intensity", 0)) for v in acc)
        rotation = max(float(v.get("rotation", 0)) for v in gyro)
        steps = sum(int(v.get("steps", 0)) for v in pedo)

        if intensity < 0.2 and steps < 50:
            logits[self.basic_labels.index("静止")] += 2.2
            evidence["静止"] = [f"运动强度低={intensity:.2f}", f"步数少={steps}"]

        if steps > 300 or intensity > 0.45:
            logits[self.basic_labels.index("步行/移动")] += 1.8
            evidence["步行/移动"] = [f"步数={steps}", f"运动强度={intensity:.2f}"]

        if rotation > 0.65 and intensity > 0.55:
            logits[self.basic_labels.index("高频手腕挥动")] += 2.4
            evidence["高频手腕挥动"] = [f"旋转强度={rotation:.2f}", f"运动强度={intensity:.2f}"]

        logits[self.basic_labels.index("未知基础动作")] += 0.2
        return logits, evidence

    def _simple_prior(self, summary: Dict[str, Any], basic_prob: np.ndarray) -> Tuple[np.ndarray, Dict[str, List[str]]]:
        logits = np.zeros(len(self.simple_labels))
        evidence: Dict[str, List[str]] = {}

        locations = " ".join(summary["locations"])
        hour = summary["hour"]
        duration = summary["duration_min"]
        values = summary["values_by_type"]
        heart_rate = max(float(v.get("bpm", 0)) for v in values.get("heart_rate", [{"bpm": 0}]))

        p_wave = basic_prob[self.basic_labels.index("高频手腕挥动")]
        p_static = basic_prob[self.basic_labels.index("静止")]

        if p_wave > 0.25:
            logits[self.simple_labels.index("球类/挥拍运动")] += 1.4 * p_wave
            evidence["球类/挥拍运动"] = [f"基础动作支持：高频手腕挥动概率={p_wave:.2f}"]

        if "羽毛球" in locations:
            logits[self.simple_labels.index("羽毛球")] += 2.0
            evidence.setdefault("羽毛球", []).append("地点包含羽毛球场/羽毛球馆")
            if p_wave > 0.25:
                logits[self.simple_labels.index("羽毛球")] += 1.2 * p_wave
                evidence["羽毛球"].append("高频手腕挥动与羽毛球场景匹配")

        if "篮球" in locations:
            logits[self.simple_labels.index("篮球")] += 2.0
            evidence.setdefault("篮球", []).append("地点包含篮球场")

        if heart_rate >= 110:
            logits[self.simple_labels.index("运动")] += 1.1
            evidence["运动"] = [f"心率较高={heart_rate:.0f}"]
            if "羽毛球" in locations:
                logits[self.simple_labels.index("羽毛球")] += 0.7
                evidence.setdefault("羽毛球", []).append(f"运动心率支持={heart_rate:.0f}")

        if "餐厅" in locations and 11 <= hour <= 14 and p_static > 0.25:
            logits[self.simple_labels.index("吃饭")] += 2.3
            evidence["吃饭"] = ["餐厅地点", "午餐时段", f"静止概率={p_static:.2f}"]

        if ("公司" in locations or "办公室" in locations) and p_static > 0.25:
            logits[self.simple_labels.index("办公")] += 1.8
            evidence["办公"] = ["办公地点", f"静止概率={p_static:.2f}"]

        if duration >= 30:
            for label, ev in list(evidence.items()):
                logits[self.simple_labels.index(label)] += 0.25
                ev.append(f"持续时间={duration:.0f}分钟")

        logits[self.simple_labels.index("未知简单行为")] += 0.1
        return logits, evidence

    def _high_prior(self, summary: Dict[str, Any], simple_prob: np.ndarray) -> Tuple[np.ndarray, Dict[str, List[str]]]:
        logits = np.zeros(len(self.high_labels))
        evidence: Dict[str, List[str]] = {}

        hour = summary["hour"]
        p_badminton = simple_prob[self.simple_labels.index("羽毛球")]
        p_basketball = simple_prob[self.simple_labels.index("篮球")]
        p_eating = simple_prob[self.simple_labels.index("吃饭")]
        p_work = simple_prob[self.simple_labels.index("办公")]
        p_sport = simple_prob[self.simple_labels.index("运动")] + simple_prob[self.simple_labels.index("球类/挥拍运动")]

        if p_badminton > 0.2:
            label = "晚间羽毛球运动" if hour >= 17 else "运动活动"
            logits[self.high_labels.index(label)] += 2.2 * p_badminton
            evidence[label] = [f"简单行为支持：羽毛球概率={p_badminton:.2f}"]

        if p_basketball > 0.2:
            logits[self.high_labels.index("篮球运动")] += 2.0 * p_basketball
            evidence["篮球运动"] = [f"简单行为支持：篮球概率={p_basketball:.2f}"]

        if p_eating > 0.2 and 11 <= hour <= 14:
            logits[self.high_labels.index("午餐")] += 2.0 * p_eating
            evidence["午餐"] = [f"简单行为支持：吃饭概率={p_eating:.2f}", "时间处于午餐时段"]

        if p_work > 0.2:
            logits[self.high_labels.index("工作")] += 2.0 * p_work
            evidence["工作"] = [f"简单行为支持：办公概率={p_work:.2f}"]

        if p_sport > 0.25:
            logits[self.high_labels.index("运动活动")] += 1.3 * p_sport
            evidence["运动活动"] = [f"运动相关简单行为总概率={p_sport:.2f}"]

        logits[self.high_labels.index("未知高级行为")] += 0.1
        return logits, evidence


def build_demo_events() -> List[SensorEvent]:
    start = datetime(2026, 6, 26, 19, 0, 0)
    end = start + timedelta(minutes=70)

    return [
        SensorEvent(start, end, "smart_band", "accelerometer", {"intensity": 0.78}, location="星河羽毛球馆"),
        SensorEvent(start, end, "smart_band", "gyroscope", {"rotation": 0.88}, location="星河羽毛球馆"),
        SensorEvent(start, end, "smart_band", "heart_rate", {"bpm": 132}, location="星河羽毛球馆"),
        SensorEvent(start, end, "phone", "gps", {"speed": 0.2}, location="星河羽毛球馆"),
        SensorEvent(start, end, "smart_band", "pedometer", {"steps": 1800}, location="星河羽毛球馆"),
    ]


def print_prediction(name: str, prediction: Prediction) -> None:
    print(f"\n{name}")
    print(f"- 输出：{prediction.label}")
    print(f"- 置信度：{prediction.confidence:.2f}")
    print("- 依据：" + "；".join(prediction.evidence))
    print("- 概率分布：")
    for label, score in sorted(prediction.distribution.items(), key=lambda x: x[1], reverse=True):
        print(f"  {label}: {score:.2f}")


def main() -> None:
    model = HierarchicalActionModel(hidden_dim=48, neural_weight=0.35)
    predictions = model.predict(build_demo_events())

    print("分层时空行为图神经模型 Demo")
    print("=" * 36)
    print_prediction("第一层：基础动作", predictions["基础动作"])
    print_prediction("第二层：简单行为", predictions["简单行为"])
    print_prediction("第三层：高级行为", predictions["高级行为"])


if __name__ == "__main__":
    main()
