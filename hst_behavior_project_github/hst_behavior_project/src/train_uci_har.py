"""
用真实 UCI HAR 数据集训练一个分层行为识别模型。

作用：
1. 自动下载 UCI HAR 公开数据集到临时目录
2. 读取真实的手机加速度计/陀螺仪特征
3. 用 NumPy 实现 MLP，并通过反向传播训练
4. 输出测试集准确率、混淆矩阵和分层行为结果

说明：
- UCI HAR 是真实公开数据集，不是模拟数据。
- 该数据集标签只有 6 类日常动作，不包含羽毛球/篮球等复杂运动。
- 这里主要验证“真实传感器数据 -> 神经网络训练 -> 分层行为输出”这条链路。
"""

from __future__ import annotations

import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


DATA_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
PROJECT_DIR = Path(__file__).resolve().parents[1]
WORK_DIR = PROJECT_DIR / "data" / "uci_har_realdata"
ZIP_PATH = WORK_DIR / "uci_har.zip"
INNER_ZIP_PATH = WORK_DIR / "UCI HAR Dataset.zip"
DATA_DIR = WORK_DIR / "UCI HAR Dataset"
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_PATH = OUTPUT_DIR / "hst_uci_har_mlp_model.npz"
RESULT_PATH = OUTPUT_DIR / "hst_uci_har_realdata_result.json"


ACTIVITY_NAMES = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}


CN_ACTIVITY_NAMES = {
    "WALKING": "步行",
    "WALKING_UPSTAIRS": "上楼",
    "WALKING_DOWNSTAIRS": "下楼",
    "SITTING": "坐着",
    "STANDING": "站立",
    "LAYING": "躺卧",
}


@dataclass
class TrainResult:
    train_accuracy: float
    test_accuracy: float
    confusion_matrix: List[List[int]]
    sample_outputs: List[Dict[str, object]]


def download_dataset() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_DIR.exists():
        return

    if not ZIP_PATH.exists():
        print("正在下载 UCI HAR 真实数据集，大约 58MB...")
        urllib.request.urlretrieve(DATA_URL, ZIP_PATH)

    print("正在解压数据集...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(WORK_DIR)

    # UCI 现在的下载包外层还包了一层 “UCI HAR Dataset.zip”。
    if INNER_ZIP_PATH.exists() and not DATA_DIR.exists():
        with zipfile.ZipFile(INNER_ZIP_PATH, "r") as zf:
            zf.extractall(WORK_DIR)


def load_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    download_dataset()

    x_train = np.loadtxt(DATA_DIR / "train" / "X_train.txt")
    y_train = np.loadtxt(DATA_DIR / "train" / "y_train.txt", dtype=int) - 1
    x_test = np.loadtxt(DATA_DIR / "test" / "X_test.txt")
    y_test = np.loadtxt(DATA_DIR / "test" / "y_test.txt", dtype=int) - 1

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    return x_train, y_train, x_test, y_test


def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((len(y), num_classes), dtype=float)
    out[np.arange(len(y)), y] = 1.0
    return out


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


class MLP:
    """一个最小但完整的反向传播模型。"""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, np.sqrt(2 / in_dim), size=(in_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.w2 = rng.normal(0, np.sqrt(2 / hidden_dim), size=(hidden_dim, out_dim))
        self.b2 = np.zeros(out_dim)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        z1 = x @ self.w1 + self.b1
        h1 = np.maximum(z1, 0)
        logits = h1 @ self.w2 + self.b2
        prob = softmax(logits)
        return z1, h1, prob

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)[2]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)

    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 40,
        batch_size: int = 128,
        lr: float = 0.015,
        weight_decay: float = 1e-4,
    ) -> None:
        rng = np.random.default_rng(123)
        y_onehot = one_hot(y, 6)

        for epoch in range(1, epochs + 1):
            order = rng.permutation(len(x))
            x_shuffled = x[order]
            y_shuffled = y_onehot[order]

            for start in range(0, len(x), batch_size):
                end = start + batch_size
                xb = x_shuffled[start:end]
                yb = y_shuffled[start:end]

                z1, h1, prob = self.forward(xb)
                batch_n = len(xb)

                dlogits = (prob - yb) / batch_n
                dw2 = h1.T @ dlogits + weight_decay * self.w2
                db2 = dlogits.sum(axis=0)

                dh1 = dlogits @ self.w2.T
                dz1 = dh1 * (z1 > 0)
                dw1 = xb.T @ dz1 + weight_decay * self.w1
                db1 = dz1.sum(axis=0)

                self.w2 -= lr * dw2
                self.b2 -= lr * db2
                self.w1 -= lr * dw1
                self.b1 -= lr * db1

            if epoch % 10 == 0 or epoch == 1:
                pred = self.predict(x)
                acc = float(np.mean(pred == y))
                print(f"epoch={epoch:02d} train_acc={acc:.4f}")

    def save(self, path: Path) -> None:
        np.savez(path, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 6) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for real, pred in zip(y_true, y_pred):
        matrix[int(real), int(pred)] += 1
    return matrix


def to_hierarchy(activity: str, confidence: float) -> Dict[str, object]:
    cn = CN_ACTIVITY_NAMES[activity]

    if activity in {"WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"}:
        basic = "周期性步态运动"
        simple = cn
        high = "移动类日常活动"
    elif activity in {"SITTING", "STANDING"}:
        basic = "低强度静态姿态"
        simple = cn
        high = "静态停留活动"
    else:
        basic = "水平静态姿态"
        simple = cn
        high = "休息/睡眠相关活动"

    return {
        "基础动作": {"label": basic, "confidence": confidence},
        "简单行为": {"label": simple, "confidence": confidence},
        "高级行为": {"label": high, "confidence": confidence},
    }


def build_sample_outputs(model: MLP, x_test: np.ndarray, y_test: np.ndarray, count: int = 8) -> List[Dict[str, object]]:
    prob = model.predict_proba(x_test[:count])
    pred = np.argmax(prob, axis=1)
    outputs = []

    for i in range(count):
        pred_activity = ACTIVITY_NAMES[int(pred[i]) + 1]
        real_activity = ACTIVITY_NAMES[int(y_test[i]) + 1]
        conf = float(prob[i, pred[i]])
        outputs.append(
            {
                "sample_id": i,
                "real_label": CN_ACTIVITY_NAMES[real_activity],
                "pred_label": CN_ACTIVITY_NAMES[pred_activity],
                "confidence": round(conf, 4),
                "hierarchy": to_hierarchy(pred_activity, conf),
            }
        )

    return outputs


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    x_train, y_train, x_test, y_test = load_dataset()
    print(f"训练集: {x_train.shape}, 测试集: {x_test.shape}")

    model = MLP(in_dim=x_train.shape[1], hidden_dim=96, out_dim=6)
    model.train(x_train, y_train)

    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)
    train_acc = accuracy(y_train, train_pred)
    test_acc = accuracy(y_test, test_pred)
    cm = confusion_matrix(y_test, test_pred)
    samples = build_sample_outputs(model, x_test, y_test)

    result = TrainResult(
        train_accuracy=train_acc,
        test_accuracy=test_acc,
        confusion_matrix=cm.tolist(),
        sample_outputs=samples,
    )

    model.save(MODEL_PATH)
    RESULT_PATH.write_text(
        json.dumps(result.__dict__, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n真实数据训练完成")
    print(f"训练准确率: {train_acc:.4f}")
    print(f"测试准确率: {test_acc:.4f}")
    print("测试集混淆矩阵:")
    print(cm)
    print(f"模型已保存: {MODEL_PATH}")
    print(f"结果已保存: {RESULT_PATH}")
    print("\n前几个样本的分层输出:")
    for item in samples[:5]:
        print(
            f"样本{item['sample_id']} | 真实={item['real_label']} | "
            f"预测={item['pred_label']} | 高级行为={item['hierarchy']['高级行为']['label']} | "
            f"置信度={item['confidence']}"
        )


if __name__ == "__main__":
    main()
