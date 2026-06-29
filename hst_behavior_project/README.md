# 分层时空行为图用户行为识别

Hierarchical Spatio-Temporal Behavior Graph (HST) - 基于多源传感器数据的分层用户行为识别系统。

## 项目简介

本项目实现了一个分层时空行为图模型，用于从动态多源传感器事件中识别用户行为。模型采用三层结构：**基础动作 → 简单行为 → 高级行为**，通过图结构建模传感器、动作和行为之间的证据关系。

项目包含三个可运行Demo：
- **规则版** (`demo_rule_graph.py`)：基于规则的分层推理骨架，可快速验证建模思路
- **神经图版** (`demo_neural_graph.py`)：NumPy 实现的轻量图注意力模型原型
- **真实数据训练** (`train_uci_har.py`)：使用 UCI HAR 公开数据集训练 MLP 模型

## 特性

-  🏗️ **三层分层结构**：基础动作 → 简单行为 → 高级行为，符合人类行为认知
-  📊 **图结构建模**：用节点和边表示传感器、动作、行为及其证据关系
-  🧠 **两种实现**：规则推理 + 神经网络（GAT图注意力）
-  📱 **多源传感器支持**：加速度计、陀螺仪、心率、GPS、计步器、WiFi、屏幕等
-  📈 **真实数据验证**：支持 UCI HAR 数据集训练与测试
-  🔧 **纯 NumPy 实现**：无需复杂依赖，易于理解和扩展

## 环境要求

- Python 3.9+
- NumPy >= 1.21

## 安装

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 规则版分层行为图 Demo

无需数据，直接运行：

```bash
python src/demo_rule_graph.py
```

输出示例：
```text
最终输出
19:00-20:10 | 晚间羽毛球运动 | 置信度=0.82
依据：简单行为=羽毛球；地点包含羽毛球场/羽毛球馆；...
```

### 2. 神经图模型原型 Demo

基于 NumPy 实现的图注意力模型：

```bash
python src/demo_neural_graph.py
```

### 3. UCI HAR 真实数据训练

首次运行会自动下载约 58MB 的 UCI HAR 数据集：

```bash
python src/train_uci_har.py
```

训练完成后会在 `outputs/` 目录生成：

| 文件 | 内容 |
|---|---|
| `hst_uci_har_mlp_model.npz` | MLP 模型权重 |
| `hst_uci_har_realdata_result.json` | 训练准确率、测试准确率、混淆矩阵、样本分层输出 |

## 模型架构

### 分层结构

```text
动态多源传感器输入
    ↓
基础动作（走、跑、坐、站、躺、挥手、静止、上下楼...）
    ↓
简单行为（通勤、运动、吃饭、睡觉、办公、做饭、羽毛球...）
    ↓
高级行为（工作、休息、锻炼、外出、社交、生活规律...）
    ↓
输出：行为、时间、置信度、依据
```

### 图结构

- **节点**：传感器、基础动作、简单行为、高级行为、上下文
- **边**：证据支撑关系（supports）
- **传播**：自底向上的证据传递，上层推理复用下层输出

## 项目结构

```
hst_behavior_project/
├── README.md              # 项目说明文档
├── requirements.txt       # Python 依赖
├── .gitignore             # Git 忽略规则
├── src/                   # 源代码目录
│   ├── demo_rule_graph.py    # 规则版分层行为图 Demo
│   ├── demo_neural_graph.py  # 神经图注意力模型原型
│   └── train_uci_har.py      # UCI HAR 数据集训练脚本
├── data/                  # 数据目录（运行时自动生成）
└── outputs/               # 输出目录（模型、结果等）
```

## 文件说明

| 文件 | 作用 | 是否需要下载数据 |
|---|---|---|
| `src/demo_rule_graph.py` | 规则版分层时空行为图 Demo，输入一段"羽毛球馆运动"的多源传感器事件，输出基础动作、简单行为、高级行为、置信度和依据 | 否 |
| `src/demo_neural_graph.py` | NumPy 实现的轻量图注意力模型原型，展示"传感器图编码 + 三层行为输出" | 否 |
| `src/train_uci_har.py` | 使用 UCI HAR 真实公开数据集训练 MLP，输出准确率、混淆矩阵和分层样本结果 | 是，首次运行会自动下载约 58MB 数据 |

## 扩展方向

- 用 PyTorch / JAX 重写神经网络部分，支持真正的端到端训练
- 接入更多传感器类型（智能家居、可穿戴设备等）
- 增加时间序列建模（LSTM、Transformer 等）
- 实现实时在线推理
- 添加可视化界面，展示图结构和推理过程

## 许可证

MIT License
