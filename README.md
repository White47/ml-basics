## 1.项目介绍

------

```markdown
# 🚢 泰坦尼克号生存预测（Titanic Survival Prediction）

> 用机器学习预测乘客是否在泰坦尼克号沉没中生还  
> **Kaggle 公开分数：0.7703** ✅

## 🎯 项目目标
- 从零实现逻辑回归（手写梯度下降）
- 使用 Scikit-learn 构建完整机器学习 Pipeline
- 完成 EDA、特征工程、交叉验证
- 提交结果至 [Kaggle Titanic](https://www.kaggle.com/c/titanic) 并获得 ≥ 0.75 的分数

## 🛠️ 技术栈
- **语言**: Python 3.10+
- **核心库**: 
  - `pandas`, `numpy`（数据处理）
  - `scikit-learn`（模型训练、Pipeline、交叉验证）
  - `matplotlib`, `seaborn`（可视化）
- **自定义模块**:
  - `src/logistic_regression_scratch.py`：手写逻辑回归
  - `src/utils.py`：数据加载与预处理

## 🔍 关键发现（EDA 结论）
- 👩 **女性生存率（74%）远高于男性（19%）**
- 🥇 **头等舱（Pclass=1）乘客生存率最高（63%）**
- 👨‍👩‍👧‍👦 **家庭规模适中（2-4人）的乘客更可能生还**
- 📉 **年龄和票价是重要预测特征**

## 📂 项目结构
```

ml-basics/
├── data/                   # 原始数据（从 Kaggle 下载）
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   └── titanic_eda_and_modeling.ipynb  # 主分析 Notebook
├── src/
│   ├── logistic_regression_scratch.py  # 手写逻辑回归
│   └── utils.py                        # 数据预处理工具
├── submission.csv          # Kaggle 提交文件（生成后）
└── README.md               # 本文件

```
## ▶️ 如何运行

### 1. 安装依赖
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

## 2. 准备数据

- 从 [Kaggle Titanic](https://www.kaggle.com/c/titanic/data) 下载 `train.csv` 和 `test.csv`
- 放入 `data/` 文件夹

## 3. 运行 Notebook

```bash
cd notebooks
jupyter notebook titanic_eda_and_modeling.ipynb
```

> 💡 Notebook 中包含：
>
> - 数据探索（图表）
> - 特征工程（Title、FamilySize 等）
> - Sklearn 模型（带 Imputer + StandardScaler）
> - 手写 LR 与 Sklearn 对比
> - 自动生成 `submission.csv`

## 4. 提交到 Kaggle

- 运行 Notebook 后，根目录会生成 `submission.csv`
- 上传至 [Titanic - Machine Learning from Disaster | Kaggle](https://www.kaggle.com/competitions/titanic/submissions)中的Submit to Competition

### 📊 模型性能

| 模型                                 | 本地准确率 | Kaggle Public Score |
| ------------------------------------ | ---------- | ------------------- |
| **Scikit-learn Logistic Regression** | 0.8047     | **0.7703+**         |
| 手写逻辑回归（基础版）               | 0.6162     | —                   |

> 💡 手写版用于理解原理；实际提交使用 Sklearn 版本以获得高分。

## 🧠 学习收获

- 掌握了逻辑回归的数学原理与梯度下降实现
- 学会使用 Sklearn Pipeline 处理缺失值、标准化、建模
- 理解了特征工程对模型性能的关键影响
- 完成了端到端机器学习项目流程（EDA → 预处理 → 训练 → 提交）

------

