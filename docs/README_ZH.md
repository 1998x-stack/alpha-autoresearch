# Alpha Autoresearch — 自主 Alpha 因子研究系统

> 基于 Karpathy [autoresearch](https://github.com/karpathy/autoresearch) 思想的 A 股量化因子自主研究框架。AI agent 自主发明、迭代并优化 Alpha101 风格的量化因子，通过帕累托前沿进行多目标优化。

---

## 核心概念

让 AI agent 自主进行因子研究：agent 修改 `factors.py`（编辑面），运行标准化评估引擎 `prepare.py`，由 3 个第一性原理指标（RankIC、IC IR、换手稳定性）驱动，通过帕累托前沿维护非支配因子集合。

**100 次实验 ≈ 10 分钟 | 500 次实验 ≈ 一个晚上**

---

## 快速开始

### 环境要求

- Python 3.8+
- [uv](https://docs.astral.sh/uv/) 包管理器

### 安装

```bash
cd alpha_autoresearch
uv sync
```

### 开箱即用

```bash
# 直接评估（项目内置 50 只 A 股样本数据集，约 6.7 MB）
uv run python prepare.py
```

如需使用完整 495 只股票数据集，运行：
```bash
uv run python prepare.py --build-cache   # 从 alpha101_factory 的 klines_daily 构建
```

---

## 三大核心指标

从第一性原理出发，一个因子必须满足三个条件：

| 指标 | 衡量什么 | 越高表示 |
|------|---------|---------|
| **RankIC** | 横截面预测能力（Spearman 相关系数） | 更强的预测信号 |
| **IC IR** | 预测稳定性（均值/标准差） | 更一致的日度预测 |
| **换手稳定性** | 因子排名的日度变化程度 | 更低的交易成本 |

三者构成帕累托前沿——优化一个指标通常以牺牲其他指标为代价。系统的目标是不断向外推进前沿。

---

## 项目结构

```
alpha_autoresearch/
├── prepare.py               ← 只读 — 评估引擎
│   ├── 算子库（12 个运算符）
│   ├── Factor 基类
│   ├── 统一数据集加载器
│   ├── 三指标计算（RankIC、IC IR、换手稳定性）
│   ├── 帕累托支配逻辑 + 档案管理
│   └── 主调度（编排评估流程）
├── factors.py                ← Agent 编辑面
│   └── Agent 编写 Factor* 子类
├── program.md                ← 人类编辑 — Agent 指令
│   ├── 实验循环协议
│   ├── 6 条迭代原则
│   ├── 帕累托接受标准
│   └── results.tsv 格式
├── pareto_frontier.json      ← 当前帕累托前沿（git 跟踪）
├── results.tsv               ← 实验日志（gitignore）
└── tests/                    ← 31 个测试
    ├── test_operators.py     ← 15 个算子测试
    ├── test_metrics.py       ← 7 个指标测试
    └── test_pareto.py        ← 9 个帕累托测试
```

---

## Agent 可用的运算符

| 运算符 | 签名 | 描述 |
|--------|------|------|
| `ops.cs_rank(s)` | → Series | 截面百分位排名 |
| `ops.cs_zscore(s)` | → Series | 截面 z-score |
| `ops.ts_rank(s, n)` | → Series | 时间序列百分位排名 |
| `ops.rolling_corr(s1, s2, n)` | → Series | 滚动相关系数 |
| `ops.rolling_cov(s1, s2, n)` | → Series | 滚动协方差 (ddof=0) |
| `ops.rolling_std(s, n)` | → Series | 滚动标准差 (ddof=0) |
| `ops.rolling_sum(s, n)` | → Series | 滚动求和 |
| `ops.rolling_min(s, n)` | → Series | 滚动最小值 |
| `ops.rolling_max(s, n)` | → Series | 滚动最大值 |
| `ops.delta(s, n=1)` | → Series | 差分 |
| `ops.delay(s, n=1)` | → Series | 滞后 |
| `ops.decay_linear(s, n)` | → Series | 线性衰减加权平均 |

## 可用数据列

`open, high, low, close, volume, vwap, returns, adv5, adv10, adv20, adv30, adv40, adv60, adv120, adv150, adv180`

---

## 编写新因子

```python
from prepare import Factor, ops

class Factor001(Factor):
    name = "momentum_5d"

    def compute(self, df):
        m = df.set_index(["datetime", "symbol"])
        val = ops.cs_rank(m["close"] - ops.delay(m["close"], 5))
        return Factor.as_cs_series(df, val)
```

Agent 在 `factors.py` 中编写 1-10 个 `Factor*` 类，运行 `uv run python prepare.py` 即可自动发现并评估。

---

## 帕累托逻辑

因子 A **支配** 因子 B，当且仅当 A 在所有三个指标上 ≥ B，且至少在某一指标上 > B。

| 条件 | 动作 |
|------|------|
| 新因子支配任意前沿因子 | **KEEP** — 加入前沿，移除被支配因子 |
| 新因子被所有前沿因子支配 | **DISCARD** |
| 新因子非支配但不支配任何已有因子 | **KEEP** — 扩展前沿 |
| 因子崩溃 (NaN / 异常) | **CRASH** — 记录并跳过 |

---

## 六条迭代原则（详见 program.md）

1. **攻击最弱指标** — 找出前沿的短板并针对性改进
2. **先利用再探索** — 改进现有最佳因子优于发明全新结构
3. **小突变致胜** — 改一个参数胜过推倒重来
4. **合并前沿因子** — 每 10 次实验，尝试合并两个非支配因子
5. **档案意识** — 每次会话前读取 pareto_frontier.json
6. **简洁性偏好** — 3 行代码 IC=0.05 胜过 30 行代码 IC=0.051

---

## 实验循环

```
LOOP FOREVER:
  1. 选择策略：exploit / explore / combine
  2. 修改 factors.py
  3. git commit
  4. uv run python prepare.py
  5. grep 提取结果（factor / rank_ic / ic_ir / turnover_stability / status）
  6. 根据帕累托逻辑决定 KEEP / DISCARD / CRASH
  7. 记录到 results.tsv（不要提交）
  8. 连续 5 次 discard 后切换策略
  9. 永不停止
```

---

## 运行测试

```bash
uv run pytest tests/ -v    # 31 个测试
```

---

## 配置

| 环境变量 | 默认值 | 描述 |
|----------|--------|------|
| `ALPHA101_DATA_ROOT` | `../alpha101_factory/data` | 数据根目录 |

---

## 设计约束

| 约束 | 详情 |
|------|------|
| 不可修改 | `prepare.py` |
| 不可添加包 | 仅限 pyproject.toml 中的依赖 |
| 不可提交 | `results.tsv` |
| 因子预算 | 每次实验最多 10 个因子 |
| 安全超时 | 60 秒墙钟时间（防止意外耗时因子） |
| GPU | 不需要（纯 CPU 计算） |

---

## 文档

- [README.md](../README.md) — English overview
- [REPORT.md](REPORT.md) — 30 轮实验详细报告（中文）
- [spec.md](../spec.md) — 完整系统规格
- [CONTEXT.md](../CONTEXT.md) — 领域术语表
- [program.md](../program.md) — Agent 指令文件
