# Midway Report — 实验记录

> 本文档用于在实验过程中同步记录结果，最终作为撰写 midway report 的素材来源。
> 与 [`midway_report_plan.md`](midway_report_plan.md) 配合使用：plan 定义"做什么"，本文档记录"做了什么、结果如何"。

### 关联文档

| 文档 | 作用 | 链接 |
|------|------|------|
| Proposal Summary | 项目设计全貌 (问题定义、数据、模型、Related Work) | [`proposal_summary.md`](proposal_summary.md) |
| Execution Plan | 执行蓝图 (notebook 步骤、report section 写作指南、timeline) | [`midway_report_plan.md`](midway_report_plan.md) |
| Grading Criteria | 评分标准原文 | [`midway_report_grading_criteria.md`](midway_report_grading_criteria.md) |

> **Problem Statement / Related Work** 素材直接引用 `proposal_summary.md` 的 Section 1 和 Section 9。如果在实验中发现需要修改或补充 (例如新增参考论文、调整问题表述)，在下方 §8 问题与决策日志中记录变更。

---

## 1. EDA 发现 (→ Report: Dataset, Experimental Design)

> Notebook: `notebooks/01_eda_preprocessing.ipynb`。以下 cell 编号对应 notebook 实际顺序。

### 1.1 数据概览 (cells 2–5)

| 项目 | 值 |
|------|-----|
| 数据来源 | `data/processed/matches.parquet` |
| 行数 | 100,041 |
| 列数 | 472 |
| Label 分布 | label=1: 50,538 (50.52%) / label=0: 49,503 (49.48%) |
| 内存占用 | 378 MB |
| dtypes | 395 float64, 77 int64 |
| NaN 情况 | 363 列有 NaN, 每列最高 18 行 (0.018%) — 集中在 `*_per_soul` 字段 |
| Unique heroes | 38 |
| Unique account_ids | 407,144 |

### 1.2 Leakage 分析与列审计 (cells 7–8)

**Drop 决策** (共 55 列):

| 类别 | 数量 | 列 | 原因 |
|------|------|---|------|
| leakage | 2 | `winning_team`, `duration_s` | 赛后信息 |
| identifier | 26 | `match_id`, `start_time`, `account_id`×12, `player_slot`×12 | 非特征 |
| zero variance | 1 | `game_mode` | 全部为 1 |
| redundant | 2 | `average_badge_team0/1` | 与 `avg_mmr_rank` r=0.993 (cell 7) |
| temporal | 12 | `last_played`×12 | 绝对时间戳，不可泛化 |
| fixed structure | 12 | `assigned_lane`×12 | 2-2-2 固定分配，无预测信号 (cell 15) |

**保留但标记**:
- `player_hero_wr` (r=0.778 with label, cell 7): API 快照含赛后数据，存在时序泄漏风险。保留用于 baseline，后续做 ablation 验证影响。

**Drop 后**: 472 − 55 − 1(label) = **416 features**（实测 `df_clean.shape = (100041, 416)`）

### 1.3 特征分布 (cells 10–15)

以下分析均在 leakage drop 后的 416 列上进行。

**Numeric 偏度** — per-player float64 (cell 11, t0_p0 代表; cell 12 slot 一致性验证: max skew_std=0.77):

| skewness 区间 | 特征数 | 代表特征 | 需要 log1p? |
|---|---|---|---|
| \|skew\| > 6 | 6 | deaths (9.0), kills (7.6), matches_played (7.5), wins (7.1), assists (7.0), time_played (6.7) | **Yes** — 累积量 |
| 1 < \|skew\| < 3 | 8 | accuracy (-2.3), denies_per_min (1.7), ending_level (-1.6) 等 | 可选 |
| \|skew\| < 1 | 剩余 | damage_per_min, networth_per_min, mmr_rank 等 rate/average 类 | No |

**Team-level agg 偏度** (cell 11, 全部 20 列):

| skewness 区间 | 特征 |
|---|---|
| \|skew\| > 3 | `avg_matches_played` (t0: 3.28, t1: 3.40) |
| 1 < \|skew\| < 2 | `avg_ending_level` (t0: -1.64, t1: -1.74) |
| \|skew\| < 1 | 其余 16 列 |

log1p 目标: per-player 累积统计量 (6 类 × 12 players = 72 列) + team-level `avg_matches_played` (2 列)。

**Categorical 列处理**:

| 列 | EDA 依据 | 处理 |
|---|---|---|
| `hero_id` ×12 | 38 unique, max/min 出场比 4.2x, 无稀有类 (cell 14) | **one-hot** → 38×12=456 binary cols |
| `assigned_lane` ×12 | 100% matches 2-2-2 固定; 所有 hero max lane pref <40% (cell 15) | **drop** (已含在 leakage 的 55 列中) |
| `global_hero_wins/matches` ×12 | skew <1, slot 间一致 | 保留 numeric |

**NaN 处理**: `*_per_soul` 系列最多 18 行缺失 (0.018%) → median imputation

### 1.4 相关性分析 (cells 17–18)

**Team-level numeric vs label** (cell 17):

| Rank | Feature | r |
|------|---------|---|
| 1 | `t0_avg_player_hero_wr` | +0.778 |
| 2 | `t1_avg_player_hero_wr` | −0.769 |
| 3 | `t0_avg_assists_per_min` | +0.483 |
| 4 | `t0_avg_networth_per_min` | +0.480 |
| 5 | `t1_avg_assists_per_min` | −0.468 |

t0 特征正相关 / t1 负相关，对称结构符合预期。`player_hero_wr` 两端最极端。

**Hero win rate** (cell 18): 各 hero 胜率范围 [0.42, 0.57]，差距 ~14 pp。hero 13/2/66 最强，hero 52/63/14/15 最弱。信号存在但不强，one-hot 保留合理。

### 1.5 数据分割

> **统一方案**: `GroupShuffleSplit` by `account_id`, **train 70% / val 10% / test 20%**, stratified by `label`。
> 同一 `account_id` 的所有比赛必须落在同一 split 中，防止 player-level leakage。

- [ ] GroupShuffleSplit 完成

| 分割 | 目标比例 | 实际行数 | 实际 Label=1 占比 | 说明 |
|------|---------|---------|------------------|------|
| Train | 70% | | | |
| Validation | 10% | | | |
| Test | 20% | | | |

- account_id 分割验证: train/val/test 之间是否有 account 重叠? ____

---

## 2. 预处理 Pipeline (→ Report: Experimental Design)

### 2.1 预处理步骤记录

按实际执行顺序记录，每步标注是否完成:

- [ ] **Step 1**: Drop 非预测列 (`match_id`, `winning_team`, `duration_s`, `start_time`, `game_mode`, `account_id` ×12, `player_slot` ×12)
- [ ] **Step 2**: Feature engineering
  - `team_skill_gap` = `t0_avg_mmr_rank` − `t1_avg_mmr_rank`
  - `experience_gap` = `t0_avg_matches_played` − `t1_avg_matches_played`
  - 其他新增特征: ____
- [ ] **Step 3**: Log1p transform (skewed features)
  - 应用的列: ____
  - 变换前后 skewness 对比: ____
- [ ] **Step 4**: Z-score standardization (fit on train only)
- [ ] **Step 5**: 其他 (如有)

### 2.2 预处理后特征矩阵

| 项目 | 值 |
|------|-----|
| 最终特征数 | |
| Train shape | |
| Val shape | |
| Test shape | |

---

## 3. Baseline Models (→ Report: Approach, Partial Results)

### 3.1 模型训练记录

对每个模型记录: 参数设置、训练时间、train/val 性能。

#### DummyClassifier (Majority class)

- 参数: `strategy="most_frequent"`
- 训练时间: ____

| Metric | Train | Validation |
|--------|-------|------------|
| Accuracy | | |
| ROC-AUC | | |
| Macro Precision | | |
| Macro Recall | | |
| Macro F1 | | |

#### Logistic Regression (default)

- 参数: `C=1.0, max_iter=1000, solver=____`
- 训练时间: ____
- 收敛状态: ____

| Metric | Train | Validation |
|--------|-------|------------|
| Accuracy | | |
| ROC-AUC | | |
| Macro Precision | | |
| Macro Recall | | |
| Macro F1 | | |

#### Random Forest (default)

- 参数: `n_estimators=100, max_depth=None, ...`
- 训练时间: ____

| Metric | Train | Validation |
|--------|-------|------------|
| Accuracy | | |
| ROC-AUC | | |
| Macro Precision | | |
| Macro Recall | | |
| Macro F1 | | |

- Train vs Val gap → 过拟合程度: ____

#### XGBoost (default)

- 参数: `n_estimators=100, max_depth=6, learning_rate=0.3, ...`
- 训练时间: ____

| Metric | Train | Validation |
|--------|-------|------------|
| Accuracy | | |
| ROC-AUC | | |
| Macro Precision | | |
| Macro Recall | | |
| Macro F1 | | |

- Train vs Val gap → 过拟合程度: ____

### 3.2 Sanity Checks

- [ ] 所有模型 > DummyClassifier? ____
- [ ] 是否有模型 accuracy > 72% 或 AUC > 0.78? (leakage 警告线) ____
- [ ] Train vs Val gap 合理? ____

---

## 4. Multi-Split Evaluation (→ Report: Partial Results)

### 4.1 实验设置

- 评估方式: ____ (e.g., 10 random splits / 5-fold CV)
- 每次 split 方法: GroupShuffleSplit by account_id
- 记录 metrics: ROC-AUC, Accuracy, Macro Precision, Macro Recall, Macro F1

### 4.2 汇总结果表

**TODO**: 跑完 multi-split 后填写

| Model | ROC-AUC (mean±std) | Accuracy (mean±std) | Macro F1 (mean±std) | Macro Precision (mean±std) | Macro Recall (mean±std) |
|-------|-------------------|--------------------|--------------------|---------------------------|------------------------|
| DummyClassifier | | | | | |
| Logistic Regression | | | | | |
| Random Forest | | | | | |
| XGBoost (default) | | | | | |

Best model per metric (bold):

### 4.3 可视化

> **页数预算提醒**: 报告 ≤5 pages, Partial Results 建议 ~0.5 page。精选 1–2 张最有说服力的图即可 (推荐: boxplot + summary table)。

- [ ] Boxplot: ROC-AUC across splits — 保存路径: ____ (**优先入选报告**)
- [ ] ROC curves overlaid — 保存路径: ____ (备选, 如页数允许)
- [ ] Bar chart: mean metrics comparison — 保存路径: ____ (备选)

### 4.4 结果分析

```
(在此记录关键观察)
- 哪个模型整体表现最好?
- LR vs tree-based 差距多大? (→ H2)
- 是否有 leakage 迹象?
- 结果的 variance 如何?
```

---

## 5. 假设验证进展 (→ Report: Hypotheses, Partial Results)

对照 plan 中定义的假设，记录当前实验能得出的初步结论。

### H1: Pre-match player statistics achieve ROC-AUC ≥ 0.60 on unseen matches

> 非显而易见的阈值: 基于 Semenov et al. 在 Dota 2 上仅用 draft features 达到 ~60% accuracy 的先例。

- Best model ROC-AUC: ____
- Best model Accuracy: ____
- AUC ≥ 0.60? ____
- **判断标准**: AUC ≥ 0.60 → supported; 0.50–0.60 → weak evidence; ≤ 0.50 → rejected
- **初步结论**: ____

### H2: Tree-based models outperform LR by ≥ 2% ROC-AUC

> 理由: player-stat interactions (MMR × hero winrate) 是非线性的, linear boundary 无法捕获。

- RF ROC-AUC vs LR ROC-AUC: ____ vs ____ (Δ = ____)
- XGBoost ROC-AUC vs LR ROC-AUC: ____ vs ____ (Δ = ____)
- **判断标准**: Δ ≥ 2% → supported; Δ < 2% → not supported (nonlinear interactions may be weak)
- **初步结论**: ____

### H3: Team-level features rank among MI top-10

> 理由: inter-team imbalance signal (skill gap, experience gap) 比个体原始统计更具预测力。

- **状态**: ☐ 未开始 / ☐ 进行中 / ☐ 已完成
- MI top-10 features:
  1. ____
  2. ____
  3. ____
  4. ____
  5. ____
  6. ____
  7. ____
  8. ____
  9. ____
  10. ____
- `team_skill_gap` rank: ____
- `experience_gap` rank: ____
- **判断标准**: 至少一个 team-level feature 进入 top-10 → supported
- **初步结论**: ____

### H4: Log transform improves LR by ≥ 1% AUC

> 理由: LR 假设特征近似正态, 右偏特征 log 变换后更接近对称分布。

- **状态**: ☐ 未开始 / ☐ 进行中 / ☐ 已完成
- LR AUC without log: ____
- LR AUC with log: ____
- Difference: ____
- **判断标准**: Δ ≥ 1% → supported; Δ < 1% → log transform not critical for this dataset
- **初步结论**: ____

---

## 6. 初步调参记录 (→ Report: Hyperparameters Appendix)

如果在 midway 阶段进行了部分调参，记录在此。否则标注 "deferred to final report"。

### Logistic Regression

- **Status**: ☐ 未开始 / ☐ 已完成 / ☐ deferred
- Tuning method: Random search / Grid search
- Validation: 5-fold CV on train
- HP: `C` ∈ {0.01, 0.1, 1, 10, 100}
- Best `C`: ____
- Best CV ROC-AUC: ____

### Random Forest

- **Status**: ☐ 未开始 / ☐ 已完成 / ☐ deferred
- HP tuned: ____
- Best params: ____
- Best CV ROC-AUC: ____

### XGBoost

- **Status**: ☐ 未开始 / ☐ 已完成 / ☐ deferred
- HP tuned: ____
- Best params: ____
- Best CV ROC-AUC: ____

---

## 7. Future Experiments 进度追踪 (→ Report: Design of Future Experiments)

> **完整实验设计定义** 见 [`midway_report_plan.md` Section 9](midway_report_plan.md#section-9-design-of-future-experiments-20-pts)。
> 本节仅追踪进度、增量变更和实际执行结果。报告撰写时以 Plan 为 source of truth, 本节提供补充素材。

### 进度总览

| # | Experiment | 绑定假设 | 目标 Artifact | Status | 备注 |
|---|-----------|---------|--------------|--------|------|
| 1 | HP Tuning (LR/RF/XGBoost) | H1, H2 | `notebooks/04_hp_tuning.ipynb` | ☐ 未开始 | |
| 2 | Log Transform Ablation | H4 | `notebooks/05_ablations.ipynb` | ☐ 未开始 | |
| 3 | Feature Selection (MI) | H3 | `notebooks/05_ablations.ipynb` | ☐ 未开始 | |
| 4 | Feature Engineering (KDA + Synergy) | H3 | `notebooks/05_ablations.ipynb` | ☐ 未开始 | |
| 5 | Discretization | — | `notebooks/05_ablations.ipynb` | ☐ 未开始 | |
| 6 | Final Evaluation (test set) | all | `notebooks/06_final_evaluation.ipynb` | ☐ 未开始 | |

### 增量变更记录

> 如果实验过程中需要调整 Plan 中定义的实验设计 (例如修改 search space, 增减实验), 在此记录变更及原因。

| 日期 | 变更内容 | 原因 |
|------|---------|------|
| | | |

### 实际执行结果

> 每完成一个 future experiment, 在此记录结果 (补充 Plan 中的预期 vs 实际)。

**Experiment 1 — HP Tuning**:
```
(待填写: best params, best CV AUC, comparison table)
```

**Experiment 2 — Log Transform Ablation**:
```
(待填写: with vs without AUC, delta)
```

**Experiment 3 — Feature Selection (MI)**:
```
(待填写: MI ranking table, best k, AUC comparison)
```

**Experiment 4 — Feature Engineering**:
```
(待填写: new features added, AUC before/after)
```

**Experiment 5 — Discretization**:
```
(待填写: binned vs continuous AUC comparison)
```

**Experiment 6 — Final Evaluation**:
```
(待填写: final test set metrics for all models)
```

---

## 8. 问题与决策日志

在实验过程中遇到的问题和做出的决策，按时间记录。

| 日期 | 问题 / 决策 | 结论 / 解决方案 |
|------|-----------|----------------|
| | | |

---

## 9. 生成报告 Checklist

完成实验后，对照此表确认报告所需素材是否齐全。

| Report Section | 所需素材 | 来源 | 状态 |
|---------------|---------|------|------|
| Problem Statement (5) | 问题描述 + 动机 | [`proposal_summary.md` Sec 1](proposal_summary.md#1-problem-description) | ☐ |
| Related Work (5) | ≥3 papers 摘要 + 关联 | [`proposal_summary.md` Sec 9](proposal_summary.md#9-related-work) | ☐ |
| Approach (15) | Pipeline + models + novelty (**注意**: what & how) | 本文档 §2 + §3 + [Plan Sec 3](midway_report_plan.md#section-3-approach-15-pts) | ☐ |
| Rationale (10) | "为什么" 论证 (**注意**: why, 与 Approach 不重叠) | [`proposal_summary.md`](proposal_summary.md) + 本文档 §3 | ☐ |
| Dataset(s) (5) | EDA 统计 + 特征组 | 本文档 §1 | ☐ |
| Hypotheses (5) | H1–H4 假设文本 (须 non-obvious) | 本文档 §5 + [Plan Sec 6](midway_report_plan.md#section-6-hypotheses-5-pts) | ☐ |
| Experimental Design (10) | Pipeline steps + split + code artifacts | 本文档 §2 | ☐ |
| Partial Results (10) | 汇总表 + 图 | 本文档 §4 | ☐ |
| Design of Future Experiments (20) | 详细实验规划 (hypothesis binding + procedures + artifacts) | [Plan Sec 9](midway_report_plan.md#section-9-design-of-future-experiments-20-pts) + 本文档 §7 增量 | ☐ |
| HP Appendix (10) | 5 elements × 3 classifiers + rationale | 本文档 §6 + [Plan Sec 10](midway_report_plan.md#section-10-hyperparameters-appendix-10-pts) | ☐ |
| Plan of Activities (5) | Timeline + roles | [Plan 时间估算](midway_report_plan.md#建议执行顺序与时间估算) | ☐ |
| References | BibTeX entries | Overleaf | ☐ |
| Format | ACM, ≤5 pages | Overleaf | ☐ |
