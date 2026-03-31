# Deadlock Match Forecast — Midway Report 执行计划

## 当前状态

- **数据**: `data/processed/matches.parquet` — 100,041 rows × 472 columns, 内存 377.8 MB
- **预处理代码**: `src/preprocess.py` 已完成 (raw JSON → flat parquet, per-player + team-level features)
- **Notebook / 模型代码**: 尚未创建 (仅有 `notebooks/example_template/` 参考模板)
- **报告工具**: Overleaf (ACM format, 5 pages max)

### 已有数据文件详情

**原始数据 (`data/raw/`)**:

| 目录 | 文件数 | 内容 |
|------|--------|------|
| `raw/hero_stats/` | 3 | `hero_stats.json`, `hero_counter_stats.json`, `hero_synergy_stats.json` (全局英雄数据) |
| `raw/matches/` | 100,042 | 每场比赛一个 JSON (slim mode, 仅 pre-match 字段) + `checkpoint.json` |
| `raw/match_list/` | 104 | 批量元数据 (103 个 batch + checkpoint) |
| `raw/player_stats/` | 4,074 | `hero_stats_NNNN.json` + `mmr_NNNN.json` 批量文件 + checkpoint |

**处理后数据 (`data/processed/`)**:

| 文件 | 大小 | 说明 |
|------|------|------|
| `matches.parquet` | 244 MB | 主特征矩阵, 100,041 rows × 472 cols |

> 注意: `matches_meta.json` (列描述/分割信息) 在 `data/README.md` 中提到但实际**尚未生成**。

### 数据关键统计

- **Label 分布**: label=1 (Team0 wins): 50,538 (50.52%) / label=0: 49,503 (49.48%) — 近乎完美平衡
- **NaN 情况**: 363 个列含 NaN，但每列缺失量极低 (最高 18 行 / 0.02%), 主要集中在 `damage_per_soul`, `obj_damage_per_soul`, `damage_taken_per_soul` 等字段
- **列结构**: 7 个 match-level 列 + 12 × 37 = 444 个 per-player 列 + 2 × 10 = 20 个 team-aggregation 列 + 1 label
- **数据类型**: 395 float64, 77 int64
- **game_mode**: 全部为 1 (单一值, 零方差, 应直接 drop)
- **Unique heroes**: 38
- **Unique account_ids**: 407,144 (平均每人 2.9 场, 中位数 2 场, 最多 54 场, >5 场的有 51,265 人)
- **关键特征范围**:
  - `avg_mmr_rank`: mean=63.5, std=26.3, range=[12, 116]
  - `avg_player_hero_wr`: mean=0.513, std=0.19, range=[0, 1]
  - `duration_s`: mean=2225s (~37min), range=[230s, 8198s]

### Notebook 中需注意的数据问题

1. **NaN 处理**: 虽然极少但存在, 需要选择 imputation 策略 (mean/median 或直接 drop)
2. **game_mode 零方差**: 直接 drop
3. **duration_s**: 是 leakage (比赛后才知道时长), 必须 drop
4. **account_id 重复出场**: 需要 `GroupShuffleSplit` 按 account_id 分割, 防止同一玩家出现在 train 和 test
5. **hero_counter_stats.json / hero_synergy_stats.json**: 原始数据中已有, 但 `preprocess.py` 尚未将其加入特征矩阵 (pair-wise matrix, 非 per-player 结构), 可作为 future feature engineering
6. **match_mode**: 当前数据全部是同一模式, `preprocess.py` 输出不含此列

---

## Part A: Notebook 代码

参考 `example_template` 的 Part1/Part2/Part3 结构，按项目需求调整。

### Notebook 1: EDA + Data Preprocessing

**文件**: `notebooks/01_eda_preprocessing.ipynb`
**对应**: template Part1 | rubric: Dataset(5), Experimental Design(10 partial)

1. **Load data**: `pd.read_parquet("data/processed/matches.parquet")`
2. **Basic statistics**: shape, dtypes, `df.describe()`, NaN check
3. **Target analysis**: label distribution bar chart, confirm ~50/50 balance
4. **Feature distribution EDA**:
   - Histogram / boxplot: key numeric features (`t0_avg_mmr_rank`, `t0_avg_player_hero_wr`, `t0_avg_kills_per_min` etc.)
   - Skewness check: identify right-skewed features for log transform (per proposal: `matches_played`, `kills`, `deaths`, etc.)
   - Correlation heatmap: team-level aggregation features vs label
5. **Data validation**:
   - Confirm no match-level leakage columns remain (drop `match_id`, `winning_team`, `duration_s`, `start_time`)
   - Check `game_mode` distribution, decide if to encode or drop
   - Identify categorical columns (`hero_id`, `assigned_lane`) vs numeric
6. **Data splitting** (统一比例: **70 / 10 / 20**):
   - Collect all unique `account_id` from the 12 player columns
   - `GroupShuffleSplit` by `account_id`, 一次性分为 train 70% / val 10% / test 20%
   - Stratified by `label` 以保持各 split 中 ~50/50 class 比例
   - 验证: 三个 split 之间 account_id 无重叠
7. **Preprocessing pipeline** (fit on train only):
   - Drop non-predictive columns
   - Feature engineering: `team_skill_gap = t0_avg_mmr_rank - t1_avg_mmr_rank`, `experience_gap = t0_avg_matches_played - t1_avg_matches_played`
   - Log1p transform on skewed features
   - Z-score standardization (all numeric)
   - Export preprocessed train / val / test for Notebook 2

### Notebook 2: Models + Tuning

**文件**: `notebooks/02_models.ipynb`
**对应**: template Part2 | rubric: Approach(15), Partial Results(10)

1. **Import preprocessed data** from Notebook 1 (or reproduce pipeline)
2. **Baseline models** (train on training set, evaluate on validation set):
   - **Majority-class DummyClassifier**: establish floor (accuracy ~50.52%, ROC-AUC = 0.50)
   - **Logistic Regression** (C=1.0, default): linear baseline
   - **Random Forest** (default params): nonlinear baseline
   - **XGBoost** (default params): gradient boosting baseline
3. **Sanity checks**: train accuracy vs val accuracy, check for gross overfitting
4. **Initial hyperparameter tuning** (if time permits, otherwise mark as future):
   - Logistic Regression: `C` ∈ {0.01, 0.1, 1, 10, 100} via 5-fold CV on train
   - RF / XGBoost tuning deferred to final report
5. **Reusable evaluation functions**: `evaluate_model(model, X_train, X_test, y_train, y_test)` → dict of all metrics

### Notebook 3: Evaluation + Results

**文件**: `notebooks/03_evaluation.ipynb`
**对应**: template Part3 | rubric: Partial Results(10)

1. **Multi-split evaluation**: run each model on 10+ random train/test splits, collect metrics
2. **Metrics per model**: ROC-AUC, Accuracy, Macro Precision, Macro Recall, Macro F1
3. **Visualizations**:
   - Boxplot: ROC-AUC across splits for each model
   - Bar chart: mean metrics comparison
   - ROC curves overlaid (one per model)
4. **Summary table**: mean ± std for each metric per model (bold best)
5. **Brief analysis**: which model performs best, any signs of leakage (>72% accuracy?)

---

## Part B: Midway Report (Overleaf, ACM Format, ≤5 pages)

### Section 1: Problem Statement (5 pts)

- **来源**: proposal Section 1 (Abstract + Prediction Task)
- **内容**: Deadlock 比赛预测问题, 为什么重要 (玩家体验 / 电竞博彩), binary classification, target = `match_outcome`
- **长度建议**: ~0.3 page

### Section 2: Related Work (5 pts)

- 至少 3 篇 scholarly works (proposal 已有 3 篇):
  1. **Gu et al., "NeuralAC" (AAAI 2021)** — attention-based team cooperation / competition modeling
  2. **Yang et al., "Identifying Patterns in Combat" (FDG 2014)** — MOBA match prediction domain background
  3. **Semenov et al. (2016)** — logistic regression on Dota 2 draft features
- 每篇需要: 解释内容 + 与本项目的关联
- **长度建议**: ~0.4 page

### Section 3: Approach (15 pts)

- ML techniques with specific details:
  - Pipeline: data collection API → preprocessing → feature engineering → model training
  - Models: Logistic Regression, Random Forest, XGBoost (with / without tuning)
  - Feature engineering: `team_skill_gap`, `experience_gap`, log transforms, z-score standardization
- **Novelty** (需要在报告中具体论证，不能仅说 "novel dataset"):
  1. **Domain transfer**: 将 Dota 2 已验证的 pre-match prediction 方法论迁移到 Deadlock 这一新游戏, Deadlock 有不同的游戏机制 (6v6 而非 5v5, 不同的经济系统和角色设计), 需要验证 Dota 2 的结论是否可迁移
  2. **End-to-end data pipeline**: 自建 API 采集 → 清洗 → 特征工程 pipeline, 处理真实 API 的缺失值、私人档案、时序一致性等问题, 非课内预清洗数据集
  3. **Team-level feature engineering from raw per-player stats**: 从 12 个玩家的独立统计聚合出 team-level 差异特征 (`skill_gap`, `experience_gap`), 将个体数据转化为团队对抗信号
- **长度建议**: ~0.8 page

> **写报告注意**: Approach 侧重 "what & how" (做了什么、怎么做的), 与 Rationale 的 "why" (为什么这么做) 明确区分, 避免内容重叠。rubric 中两项描述文本相同是 copy-paste 错误, Plan 已做正确解读。

### Section 4: Rationale (10 pts)

- 为什么选择这些 techniques:
  - **LR**: interpretable + comparable to Semenov et al. (2016)
  - **RF**: captures nonlinear interactions + provides feature importance
  - **XGBoost**: SOTA on tabular data + with / without tuning comparison isolates tuning contribution
  - **Z-score**: required for LR / SVM / k-NN; consistent pipeline across models
  - **GroupShuffleSplit**: prevent player-level leakage across train / test
- **长度建议**: ~0.5 page

### Section 5: Dataset(s) (5 pts)

- **来源**: proposal Section 2 + EDA notebook results
- 100,041 matches, 472 features, ~50/50 balanced
- HuggingFace link: https://huggingface.co/datasets/sdxshuai/deadlock-match-forecast
- Feature groups summary (per-player hero stats ×12, MMR ×12, global hero stats ×12, team aggregations ×2)
- Data validation findings
- **长度建议**: ~0.3 page

### Section 6: Hypotheses (5 pts)

> **重要**: rubric 要求的是 hypotheses (可验证假设), 不仅是 research questions。需要将 RQ 转化为具体假设。

- **H1**: Pre-match player statistics can predict Deadlock match outcomes with ROC-AUC ≥ 0.60 on unseen matches — a non-trivial threshold based on comparable Dota 2 prediction work (Semenov et al. achieved ~60% accuracy with draft-only features)
- **H2**: Tree-based models (RF, XGBoost) will outperform Logistic Regression by ≥ 2% ROC-AUC, because player-stat interactions (e.g., MMR × hero winrate) are nonlinear and cannot be captured by a linear decision boundary
- **H3**: Team-level aggregated features (`team_skill_gap`, `experience_gap`) will rank among the top-10 most important features by mutual information, indicating that inter-team imbalance is more predictive than individual raw player stats
- **H4**: Log transformation on right-skewed features (matches_played, kills, etc.) will improve Logistic Regression ROC-AUC by ≥ 1%, because LR assumes approximately Gaussian-distributed features

**长度建议**: ~0.3 page

### Section 7: Experimental Design (10 pts)

- Preprocessing pipeline (step-by-step, reproducible)
- Data splitting strategy (`GroupShuffleSplit` by `account_id`, train 70% / val 10% / test 20%, stratified by label)
- Model training procedure
- Software artifacts: notebooks, `src/preprocess.py`
- **长度建议**: ~0.5 page

### Section 8: Partial Results (10 pts)

- **来源**: Notebook 3 的输出
- Comparison table: all baselines + metrics (ROC-AUC, Accuracy, Macro F1)
- Key visualizations: boxplot, ROC curves
- Brief discussion: what works, what doesn't
- **长度建议**: ~0.5 page

### Section 9: Design of Future Experiments (20 pts)

> **这是最高分值的评分项 (20 pts)**。rubric 要求 "clearly explains the procedures/experiments" + "software artifacts"。
> 每个 future experiment 都必须: (1) 绑定到 hypothesis, (2) 给出具体步骤和代码实现方式, (3) 定义预期结果和判断标准。

**Experiment 1: Hyperparameter Tuning** → validates H1, H2
- **Procedure**: `sklearn.model_selection.RandomizedSearchCV(estimator, param_distributions, n_iter=N, cv=5, scoring='roc_auc')`
- LR: `C` ∈ `scipy.stats.loguniform(0.01, 100)`, 20 iter
- RF: `max_depth` ∈ {3, 5, 10, None} × `n_estimators` ∈ {100, 200, 500}, 20 iter
- XGBoost: `learning_rate` ∈ {0.01, 0.1, 0.3} × `max_depth` ∈ {3, 5, 10}, `n_estimators`=200 fixed, 30 iter
- **Judgment**: If tuned XGBoost > tuned LR by ≥ 2% AUC → supports H2; if best model AUC ≥ 0.60 → supports H1
- **Complexity**: worst case 30 iter × 5-fold × 3 feature selection = 450 runs

**Experiment 2: Log Transform Ablation** → validates H4
- **Procedure**: Train LR with identical pipeline except toggle log1p on skewed features; compare on val set
- **Code**: `np.log1p(X_train[skewed_cols])` vs raw; pipeline variant via `sklearn.pipeline.Pipeline`
- **Judgment**: If AUC improves ≥ 1% → H4 supported; else log transform is not critical for this dataset

**Experiment 3: Feature Selection (MI)** → validates H3
- **Procedure**: `sklearn.feature_selection.SelectKBest(mutual_info_classif, k=k)` for k ∈ {15, 20, 30} vs no selection
- Run on both LR and RF; compare val AUC
- **Judgment**: If `team_skill_gap` / `experience_gap` appear in MI top-10 → H3 supported; report feature ranking table

**Experiment 4: Feature Engineering — KDA + Hero Synergy** → extends H3
- KDA = (kills + assists) / deaths per player; aggregate to team avg
- Hero synergy/counter scores from `hero_synergy_stats.json` / `hero_counter_stats.json`
- **Procedure**: Add features → retrain best model → compare AUC before/after
- **Judgment**: If AUC improves → these team composition features capture information beyond individual stats

**Experiment 5: Discretization** → preprocessing sensitivity analysis
- Bin kills/deaths/assists/networth → high/medium/low via quantile binning
- Compare LR + RF performance with vs without binning
- **Judgment**: If no improvement → continuous features are sufficient

**Experiment 6: Final Evaluation**
- Freeze all decisions from Experiments 1–5
- Evaluate on held-out 20% test set (never touched before)
- Report: ROC-AUC, Accuracy, Macro Precision, Macro Recall, Macro F1
- Compare against baselines and leakage thresholds (>72% accuracy → investigate)

**Software Artifacts** (报告中需明确说明将创建的代码产物):
- `notebooks/04_hp_tuning.ipynb` — Experiments 1 (HP tuning for all 3 models)
- `notebooks/05_ablations.ipynb` — Experiments 2–5 (preprocessing ablations + feature engineering)
- `notebooks/06_final_evaluation.ipynb` — Experiment 6 (test set evaluation, frozen pipeline)
- `src/preprocess.py` — 已有, 作为 data processing artifact 在报告中引用

**长度建议**: ~0.8 page

### Section 10: Hyperparameters Appendix (10 pts)

每个 classifier 需要 5 个 required elements + rationale:

| Element | Logistic Regression | Random Forest | XGBoost |
|---------|-------------------|---------------|---------|
| **Evaluation metric** | ROC-AUC | ROC-AUC | ROC-AUC |
| **Tuning method** | Random search (20 iter) | Random search (20 iter) | Random search (30 iter) |
| **Validation approach** | 5-fold CV on train | 5-fold CV on train | 5-fold CV on train |
| **HPs to tune** | `C` | `max_depth`, `n_estimators` | `max_depth`, `learning_rate` |
| **HP ranges** | C: log-uniform [0.01, 100] | max_depth: {3,5,10,None}, n_estimators: {100,200,500} | max_depth: {3,5,10}, lr: {0.01,0.1,0.3} |

**Rationale**: 每个选择都需要简要说明理由 (e.g., 为什么选 ROC-AUC? 为什么 random search? 为什么这些 HP ranges?)

> **已确认**: 5 required elements (来自课程 slides) = Evaluation metric / Tuning method / Validation approach / HPs to tune / HP ranges。上表完全匹配。

### Section 11: Plan of Activities (5 pts)

- Step-by-step remaining tasks with estimated timeline
- Role assignments (Person 1–4, as in proposal)
- **长度建议**: ~0.3 page

### 格式要求

- ACM Format (Overleaf ACM template)
- ≤ 5 pages (page limit not exceeded)
- References correctly formatted (BibTeX)

### 页数预算 (风险: 偏紧)

各 section 建议页数加总约 4.7–5.2 pages, 加上 References / 标题 / Abstract 后有超 5 页风险。**压缩策略**:

- HP Appendix 用紧凑表格 (如 Plan Section 10 的格式), 不用段落叙述, 可控制在 ~0.3 page
- Partial Results 可视化精选 1–2 张最有说服力的图 (推荐: 1 张 boxplot + 1 张 summary table), 不需要 ROC curves overlaid
- Approach 和 Rationale 注意不重叠, 合并写可省 ~0.2 page
- Problem Statement + Dataset 可适当压缩, 各 ~0.2 page 即可

---

## Part C: Proposal 与 Rubric 冲突分析

**结论: 无实质冲突**, proposal 的 midterm milestone 可以覆盖绝大部分 rubric 评分项。需要额外注意以下补充项:

| Rubric 项目 | Proposal Milestone 是否覆盖 | 补充说明 |
|---|---|---|
| Problem Statement (5) | 覆盖 (proposal abstract) | 直接引用 |
| Related Work (5) | 覆盖 (3 篇已有) | 直接引用 |
| Approach (15) | 覆盖 (pipeline + models) | 需要更详细的技术描述 |
| Rationale (10) | 部分覆盖 | 需要补充 "为什么" 的论证 |
| Dataset(s) (5) | 覆盖 (EDA done) | 来自 notebook |
| **Hypotheses (5)** | **未覆盖** | **需要新增: 将 RQ 转化为可验证假设 H1–H3** |
| Experimental Design (10) | 覆盖 (preprocessing + training) | 需要写成可复现步骤 |
| Partial Results (10) | 覆盖 (comparison table) | 来自 notebook |
| **Design of Future Experiments (20)** | **部分覆盖** | **最高分值! 需要详细规划未来的调参 / 特征选择 / 消融实验** |
| Hyperparameters Appendix (10) | 部分覆盖 (proposal 有 grid) | 需要按 5-element 格式整理, 含 rationale |
| Plan of Activities (5) | 覆盖 (team roles) | 需要更细的 timeline |

---

## 建议执行顺序与时间估算

### Week 1

| 负责人 | 任务 | 预计时间 |
|--------|------|----------|
| Person 1 (EDA & Report) | Notebook 1: EDA + preprocessing | 2–3 days |
| Person 4 (Feature Engineering) | Feature engineering: `team_skill_gap`, `experience_gap` | 1–2 days |
| All | Overleaf 建立 ACM 模板, 分配 sections | 0.5 day |

### Week 2

| 负责人 | 任务 | 预计时间 |
|--------|------|----------|
| Person 2 (Baseline Models) | Notebook 2: LR + DummyClassifier baselines | 2 days |
| Person 3 (Main Model) | Notebook 2: RF + default XGBoost | 2 days |
| Person 1 | Notebook 3: Evaluation + plots | 1–2 days |
| Person 2 | Report: Problem Statement + Dataset + Hypotheses | 1–2 days |
| Person 3 | Report: Approach + Rationale | 1–2 days |
| Person 4 | Report: Design of Future Experiments + HP Appendix | 1–2 days |

### Week 3 (finalize)

| 负责人 | 任务 | 预计时间 |
|--------|------|----------|
| Person 1 | Report: Partial Results (from notebook output) | 1 day |
| All | Report: Related Work, Plan of Activities | 1 day |
| All | Review + polish, ensure ≤ 5 pages | 1 day |

---

## Checklist

- [ ] Notebook 1: EDA + preprocessing 完成
- [ ] Notebook 2: 4 个 baseline models 训练完成
- [ ] Notebook 3: multi-split evaluation + plots 完成
- [ ] Overleaf ACM 模板建立
- [ ] Report Section 1: Problem Statement
- [ ] Report Section 2: Related Work (≥3 papers)
- [ ] Report Section 3: Approach
- [ ] Report Section 4: Rationale
- [ ] Report Section 5: Dataset(s)
- [ ] Report Section 6: Hypotheses (H1–H3/H4)
- [ ] Report Section 7: Experimental Design
- [ ] Report Section 8: Partial Results (tables + figures)
- [ ] Report Section 9: Design of Future Experiments
- [ ] Report Section 10: Hyperparameters Appendix (5 elements × 3 classifiers)
- [ ] Report Section 11: Plan of Activities
- [ ] References (BibTeX, correctly formatted)
- [ ] Final check: ≤ 5 pages, ACM format
