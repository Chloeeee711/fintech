# 独立原始数据预处理

入口：`/private/tmp/fintech-quality-venv/bin/python formal_project/independent_preprocessing/run.py`。
规则检查：`/private/tmp/fintech-quality-venv/bin/python -m unittest discover -s formal_project/independent_preprocessing -p 'test_*.py' -v`。

本流程按`analysis/资本投资选题最终方案.md`构造原始F/Y和文本—财务期间候选，不读取第一版数据产物或模型。仅复用已检查的`src/cleaning.py`确定性文本清洗；自然文本保留数字、否定和日期。每次重新读取原始年度表、SEC索引与窗口内全文，独立run保存源数据/代码SHA256、逐篇文本哈希、运行配置、失败记录和输出哈希。

## 财务规则

- STD/INDL/C、USD、有效经营行业且排除6000—6999。行业优先历史sich，回退公司sic时标注。保留缺失财务指标，不因某个X或辅助Y缺失删除主任务候选。
- 同GVKEY同fyear多条记录整组隔离。历史与未来均按准确fyear连接，缺年不跨接。
- 实际期末优先apdedate，缺失时回退datadate并标注；相邻期末330—400天作为可比完整年度检查。两者均为数据库期间字段，不当作申报日。
- 主Y：CAPX(t+1)/AT(t)−CAPX(t)/AT(t−1)；辅助Y：NI(t+1)/平均资产(t,t+1)−NI(t)/平均资产(t−1,t)。资产须正，所需CAPX须非负。Y及方向标签缺失保留缺失。
- F：历史投资强度、ROA、收入增长、现金/资产、负债/资产、ln资产、两位SIC。原始数值不缩尾、不填补、不标准化。

## 文本期间候选与使用边界

- 搜索原文明确的year(s) ended/ending及完整英文月日年，保留上下文与规范化文本偏移。预测措辞单列；取非预测语境下最近明确年度日期作为候选。既不从申报年减1，也不把fyear当日历年。
- 日期证据范围为申报年及此前3年；更早的迟报本轮不会自动识别。正文没有完整英文日期、仅财年编号、月份缩写、数字日期或“52 weeks ended”等表达的记录可能漏掉。候选算法优先可解释性，目前不声称覆盖全部年度表述。
- 优先索引GVKEY；缺GVKEY时才按CIK查找；匹配后核对非空财务CIK。实际期末匹配唯一才保留候选。仍可能碰到收购对象、历史比较、引用其他报告等语义歧义，必须复核。
- 同公司同年度多份文本、相同清洗文本分别标记，不按编号擅自取最早。没有表格类型时不声称已区分10-K和10-K/A。
- `candidate_sample`及主/辅助任务表均是待复核候选。`proposed_split`仅描述最终方案候选年份，未通过实际申报日和标签可得性筛选。
- 申报日期、表格类型、下一期标签公开日未知时保持缺失；`strict_training_ready=False`。约100份复核表保留两名复核者的空白列，未伪造人工结果。

## 输出

`financial_panel.parquet`、`financial_excluded.parquet`、`filing_audit.parquet`、`period_evidence.parquet`、`texts/*.parquet`、`candidate_sample.parquet`、主/辅助任务候选、`period_review_100.csv`、`summary.json`和`validation.json`。

## 尚需完成

期间候选复核、真实申报日/表格类型核验、逐公司年度去重及标签公开日检查；随后才能冻结训练样本并拟合词表、填补、缩尾、标准化和降维。没有执行FinBERT/MPNet/Qwen、Transformer或RL训练。保留最终方案的真实申报日要求，不将年度近似偷偷当作同一种设计。

## 已保存清洗产物时重新核对期间

```bash
/private/tmp/fintech-quality-venv/bin/python formal_project/independent_preprocessing/rematch.py --source-run formal_project/runs/independent_20261023T054745_976329
/private/tmp/fintech-quality-venv/bin/python formal_project/independent_preprocessing/validate.py
```

`--source-run`须指向包含`texts/`及完整输出哈希的初始独立run。重新读取全部原文并核对哈希，财务与清洗文本复用前核对哈希，修正结果写入新run。最终run的`reused_artifacts.json`明确记录清洗文本目录与指纹，不能仅复制新run而丢弃被引用的原轮。

抽查发现交易日误入日期列表及“year ended”被用于预测语境两类问题，已分别加回归检查。预测语境目前检查年度短语前120字符及完整日期后100字符，会保守排除部分靠近预测措辞的历史描述；遗漏和误判仍须用人工期间复核表评估。

## 最新复核入口

`latest_review.json`指向最新范围修正与复核输出；`latest_run.json`保留早期全量候选指针。运行`review_candidates.py`进行证据/重复分组，`refine_review.py`修正历史行业与隔离旧报告引用，`validate_review.py`做独立回算。历史sich有值时不再被当前sic覆盖。所有输出仍需真实日期与期间核验。

按用户最新要求，不再创建Markdown；独立run/rematch/report入口的后续文字输出追加既有文档，旧run文档保留。
