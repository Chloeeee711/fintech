from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'formal_project'
r=Path(json.loads((P/'independent_preprocessing/latest_run.json').read_text())['run_dir']);s=json.loads((r/'summary.json').read_text());v=json.loads((r/'independent_validation.json').read_text())
c=pd.read_parquet(r/'candidate_sample.parquet');a=pd.read_parquet(r/'filing_audit.parquet')
period_delta=int(c.period_end.ne(c.datadate).sum());precheck=int((~c.duplicate_company_period&~c.requires_text_review&c.d_capex_at.notna()).sum())
text=f'''# 独立预处理结果（2026-10-23）

## 结论

从原始Compustat、SEC索引和全文重新处理，现已形成{s['period_candidates']:,}份文本—财务报告期候选，其中{s['main_y_candidates']:,}份主Y可计算。报告期匹配具有可行性，不需因此更换选题。候选尚待语义复核、申报日和表格类型核验；没有将其称为正式训练样本。

最终运行：`{r.name}`。代码入口：`independent_preprocessing/run.py`；修正重匹配入口：`rematch.py`。不读取第一版输出，保留原始数据不改；仅复用经阅读的确定性自然文本清洗函数。

## 数量

| 项目 | 数量 |
|---|---:|
| 原始年度财务行 | 208,735 |
| 财务范围内行 | {s['financial_rows']:,} |
| 财务主Y非缺失 | {s['financial_main_y']:,} |
| 财务辅助Y非缺失 | {s['financial_aux_y']:,} |
| 2016—2023原文读取成功 | {s['read_ok']:,} |
| 期间匹配候选 | {s['period_candidates']:,} |
| 主Y可算候选 | {s['main_y_candidates']:,} |
| 辅助Y可算候选 | {s['aux_y_candidates']:,} |
| 候选实际期末不同于datadate | {period_delta:,} |
| 同公司同年度多份候选行 | {s['candidate_duplicate_company_period']:,} |
| 带文本复核标记的候选行 | {s['candidate_text_review']:,} |
| 主Y可算且无上述两类标记的候选 | {precheck:,} |

最后一项仍未通过报告期人工复核、修订身份与申报日检查，不能作为最终样本量；标记可能重叠，不能直接相加扣除。

## 方法与重要差异

1. 财务期末优先apdedate，缺失时回退datadate，完整保留两者。按GVKEY和fyear查准确前后年，以实际期末间隔330—400天检查可比性。
2. 财务范围筛选STD/INDL/C、USD、经营行业且非金融，同公司同财年歧义整组隔离。缺失指标保留，不因为辅助Y或某个X缺失删除主任务候选。89,744行与旧87,257行不能直接比较模型效果：本轮不提前丢弃空核心财务行，Y有效性在任务层筛选；实际期末规则亦不同。
3. 主Y严格为CAPX(t+1)/AT(t)−CAPX(t)/AT(t−1)；辅助Y为平均资产口径ROA变化。缺失Y及其方向标签保持缺失。财务F保存原始尺度，没有拟合缩尾、填补或标准化。
4. 文本重新清洗、保存全文与哈希；年度短语引出的日期列表保留原文证据，并排除附近预测措辞。按明确日期与公司信息匹配，日期不足不按申报年猜测。索引GVKEY存在但冲突时不擅自覆盖。
5. 原始10-K与修订版身份未确认，重复公司年度仅标记；不按申报编号任取一份。未知申报日、标签公开日仍缺失。
6. 2016—2019/2020/2021/2022—2023仅写入候选split。尚未进行最终方案要求的公开日期净化，不可据此开始真实拟合。

## 提取错误、修正与限制

- 第一轮宽日期窗口把年度后交易日混入期间列表，修正为仅接收日期/比较年份/连接词构成的列表。原轮与源码快照保留。
- 抽查发现一篇2023 Outlook仍用year ended描述预测，补充anticipate、日期后的expected等语境；回查该文应取2022-06-30。另一交易日案例回查为2022-12-31。
- 当前保守规则可能遗漏缩写月份、数字日期、52 weeks ended、较早迟报或靠近预测语句的历史日期，也可能误选第三方或引用报告中的年度。全部输出为候选，未声称全量语义准确率。
- 100份复核表已生成，两名复核者栏目留空。助手查看的10份证据用于发现规则问题，不替代两名组员独立标注，不直接放行样本。

## 校验与产物

6项规则测试通过；最终run独立校验{v['passed']}/{v['total']}项通过，包含两类Y各500条原始财务回算、实际期末优先、期间匹配、未知日期未编造、清洗文本行数守恒。修正重匹配逐篇检查47,604份原始文本SHA256，并验证复用财务/清洗分片哈希。PyArrow CPU探测权限提示没有影响读取和落盘。

- [最终摘要](../runs/{r.name}/summary.json)
- [独立校验](../runs/{r.name}/independent_validation.json)
- [年度候选数量](../runs/{r.name}/candidate_counts_by_year.csv)
- [100份期间复核表](../runs/{r.name}/period_review_100.csv)
- [运行记录](../runs/{r.name}/操作记录.md)
- [代码和使用说明](../independent_preprocessing/README.md)

同run包含`financial_panel`、`filing_audit`、`period_evidence`、`candidate_sample`及主/辅助任务候选Parquet。`reused_artifacts.json`指向首轮清洗全文和指纹；必须同时保留被引用的首轮目录。

## 尚未完成

正式方案中的实际申报日与下一财年标签公开日尚未确认，期间候选及表格类型仍需复核。下一步先核验期间和10-K/10-K/A身份、决定公司年度去重，再按真实日期完成时间划分。若最终无法取得申报日，应明确另行修改为年度比较设计；本轮未擅自修改最终方案。TF-IDF/预训练向量、Qwen、深度学习、RL均未拟合或训练。
'''
with (P/'docs/独立预处理结果.md').open('a') as f:f.write('\n\n'+text)
with (P/'docs/操作日志.md').open('a') as f:f.write(f'\n## 2026-10-23：完成独立预处理本轮结果与核验\n\n最终run：`{r.name}`。原文47,604份全部重读并校验；期间候选{s["period_candidates"]:,}、主Y可算{s["main_y_candidates"]:,}、辅助Y可算{s["aux_y_candidates"]:,}；其中{period_delta:,}份实际期末不同于datadate。6项规则测试通过，独立校验{v["passed"]}/{v["total"]}通过，实际命令`python formal_project/independent_preprocessing/validate.py`（临时venv）。报告生成命令`python formal_project/independent_preprocessing/report.py`（同venv）。详细输入输出、数量、修正规则、复核与未完成事项见[独立预处理结果](独立预处理结果.md)。未拟合真实特征或模型，候选不等同最终样本。\n')
readme=P/'README.md';old=readme.read_text();marker='## 当前阶段\n';pos=old.index(marker)+len(marker)
old=old[:pos]+'\n2026-10-23新增独立原始数据预处理，最新结果见[独立预处理结果](docs/独立预处理结果.md)，入口与限制见[independent_preprocessing](independent_preprocessing/README.md)。已生成报告期候选与F/Y，申报日、期间复核及正式时间筛选尚未完成。以下第一版说明保留作历史记录。\n'+old[pos:];readme.write_text(old)
print(r);print(text[:1500])
