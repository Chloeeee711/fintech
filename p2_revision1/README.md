# 第一版修正：数据预处理

原包只读保存在`../vendor/p2_code_original/`。本目录复制原包源码，修正预处理；不继承原包结果、缓存模型或报告。Transformer和RL源码保留，尚未修复训练问题。

详细改动、运行结果与Qwen获取说明见[第一版修正](../docs/第一版修正.md)。

在项目根目录运行：

```bash
python -m pip install -r formal_project/p2_revision1/requirements-preprocess.txt
python formal_project/p2_revision1/src/run_all.py --preprocess-only
python -m unittest discover -s formal_project/tests -v
```

数据预处理每次生成独立`runs/时间戳/`。`latest_run.json`指向最新成功运行；失败目录也保留。

先将真实申报元数据放入`formal_project/data/external/filing_metadata.csv`，再重新运行。字段参见既有`filing_metadata_template.csv`，必须包含真实报告期、申报日、表格类型及来源。索引中没有这些信息，不能按文件年份猜财年；2023样本的标签日期还需2024及之后的相关申报记录。

匹配及文本复核通过后，才可运行：

```bash
python formal_project/p2_revision1/src/features_text.py
python formal_project/p2_revision1/src/capex_plan.py
```

两步均使用最新成功预处理run，不覆盖既有特征和人工标注表。没有匹配样本时会明确报错；全流程训练入口也暂时阻止运行。`status.json`描述财务/匹配阶段；特征阶段另写`feature_status.json`。

财务候选表与可用于训练的文本样本是两种产物，不能将候选行数写成最终样本量。`financial_panel.parquet`保留未缩尾、未填补的Y及F；`financial_features.parquet`另存训练集拟合后的F与缺失标志。主/辅助任务分开筛选，不以股价、收益率或另一任务Y是否缺失作为主任务筛选条件。

本轮没有生成收益率。RL需另补预测时点之后、可复核的价格和收益标签，再修策略梯度和样本外评价；本轮保留代码不表示这些问题已修复。
