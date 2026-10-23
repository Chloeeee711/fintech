# 正式项目：资本投资文本预测

正式代码、配置、输出和操作记录统一放在本目录；早期探索结果仍在`../analysis/`。

## 当前阶段

2026-10-23新增独立原始数据预处理，最新结果见[独立预处理结果](docs/独立预处理结果.md)，入口与限制见[independent_preprocessing](independent_preprocessing/README.md)。已生成报告期候选与F/Y，申报日、期间复核及正式时间筛选尚未完成。以下第一版说明保留作历史记录。

2026-10-23已在`p2_revision1/`建立原包的独立修正副本：财务Y已构造，文本匹配待补真实申报元数据，尚未拟合真实特征或训练模型。改动和Qwen获取方式见[第一版修正](docs/第一版修正.md)，使用命令见[工作副本README](p2_revision1/README.md)。原`runs/`继续保留阶段01结果，第一版新输出在`p2_revision1/runs/`。

查看[清洗结果](docs/阶段01_清洗结果.md)、[持续操作日志](docs/操作日志.md)、[清洗规则](docs/01_数据清洗规则.md)和[抽查记录](docs/阶段01_清洗抽查.md)。

先执行本地数据清洗。准确申报日、报告期与表格类型仍需补充，不能以申报年减1猜财年。本阶段不生成“已可训练”的样本。

## 目录

```text
config.json              年份、字段、筛选及清洗参数
requirements.txt         固定依赖版本
src/cleaning.py          标识、文本及TF-IDF预览清洗函数
src/preprocess.py        全量清洗入口，自动写逐步Markdown记录
src/validate_run.py      独立校验全部输出并生成清洗结果Markdown
tests/                   保留否定、金额、日期等关键规则的验证
docs/                    数据规则和持续操作日志
data/external/           后续导入的申报元数据
runs/<时间戳>/           每轮独立的数据、报告、复核清单及操作记录
latest_run.json          最近成功运行的位置与阶段状态
```

## 运行

当前已具备依赖的临时环境：

```bash
/private/tmp/fintech-quality-venv/bin/python -m unittest discover -s formal_project/tests -v
/private/tmp/fintech-quality-venv/bin/python formal_project/src/preprocess.py
/private/tmp/fintech-quality-venv/bin/python formal_project/src/validate_run.py
```

也可在自己的环境安装`requirements.txt`。入口可在任意工作目录运行；上述命令假设位于项目根目录。`--limit-texts 20`用于试跑，不能代替全量结果。

## 输出怎么用

- `annual_normalized.parquet`：选定原始财务字段、类型与标识统一、异常标记；覆盖全部原始年度行。
- `annual_scope.parquet`：统一格式、币种及行业范围，歧义财年隔离后的财务表。还没有构造Y或做日期匹配。
- `text_index_clean.parquet`：2016—2023文本索引，保留所有有效申报及待补元数据状态。
- `text_clean/*.parquet`：`accession_key、text_natural`，适合后续分别送入TF-IDF和预训练模型。
- `text_manifest.parquet`：逐文本的哈希、长度、清洗动作、质量及复核标记。
- `reports/`：财务缺失/分布/筛选表、文本分布、边界截断证据、30份转换预览及待补申报清单。
- `quarantine/`：被筛财务记录、异常索引和需要人工复核的文本；不是物理删除源数据。

## 后续记录约定

**后续每次操作必须追加到`docs/操作日志.md`**，写明目的、涉及代码与配置、输入输出、实际命令、记录数量变化、校验结果、限制与下一步。自动流水线另在每个run写`操作记录.md`；失败也记录，不能只记录成功。

在确认日期和时间划分后，才拟合缺失填补、缩尾、标准化、TF-IDF词表和降维；所有参数只能来自训练阶段。

CSV中的CIK、GVKEY、accession_key均应按字符串读取；Excel自动识别数字可能丢失前导零。优先使用Parquet保存标识类型。
