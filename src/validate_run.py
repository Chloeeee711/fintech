"""Independently validate a preprocessing run and publish a readable report."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import hashlib
import json
import pandas as pd

PROJECT=Path(__file__).resolve().parents[1]
ROOT=PROJECT.parent

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def table(headers,rows):
    return "\n".join(["| "+" | ".join(headers)+" |","| "+" | ".join(["---"]*len(headers))+" |"]+["| "+" | ".join(map(str,row))+" |" for row in rows])

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-dir",default=None,help="Relative to formal_project, otherwise use latest successful run.")
    args=parser.parse_args()
    run=PROJECT/(args.run_dir or json.loads((PROJECT/"latest_run.json").read_text())["run_dir"])
    summary=json.loads((run/"summary.json").read_text())
    cfg=json.loads((run/"config.json").read_text())
    f=pd.read_parquet(run/"data/annual_scope.parquet")
    all_f=pd.read_parquet(run/"data/annual_normalized.parquet",columns=["source_row"])
    excluded=pd.read_parquet(run/"quarantine/financial_excluded.parquet",columns=["source_row"])
    idx=pd.read_parquet(run/"data/text_index_clean.parquet")
    manifest=pd.read_parquet(run/"data/text_manifest.parquet")
    checks={}
    checks["financial_count_conservation"]=len(f)+len(excluded)==len(all_f)
    checks["financial_no_ambiguous_company_fyear"]=not f.duplicated(["gvkey","fyear"]).any()
    checks["financial_source_rows_partition"]=len(set(f.source_row)&set(excluded.source_row))==0 and len(set(f.source_row)|set(excluded.source_row))==len(all_f)
    checks["financial_format_currency"]=all(f[c].eq(v).all() for c,v in cfg["format_filters"].items())
    checks["financial_industry_scope"]=f.industry_sic.between(100,8999).all() and not f.industry_sic.between(*cfg["exclude_sic_range"]).any()
    checks["text_index_unique_accessions"]=not idx.accession_key.duplicated().any()
    checks["text_manifest_unique_accessions"]=not manifest.accession_key.duplicated().any()
    checks["requested_text_scope_matches"]=len(manifest)==(cfg.get("runtime_limit_texts") or len(idx))
    rows=0
    seen=set()
    check_hashes=True
    check_lengths=True
    expected=manifest.loc[manifest.read_status.eq("ok")].set_index("accession_key")
    for path in sorted((run/"data/text_clean").glob("*.parquet")):
        chunk=pd.read_parquet(path)
        for row in chunk.itertuples(index=False):
            if row.accession_key in seen:
                raise AssertionError("Duplicate text record across output parts")
            seen.add(row.accession_key)
            e=expected.loc[row.accession_key]
            check_hashes &= hashlib.sha256(row.text_natural.encode()).hexdigest()==e.clean_sha256
            check_lengths &= len(row.text_natural)==e.clean_characters
            rows+=1
    checks["every_clean_record_matches_manifest"]=seen==set(expected.index)
    checks["all_clean_hashes_match"]=bool(check_hashes)
    checks["all_clean_lengths_match"]=bool(check_lengths)
    checks["clean_count_matches_read_ok"]=rows==summary["read_ok"]
    sources=json.loads((run/"source_manifest.json").read_text())
    checks["raw_financial_and_index_sha256_unchanged"]=all(digest(ROOT/p)==v["sha256"] for p,v in sources.items())
    sample=manifest.loc[manifest.read_status.eq("ok")].sample(n=min(30,len(expected)),random_state=20261022)
    checks["sample_30_original_text_sha256"]=all(digest(ROOT/"SEC 10-K ITEM 7"/str(row.filing_year)/row.filename)==row.raw_sha256 for row in sample.itertuples())
    checks["no_fitted_preprocessing_claimed"]=not any(summary[k] for k in ["training_ready","fitted_scaler","fitted_imputer","fitted_vocabulary"])
    if int(idx.metadata_ready.sum())==0:
        checks["no_fiscal_year_guessing_claimed"]=not summary["financial_period_text_matching_complete"] and idx.period_match_status.eq("pending_metadata").all()
    result={"run_dir":str(run.relative_to(PROJECT)),"validation_time":datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),"validator_sha256":digest(Path(__file__)),"checks":{k:bool(v) for k,v in checks.items()},"clean_records_checked":rows,"original_text_files_rehashed":len(sample)}
    (run/"validation.json").write_text(json.dumps(result,indent=2,ensure_ascii=False))
    if not all(checks.values()):
        raise AssertionError(result)
    flow=pd.read_csv(run/"reports/financial_sample_flow.csv")
    missing=pd.read_csv(run/"reports/financial_missing.csv")
    text_flags={}
    for value in manifest.quality_flags.fillna(""):
        for flag in value.split("|"):
            if flag:
                text_flags[flag]=text_flags.get(flag,0)+1
    text_flags["same_clean_text_multiple_filings"]=int(manifest.same_clean_text_multiple_filings.sum())
    report="# 阶段01数据清洗结果\n\n"
    report+=f"运行：`{run.relative_to(PROJECT)}`。完整逐步日志见本轮`操作记录.md`，所有筛选记录和复核证据均保留。\n\n"
    report+="## 已完成\n\n"
    report+=table(["内容","实际数量"],[
        ["年度原始财务行",len(all_f)],["统一范围后财务行",len(f)],["范围内公司",f.gvkey.nunique()],
        ["窗口内索引",len(idx)],["本轮文本处理",len(manifest)],["文本读取成功",summary["read_ok"]],
        ["文本待复核",summary["text_review_rows"]],["申报元数据待补",int((~idx.metadata_ready).sum())]])
    report+="\n\n## 财务筛选过程\n\n"+table(["规则","处理前","排除","处理后"],flow.values.tolist())
    report+="\n\n排除顺序会影响各原因的计数。负CAPX和非正资产在范围表内保留原值和有效性标记，后续按不同Y的规则筛选，不在本阶段平均填补。\n\n"
    report+="## 范围内关键字段缺失\n\n"+table(["字段","缺失行","缺失率"],[[r.field,r.missing_n,f"{r.missing_pct:.2f}%"] for r in missing.itertuples()])
    report+="\n\n## 文本规则标记\n\n"+table(["标记","文档数"],list(text_flags.items()))
    report+="\n\n标记可能重叠，不可相加当作剔除数。短文、章节边界、重复内容仅要求复核，不自动判无效。尚未按未来建模Y或文本主题筛样本。\n\n"
    report+="## 校验\n\n"+table(["项目","结果"],[[k,"通过" if v else "失败"] for k,v in checks.items()])
    report+=f"\n\n已逐条核验{rows:,}条清洗输出的长度与哈希，另外重算30份原文及财务/索引源文件SHA256。保留否定、金额、日期等8项单元测试结果另见操作日志。\n\n"
    report+="## 本轮未执行与下一步\n\n- 准确申报日、报告期、表格类型仍待补；未猜财年，未生成可训练的文本—财务匹配表。\n- 未构造最终Y、拟合填补/缩尾/标准化、词表/IDF或降维，未运行FinBERT、MPNet、Qwen。\n- 补元数据后核对原始10-K与修订版，按报告期匹配；抽查文本截断与重复；再构造Y及时间划分。\n"
    (run/"结果报告.md").write_text(report)
    (PROJECT/"docs/阶段01_清洗结果.md").write_text(report)
    log=PROJECT/"docs/操作日志.md"
    with log.open("a") as out:
        out.write(f"\n## {result['validation_time']}：独立输出校验\n\n")
        out.write(f"- 执行：`/private/tmp/fintech-quality-venv/bin/python formal_project/src/validate_run.py --run-dir {run.relative_to(PROJECT)}`。\n")
        out.write(f"- {len(checks)}项完整性、范围、日期状态、输出哈希及源文件检查全部通过；逐条覆盖{rows}条清洗文本。\n")
        out.write("- 结果见[阶段01清洗结果](阶段01_清洗结果.md)，机器校验见对应run的validation.json。\n- 本次新增独立校验脚本validate_run.py，未更改源数据或清洗规则。\n")
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
