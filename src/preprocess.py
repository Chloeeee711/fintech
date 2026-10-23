"""Formal stage 01: source-based cleaning, audit outputs and Markdown journal.

Run from any directory with the environment in requirements.txt:
    python formal_project/src/preprocess.py
No external network calls. All outputs live in an immutable, timestamped run.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import traceback

import numpy as np
import pandas as pd

from cleaning import clean_text, normalize_accession, normalize_identifier, tokenize_tfidf

PROJECT=Path(__file__).resolve().parents[1]
ROOT=PROJECT.parent

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def write_json(path, obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding="utf-8")

class Run:
    def __init__(self, config):
        self.started=datetime.now(ZoneInfo("America/Los_Angeles"))
        self.path=PROJECT/"runs"/self.started.strftime("%Y%m%dT%H%M%S_%f")
        self.path.mkdir(parents=True)
        for name in ["data", "reports", "quarantine", "data/text_clean"]:
            (self.path/name).mkdir(parents=True,exist_ok=True)
        self.log=self.path/"操作记录.md"
        self.log.write_text("# 数据清洗运行记录\n\n状态：运行中。每一步完成后立即追加记录。\n",encoding="utf-8")
        self.events=[]
        write_json(self.path/"config.json",config)
        write_json(self.path/"environment.json",{
            "python":sys.version,"platform":platform.platform(),
            "packages":{p:importlib.metadata.version(p) for p in ["pandas","numpy","pyarrow"]},
            "scripts":{p.name:digest(p) for p in sorted((PROJECT/"src").glob("*.py"))},
        })
        self.record("运行初始化",{"run_path":str(self.path),"started":self.started.isoformat()},
                    "从原始文件读取；每次运行新建目录，避免覆盖上一轮结果。")

    def record(self, title, values, note=""):
        timestamp=datetime.now(ZoneInfo("America/Los_Angeles")).isoformat()
        event={"time":timestamp,"step":title,"values":values,"note":note}
        self.events.append(event)
        with self.log.open("a",encoding="utf-8") as f:
            f.write(f"\n## {len(self.events)}. {title}\n\n时间：{timestamp}\n\n{note}\n\n")
            for key,value in values.items():
                f.write(f"- {key}：{value}\n")
        write_json(self.path/"events.json",self.events)
        print(title,json.dumps(values,ensure_ascii=False,default=str),flush=True)

def load_financial(run,cfg):
    source=ROOT/"Compustat/g7watsrhufsgtynw.dta"
    index=ROOT/"SEC 10-K ITEM 7/index.csv"
    sources={}
    for path in [source,index]:
        stat=path.stat()
        sources[str(path.relative_to(ROOT))]={"bytes":stat.st_size,"mtime_ns":stat.st_mtime_ns,"sha256":digest(path)}
    write_json(run.path/"source_manifest.json",sources)
    run.record("源文件指纹",{p:v["bytes"] for p,v in sources.items()},"记录SHA256、大小和修改时间；本阶段不使用早期探索性缓存。")
    df=pd.read_stata(source,columns=cfg["financial_columns"],convert_categoricals=False)
    df.insert(0,"source_row",np.arange(len(df),dtype=np.int64))
    raw_n=len(df)
    invalid={}
    for c,width in [("gvkey",6),("cik",None)]:
        raw=df[c].copy()
        df[c]=pd.Series([normalize_identifier(v,width) for v in raw],dtype="string")
        invalid[c]=int((raw.notna() & raw.astype(str).str.strip().ne("") & df[c].isna()).sum())
    for c in ["conm","tic","sic","sich","curcd","datafmt","indfmt","consol","fic","costat"]:
        df[c]=df[c].astype("string").str.strip().replace("",pd.NA)
    for c in ["datadate","fdate","pdate"]:
        df[c]=pd.to_datetime(df[c],errors="coerce")
    numeric=["fyear","fyr","at","ni","ib","sale","revt","capx","che","lt","oancf","emp","ppent","aqc"]
    invalid_numeric={}
    for c in numeric:
        v=pd.to_numeric(df[c],errors="coerce")
        invalid_numeric[c]=int((v.notna() & ~np.isfinite(v)).sum())
        df[c]=v.where(np.isfinite(v))
    df["fyear"]=df.fyear.where(df.fyear.mod(1).eq(0)).astype("Int64")
    historical=pd.to_numeric(df.sich,errors="coerce")
    current=pd.to_numeric(df.sic,errors="coerce")
    h_valid=(historical.between(100,8999)&historical.mod(1).eq(0)).fillna(False)
    df["industry_sic"]=historical.where(h_valid,current).astype("Int64")
    df["industry_source"]=np.where(h_valid,"sich_historical","sic_company_fallback")
    df["flag_negative_capx"]=df.capx.lt(0)
    df["flag_nonpositive_assets"]=df["at"].le(0)
    df["flag_negative_sales"]=df.sale.lt(0)
    df["flag_empty_core_financial"]=df[["at","revt","ib"]].isna().all(axis=1)
    df.to_parquet(run.path/"data/annual_normalized.parquet",index=False,compression="zstd")
    run.record("财务标识与类型清洗",{"rows":raw_n,"invalid_identifiers":invalid,"nonfinite_numeric":invalid_numeric},
               "GVKEY统一6位，CIK去前导零；保留source_row。非有限数转缺失；金额不换单位、不填补、不缩尾。行业优先sich，缺失时回退公司sic并标记来源。")

    rejected=[]
    funnel=[]
    def retain(mask,reason):
        nonlocal df
        mask=mask.fillna(False)
        bad=df.loc[~mask].copy()
        if len(bad):
            bad["exclusion_reason"]=reason
            rejected.append(bad)
        before=len(df)
        df=df.loc[mask].copy()
        funnel.append({"step":reason,"before":before,"removed":before-len(df),"after":len(df)})
    for c,v in cfg["format_filters"].items():
        retain(df[c].eq(v),f"require_{c}_{v}")
    retain(df.gvkey.notna()&df.datadate.notna()&df.fyear.notna(),"require_valid_company_date_fyear")
    retain(~df.flag_empty_core_financial,"exclude_empty_core_financial")
    retain(df.industry_sic.between(100,8999),"exclude_unknown_or_nonoperating_sic")
    lo,hi=cfg["exclude_sic_range"]
    retain(~df.industry_sic.between(lo,hi),"exclude_financial_sic")
    # Ambiguous fiscal years are quarantined as a group, never arbitrarily deduped.
    duplicate=df.duplicated(["gvkey","fyear"],keep=False)
    retain(~duplicate,"quarantine_ambiguous_company_fiscal_year")
    # A row may still support one outcome and not the other: preserve it with flags.
    df["usable_capx_component"]=df.capx.notna()&df.capx.ge(0)
    df["usable_asset_component"]=df["at"].notna()&df["at"].gt(0)
    df["usable_ni_component"]=df.ni.notna()
    df["fiscal_key"]=df.gvkey+"_"+df.fyear.astype("string")
    df=df.sort_values(["gvkey","datadate"])
    df.to_parquet(run.path/"data/annual_scope.parquet",index=False,compression="zstd")
    pd.DataFrame(funnel).to_csv(run.path/"reports/financial_sample_flow.csv",index=False)
    excluded=pd.concat(rejected,ignore_index=True) if rejected else pd.DataFrame()
    excluded.to_parquet(run.path/"quarantine/financial_excluded.parquet",index=False,compression="zstd")
    fields=["at","ni","ib","sale","revt","capx","che","lt","oancf","emp","ppent","aqc"]
    miss=pd.DataFrame({"field":fields,"missing_n":[int(df[c].isna().sum()) for c in fields],"missing_pct":[100*df[c].isna().mean() for c in fields]})
    miss.to_csv(run.path/"reports/financial_missing.csv",index=False)
    stats=df[fields].describe(percentiles=[.01,.05,.5,.95,.99]).T
    stats["zero_n"]=df[fields].eq(0).sum()
    stats["negative_n"]=df[fields].lt(0).sum()
    stats.to_csv(run.path/"reports/financial_distribution.csv")
    df.groupby("fyear").agg(rows=("gvkey","size"),firms=("gvkey","nunique")).to_csv(run.path/"reports/financial_by_year.csv")
    assert len(df)+len(excluded)==raw_n
    assert not df.duplicated(["gvkey","fyear"]).any()
    run.record("财务样本范围与重复处理",{"raw_rows":raw_n,"scope_rows":len(df),"excluded_rows":len(excluded),"firms":int(df.gvkey.nunique()),"negative_capx_retained_flagged":int(df.flag_negative_capx.sum()),"nonpositive_assets_retained_flagged":int(df.flag_nonpositive_assets.sum())},
               "逐项筛选数量见financial_sample_flow.csv；被排除行全部保存在quarantine。仅保留相关财务范围，不等于Y已可构造；负CAPX、非正资产保留原值并标记。没有按公司是否仍存续筛选。")
    return df,sources

def load_index(run,cfg,financial):
    idx=pd.read_csv(ROOT/"SEC 10-K ITEM 7/index.csv",dtype="string")
    idx.insert(0,"index_source_row",np.arange(len(idx),dtype=np.int64))
    all_n=len(idx)
    idx["filing_year"]=pd.to_numeric(idx.year,errors="coerce").astype("Int64")
    idx=idx.loc[idx.filing_year.between(cfg["filing_year_start"],cfg["filing_year_end"])].copy()
    idx["gvkey"]=pd.Series([normalize_identifier(v,6) for v in idx.gvkey],index=idx.index,dtype="string")
    idx["cik"]=pd.Series([normalize_identifier(v) for v in idx.cik],index=idx.index,dtype="string")
    idx["accession_key"]=pd.Series([normalize_accession(v) for v in idx.accession_number_nodash],index=idx.index,dtype="string")
    idx["index_size_bytes"]=pd.to_numeric(idx.filesize_bytes,errors="coerce")
    idx["company_in_financial_scope"]=idx.gvkey.isin(set(financial.gvkey))
    idx["multiple_filings_company_calendar_year"]=idx.gvkey.notna()&idx.duplicated(["gvkey","filing_year"],keep=False)
    bad=idx.accession_key.isna()|idx.cik.isna()|idx.accession_key.duplicated(keep=False)
    idx.loc[bad].to_csv(run.path/"quarantine/index_invalid_or_duplicate_accession.csv",index=False)
    idx=idx.loc[~bad].copy()
    # Check crosswalk consistency descriptively; CIK can change over company history.
    crosswalk=financial.dropna(subset=["gvkey","cik"])
    pairs=set(zip(crosswalk.gvkey,crosswalk.cik))
    idx["cik_gvkey_pair_seen_in_financial_scope"]=[(g,c) in pairs for g,c in zip(idx.gvkey,idx.cik)]
    metadata=PROJECT/cfg["metadata_path"]
    if metadata.exists():
        meta=pd.read_csv(metadata,dtype="string")
        required=["accession_key","filing_date","report_period_end","form_type","metadata_source"]
        if not set(required)<=set(meta):
            raise ValueError(f"Metadata requires columns: {required}")
        if meta.accession_key.duplicated().any():
            raise ValueError("Metadata accession_key must be unique")
        idx=idx.merge(meta[required],on="accession_key",how="left",validate="one_to_one")
    else:
        for c in ["filing_date","report_period_end","form_type","metadata_source"]:
            idx[c]=pd.NA
    for c in ["filing_date","report_period_end"]:
        idx[c]=pd.to_datetime(idx[c],errors="coerce")
    complete=idx[["filing_date","report_period_end","form_type","metadata_source"]].notna().all(axis=1)
    date_order=idx.report_period_end.le(idx.filing_date)
    year_consistent=idx.filing_date.dt.year.eq(idx.filing_year).fillna(False)
    idx["metadata_ready"]=complete & date_order & year_consistent & idx.form_type.isin(["10-K","10-K/A"])
    idx["period_match_status"]=np.where(idx.metadata_ready,"metadata_available_match_pending","pending_metadata")
    pending=idx.loc[~idx.metadata_ready,["accession_key","accession_number","cik","gvkey","filing_year","filename"]].copy()
    pending.to_csv(run.path/"reports/filing_metadata_required.csv",index=False)
    # Ambiguous calendar years retained: amendment status is not known without metadata.
    idx.to_parquet(run.path/"data/text_index_clean.parquet",index=False,compression="zstd")
    run.record("文本索引与元数据准备",{"all_index_rows":all_n,"window_rows_after_index_checks":len(idx),"index_quarantined":int(bad.sum()),"missing_gvkey":int(idx.gvkey.isna().sum()),"company_in_financial_scope":int(idx.company_in_financial_scope.sum()),"multiple_company_calendar_year_rows":int(idx.multiple_filings_company_calendar_year.sum()),"metadata_ready":int(idx.metadata_ready.sum()),"metadata_pending":len(pending)},
               "公司层面覆盖不等于财年匹配成功。全部保留有效申报，不按申报编号任取一份；正式原始/修订去重等待元数据。待补清单见filing_metadata_required.csv；没有将申报年减1猜财年。")
    return idx

def process_texts(run,cfg,idx,limit=None):
    selected=idx.head(limit) if limit else idx
    batch=[]
    manifest=[]
    part=0
    for row in selected.itertuples(index=False):
        filename=str(row.filename)
        base={"accession_key":row.accession_key,"gvkey":row.gvkey,"cik":row.cik,"filing_year":int(row.filing_year),"filename":filename,"company_in_financial_scope":row.company_in_financial_scope}
        # Filenames are treated as data and must stay in the configured source root.
        path=ROOT/"SEC 10-K ITEM 7"/str(row.filing_year)/filename
        if Path(filename).name!=filename or not filename.endswith(".txt"):
            manifest.append({**base,"read_status":"invalid_filename","quality_flags":"invalid_filename"})
            continue
        try:
            raw_bytes=path.read_bytes()
            try:
                raw=raw_bytes.decode("utf-8")
                encoding="utf8"
            except UnicodeDecodeError:
                raw=raw_bytes.decode("utf-8",errors="replace")
                encoding="utf8_replacement"
            cleaned,info=clean_text(raw,cfg["min_text_characters_review"])
            record={**base,**info,"read_status":"ok","encoding":encoding,"source_bytes":len(raw_bytes),
                    "index_size_matches":bool(pd.notna(row.index_size_bytes) and len(raw_bytes)==row.index_size_bytes),
                    "raw_sha256":hashlib.sha256(raw_bytes).hexdigest(),"clean_sha256":hashlib.sha256(cleaned.encode()).hexdigest()}
            manifest.append(record)
            batch.append({"accession_key":row.accession_key,"text_natural":cleaned})
            if len(batch)>=cfg["text_batch_size"]:
                pd.DataFrame(batch).to_parquet(run.path/f"data/text_clean/part-{part:05d}.parquet",index=False,compression="zstd")
                batch=[]
                part+=1
            if len(manifest)%4000==0:
                run.record("文本清洗进度",{"processed":len(manifest),"total":len(selected)},"逐文件读取、记录原文及清洗后SHA256；清洗文本分片压缩落盘。")
        except (OSError,ValueError) as exc:
            manifest.append({**base,"read_status":"read_error","quality_flags":"read_error","error":str(exc)})
    if batch:
        pd.DataFrame(batch).to_parquet(run.path/f"data/text_clean/part-{part:05d}.parquet",index=False,compression="zstd")
        part+=1
    result=pd.DataFrame(manifest)
    ok=result.read_status.eq("ok")
    result["same_clean_text_multiple_filings"]=False
    result.loc[ok,"same_clean_text_multiple_filings"]=result.loc[ok,"clean_sha256"].duplicated(keep=False)
    result["requires_review"]=~ok|result.quality_flags.fillna("").ne("")|result.same_clean_text_multiple_filings
    result.to_parquet(run.path/"data/text_manifest.parquet",index=False,compression="zstd")
    result.loc[result.requires_review].to_csv(run.path/"quarantine/text_requires_review.csv",index=False)
    result.loc[ok&result.boundary_trim_characters.gt(0),["accession_key","filename","boundary_trim_characters","boundary_excerpt"]].to_csv(run.path/"reports/boundary_trim_evidence.csv",index=False)
    for key in ["filing_year","company_in_financial_scope"]:
        result.groupby(key).agg(rows=("accession_key","size"),review=("requires_review","sum")).to_csv(run.path/f"reports/text_by_{key}.csv")
    result.loc[ok,["raw_characters","clean_characters","layout_block_count","removed_noise_lines","removed_duplicate_long_blocks","boundary_trim_characters"]].describe(percentiles=[.01,.5,.99]).to_csv(run.path/"reports/text_cleaning_distribution.csv")
    # Samples are for inspecting transformations, not selected by future outcomes.
    sample_ids=result.loc[ok].sample(n=min(30,int(ok.sum())),random_state=20261022).accession_key
    examples=[]
    wanted=set(sample_ids)
    for partpath in sorted((run.path/"data/text_clean").glob("*.parquet")):
        frame=pd.read_parquet(partpath)
        for row in frame.loc[frame.accession_key.isin(wanted)].itertuples():
            examples.append({"accession_key":row.accession_key,"natural_excerpt":row.text_natural[:1200],"tfidf_tokens_preview":tokenize_tfidf(row.text_natural[:1200])})
    write_json(run.path/"reports/text_transform_examples.json",examples)
    assert len(result)==len(selected)
    assert not result.accession_key.duplicated().any()
    run.record("全文自然语言清洗完成",{"requested":len(selected),"read_ok":int(ok.sum()),"read_failed":int((~ok).sum()),"parquet_parts":part,"requires_review":int(result.requires_review.sum()),"boundary_trim_documents":int((ok&result.boundary_trim_characters.gt(0)).sum()),"duplicate_clean_text_documents":int(result.same_clean_text_multiple_filings.sum())},
               "保留自然语言、否定词、金额、年份及裸数字行；仅去已识别页眉、标签与重复长块。7A/8明确标题在2500字符后才截断，早期标题和大比例截断单列复核。同内容不同申报仅标记，不据此自动合并。TF-IDF分词仅输出30个预览，完整分词和模型词表后续在各路线处理。")
    return result

def finish(run,cfg,sources,financial,idx,text):
    for rel,info in sources.items():
        stat=(ROOT/rel).stat()
        if stat.st_size!=info["bytes"] or stat.st_mtime_ns!=info["mtime_ns"]:
            raise RuntimeError(f"Source changed during run: {rel}")
    summary={"text_window_processing_complete":len(text)==len(idx),"financial_scope_rows":len(financial),"financial_firms":int(financial.gvkey.nunique()),
             "window_index_rows":len(idx),"processed_text_rows":len(text),
             "read_ok":int(text.read_status.eq("ok").sum()),"text_review_rows":int(text.requires_review.sum()),
             "metadata_ready_rows":int(idx.metadata_ready.sum()),"financial_period_text_matching_complete":False,
             "training_ready":False,"fitted_scaler":False,"fitted_imputer":False,"fitted_vocabulary":False,
             "next_step":"补齐并核验申报日期/报告期/表格类型，完成逐财年匹配与原始10-K去重；抽样复核文本清洗，再构造Y及时间划分。"}
    write_json(run.path/"summary.json",summary)
    run.record("校验与阶段状态",summary,"本轮完成本地清洗，不宣称已经生成可训练样本；源文件大小与修改时间前后相同。未缩尾、填补、标准化、拟合词表或运行模型。")
    with run.log.open("a",encoding="utf-8") as f:
        f.write("\n**运行结束：本地清洗完成；精确日期与财年匹配仍待补元数据。**\n")
    write_json(PROJECT/"latest_run.json",{"run_dir":str(run.path.relative_to(PROJECT)),"summary":summary})
    log=PROJECT/"docs/操作日志.md"
    if not log.exists():
        log.write_text("# 正式项目操作日志\n\n所有后续操作在此追加：目的、代码/配置、输入输出、样本变化、校验及待办。\n",encoding="utf-8")
    with log.open("a",encoding="utf-8") as f:
        rel=run.path.relative_to(PROJECT)
        f.write(f"\n## {run.started.isoformat()}：阶段01本地清洗\n\n")
        f.write(f"详细逐步记录：[本次运行](../{rel}/操作记录.md)。\n\n")
        for key,value in summary.items():
            f.write(f"- {key}：{value}\n")
        f.write("\n未完成精确日期匹配；没有将初步结果当作正式训练集。\n")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--limit-texts",type=int,default=None,help="Smoke run only; default processes every filing in configured years.")
    args=parser.parse_args()
    cfg=json.loads((PROJECT/"config.json").read_text())
    cfg["runtime_limit_texts"]=args.limit_texts
    run=Run(cfg)
    try:
        financial,sources=load_financial(run,cfg)
        idx=load_index(run,cfg,financial)
        text=process_texts(run,cfg,idx,args.limit_texts)
        finish(run,cfg,sources,financial,idx,text)
    except Exception:
        message=traceback.format_exc()
        run.record("运行失败",{"traceback":message},"保留已生成输出与失败原因；不更新latest_run为成功结果。")
        raise

if __name__=="__main__":
    main()
