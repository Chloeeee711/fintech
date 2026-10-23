"""Correct period-list parsing, re-read every raw text and verify source bytes.
Reuse independently produced financial/clean-text artifacts after SHA256 checks.
"""
from run import ROOT,PROJECT,extract,sha
from pathlib import Path
import pandas as pd
import numpy as np
import json,datetime,traceback,shutil
from zoneinfo import ZoneInfo
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--source-run',required=True);args=parser.parse_args()
old=Path(args.source_run).resolve()
r=PROJECT/'runs'/('independent_corrected_'+datetime.datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%dT%H%M%S_%f'));r.mkdir()
def save(name,obj): (r/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str))
def log(msg):
 print(msg,flush=True)
 with (PROJECT/'docs/操作日志.md').open('a') as f:f.write('\n['+r.name+'] '+msg+'\n\n')
log('# 修正日期列表后的独立预处理\n\n命令：/private/tmp/fintech-quality-venv/bin/python formal_project/independent_preprocessing/rematch.py')
try:
 manifest=json.loads((old/'output_sha256.json').read_text())
 reused=['financial_panel.parquet','financial_excluded.parquet']+[str(p.relative_to(old)) for p in (old/'texts').glob('*.parquet')]
 for name in reused:
  assert sha(old/name)==manifest[name],name
 # Preserve prior version and immutable source/output references; clean text remains in prior run.
 for name in ['financial_panel.parquet','financial_excluded.parquet']:shutil.copy2(old/name,r/name)
 save('reused_artifacts.json',{'source_run':str(old),'text_directory':str(old/'texts'),'verified_sha256':{k:manifest[k] for k in reused}})
 save('config.json',{'filing_years':[2016,2023],'period_rule':'annual phrase followed solely by date/comparison list','review_required':True,'fitted_operations':False})
 save('fingerprints.json',{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),PROJECT/'independent_preprocessing/run.py',PROJECT/'independent_preprocessing/test_rules.py',ROOT/'analysis/资本投资选题最终方案.md']})
 a=pd.read_parquet(old/'filing_audit.parquet');fin=pd.read_parquet(r/'financial_panel.parquet');ev=[]
 bygv={k:v for k,v in fin.groupby('gvkey')};bycik={k:v for k,v in fin.dropna(subset=['cik']).groupby('cik')}
 for col in ['matched_gvkey','matched_fyear','matched_period_source','text_period_candidate']:a[col]=None
 for n,(j,row) in enumerate(a.iterrows()):
  if row.read_status!='ok':continue
  p=ROOT/'SEC 10-K ITEM 7'/str(row.filing_year)/row.filename;b=p.read_bytes();assert __import__('hashlib').sha256(b).hexdigest()==row.raw_sha256
  evidence=extract(b.decode('utf-8'),int(row.filing_year));ev.extend({'accession_key':row.accession_key,**e} for e in evidence)
  dates=sorted({e['date'] for e in evidence if not e['prospective_wording']})
  date=pd.Timestamp(dates[-1]) if dates else None;a.at[j,'text_period_candidate']=date
  g=row.index_gvkey;c=row.cik;scope=bygv.get(g) if pd.notna(g) else bycik.get(c)
  status='no_annual_date_evidence' if date is None else 'company_outside_scope'
  if date is not None and scope is not None:
   matches=scope[scope.period_end.eq(date)];status='period_not_in_financial_scope'
   if len(matches)==1:
    f=matches.iloc[0]
    if pd.notna(f.cik) and f.cik!=c:status='company_identifier_conflict'
    else:
     a.at[j,'matched_gvkey']=f.gvkey;a.at[j,'matched_fyear']=int(f.fyear);a.at[j,'matched_period_source']=f.period_source;status='period_candidate_requires_review'
   elif len(matches)>1:status='ambiguous_company_period'
  a.at[j,'match_status']=status
  if (n+1)%8000==0:log(f'已重新核对原始字节及期间证据{n+1}/{len(a)}。')
 a['text_period_candidate']=pd.to_datetime(a.text_period_candidate)
 a['matched_fyear']=pd.to_numeric(a.matched_fyear).astype('Int64')
 a['duplicate_company_period']=a.matched_gvkey.notna()&a.duplicated(['matched_gvkey','matched_fyear'],keep=False)
 a.to_parquet(r/'filing_audit.parquet',index=False);e=pd.DataFrame(ev);e.to_parquet(r/'period_evidence.parquet',index=False)
 c=a[a.match_status.eq('period_candidate_requires_review')].merge(fin,left_on=['matched_gvkey','matched_fyear'],right_on=['gvkey','fyear'],how='left',validate='many_to_one',suffixes=('','_financial'))
 c['proposed_split']=np.select([c.filing_year.le(2019),c.filing_year.eq(2020),c.filing_year.eq(2021)],['train','validation','buffer'],default='test')
 c['review_status']='unreviewed';c['label_public_date']=pd.NaT
 c.to_parquet(r/'candidate_sample.parquet',index=False)
 c[c.d_capex_at.notna()].to_parquet(r/'main_task_candidates.parquet',index=False);c[c.droa.notna()].to_parquet(r/'aux_task_candidates.parquet',index=False)
 sample=c.sample(min(100,len(c)),random_state=20261023)[['accession_key','filename','filing_year','matched_gvkey','matched_fyear','text_period_candidate','matched_period_source','duplicate_company_period','requires_text_review']]
 sample=sample.join(e.groupby('accession_key').context.agg(lambda x:'\n---\n'.join(dict.fromkeys(x))),on='accession_key')
 for col in ['reviewer_1_period','reviewer_2_period','form_type_verified','filing_date_verified','review_notes']:sample[col]=''
 sample.to_csv(r/'period_review_100.csv',index=False)
 c.groupby('filing_year').agg(candidates=('accession_key','size'),main_y=('d_capex_at','count'),aux_y=('droa','count'),text_review=('requires_text_review','sum'),duplicate_period=('duplicate_company_period','sum')).to_csv(r/'candidate_counts_by_year.csv')
 olda=pd.read_parquet(old/'filing_audit.parquet');change=a.text_period_candidate.ne(olda.text_period_candidate)&~(a.text_period_candidate.isna()&olda.text_period_candidate.isna())
 a.loc[change,['accession_key','filename','text_period_candidate','match_status']].assign(previous_period=olda.loc[change,'text_period_candidate'].values).to_csv(r/'period_rule_changes.csv',index=False)
 summary={'financial_rows':len(fin),'financial_main_y':int(fin.d_capex_at.notna().sum()),'financial_aux_y':int(fin.droa.notna().sum()),'text_rows':len(a),'read_ok':int(a.read_status.eq('ok').sum()),'match_status':a.match_status.value_counts().to_dict(),'period_candidates':len(c),'main_y_candidates':int(c.d_capex_at.notna().sum()),'aux_y_candidates':int(c.droa.notna().sum()),'candidate_actual_period_matches':int(c.matched_period_source.eq('apdedate').sum()),'candidate_duplicate_company_period':int(c.duplicate_company_period.sum()),'candidate_text_review':int(c.requires_text_review.sum()),'rule_changed_periods':int(change.sum()),'strict_training_ready':False,'fitted_operations':False}
 save('summary.json',summary);log(json.dumps(summary,ensure_ascii=False));save('status.json',{'status':'completed_candidate_preprocessing','training_ready':False})
 save('output_sha256.json',{str(p.relative_to(r)):sha(p) for p in r.rglob('*') if p.is_file()})
 (PROJECT/'independent_preprocessing/latest_run.json').write_text(json.dumps({'run_dir':str(r)},indent=2))
except Exception:
 log(traceback.format_exc());save('status.json',{'status':'failed'});raise
finally:
 with (PROJECT/'docs/操作日志.md').open('a') as f:f.write(f'\n## 日期列表规则修正与重新预处理\n\n本轮命令`/private/tmp/fintech-quality-venv/bin/python formal_project/independent_preprocessing/rematch.py`；原轮保留。原文逐篇重读并核对SHA256，财务与清洗文本经哈希确认复用；日期证据和候选重新生成。实际输入输出、数量及异常见本操作日志中`{r.name}`各步记录及run内JSON。申报日期、人工复核及标签公开日仍待完成。\n')
