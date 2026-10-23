"""Raw-source preprocessing; evidence candidates are never silently certified."""
from pathlib import Path
import sys, re, json, hashlib, traceback
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; PROJECT=ROOT/'formal_project'
sys.path.insert(0,str(PROJECT/'src'))
from cleaning import clean_text, normalize_identifier
MONTHS='January February March April May June July August September October November December'.split()
DATE=re.compile(r'\b('+'|'.join(MONTHS)+r')\s+(\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(\d{4})\b',re.I)
# Explicit annual-period wording; shared-month comparative years are retained as evidence,
# but only dates written in full are parsed automatically.
ANNUAL=re.compile(r'(?:fiscal\s+)?years?\s+(?:ended|ending)\s+',re.I)
def extract(text, filing_year):
 text=re.sub(r'\s+',' ',text)
 records=[]
 for a in ANNUAL.finditer(text):
  prefix=text[max(0,a.start()-90):a.start()]
  if re.search(r'(?:three|six|nine|twelve)\s+months?\s+(?:and|or)\s*$',prefix,re.I): continue
  for m in DATE.finditer(text,a.end(),min(len(text),a.end()+180)):
   # Dates must follow the annual phrase directly or belong to its comparison list.
   between=text[a.end():m.start()]
   # Only a date-list prefix may intervene: shared years, commas and conjunctions.
   # Narrative words such as 'on', 'we closed', or 'subsequent' end the list.
   residue=DATE.sub('',between)
   residue=re.sub(r'\b(?:and|or|\d{4})\b|[,\s]', '', residue, flags=re.I)
   if residue or m.start()-a.end()>140: break
   try: date=pd.Timestamp(int(m[3]),next(j+1 for j,x in enumerate(MONTHS) if x.lower()==m[1].lower()),int(m[2]))
   except ValueError: continue
   if filing_year-3<=date.year<=filing_year:
    records.append({'date':date.isoformat()[:10],'context':text[max(0,a.start()-120):min(len(text),m.end()+180)],'offset':a.start(),'prospective_wording':bool(re.search(r'\b(?:ending|expect\w*|anticipat\w*|estimat\w*|projected|will|would|forecast\w*|budget\w*|outlook|target\w*)\b',text[max(0,a.start()-120):m.end()+100],re.I))})
 return records

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def panel(raw):
 d=raw.copy();d['source_row']=np.arange(len(d))
 for col,w in [('gvkey',6),('cik',None)]:d[col]=d[col].map(lambda v:normalize_identifier(v,w))
 h=pd.to_numeric(d.sich,errors='coerce');c=pd.to_numeric(d.sic,errors='coerce')
 d['industry_sic']=h.where(h.notna(),c)
 d['industry_source']=np.where(h.notna(),'historical_sich','company_sic_fallback')
 mask=d.datafmt.eq('STD')&d.indfmt.eq('INDL')&d.consol.eq('C')&d.curcd.eq('USD')&d.industry_sic.between(100,8999)&~d.industry_sic.between(6000,6999)&d.gvkey.notna()&d.fyear.notna()&d.datadate.notna()
 d['exclusion_reason']=np.where(mask,'','format_currency_industry_or_key')
 duplicate=mask & d.loc[mask].duplicated(['gvkey','fyear'],keep=False).reindex(d.index,fill_value=False)
 d.loc[duplicate,'exclusion_reason']='ambiguous_company_fiscal_year'
 excluded=d[d.exclusion_reason.ne('')].copy();d=d[d.exclusion_reason.eq('')].copy()
 d['period_end']=d.apdedate.combine_first(d.datadate)
 d['period_source']=np.where(d.apdedate.notna(),'apdedate','datadate_fallback')
 d['fyear']=d.fyear.astype(int)
 for delta,suffix in [(1,'prev'),(-1,'next')]:
  other=d[['gvkey','fyear','at','capx','ni','sale','period_end']].copy();other.fyear+=delta
  other=other.rename(columns={c:c+'_'+suffix for c in other if c not in ['gvkey','fyear']})
  d=d.merge(other,on=['gvkey','fyear'],how='left',validate='one_to_one')
 prev=(d.period_end-d.period_end_prev).dt.days.between(330,400)
 nxt=(d.period_end_next-d.period_end).dt.days.between(330,400)
 d['annual_periods_comparable']=prev&nxt
 d['capex_int']=(d.capx/d.at_prev).where(prev&d.capx.ge(0)&d.at_prev.gt(0))
 d['capex_int_fwd']=(d.capx_next/d['at']).where(nxt&d.capx_next.ge(0)&d['at'].gt(0))
 d['d_capex_at']=d.capex_int_fwd-d.capex_int
 d['roa']=(d.ni/((d.at_prev+d['at'])/2)).where(prev&d.at_prev.gt(0)&d['at'].gt(0))
 d['roa_fwd']=(d.ni_next/((d['at']+d.at_next)/2)).where(nxt&d['at'].gt(0)&d.at_next.gt(0))
 d['droa']=d.roa_fwd-d.roa
 for y in ['d_capex_at','droa']:d[y+'_up']=d[y].gt(0).astype(float).where(d[y].notna())
 d['sales_growth']=(d.sale/d.sale_prev-1).where(prev&d.sale_prev.gt(0)&d.sale.ge(0))
 d['cash_at']=(d.che/d['at']).where(d.che.ge(0)&d['at'].gt(0))
 d['debt_at']=(d['lt']/d['at']).where(d['lt'].ge(0)&d['at'].gt(0))
 d['log_at']=np.log(d['at'].where(d['at'].gt(0)))
 d['industry2']=d.industry_sic.floordiv(100).astype('Int64').astype('string')
 return d,excluded

def main():
 stamp=datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%dT%H%M%S_%f')
 run=PROJECT/'runs'/('independent_'+stamp);run.mkdir()
 def save(name,obj): (run/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str))
 def log(msg):
  with (PROJECT/'docs/操作日志.md').open('a') as f:f.write('\n['+run.name+'] '+msg+'\n\n')
  print(msg,flush=True)
 log('# 独立原始数据预处理\n\n命令：/private/tmp/fintech-quality-venv/bin/python formal_project/independent_preprocessing/run.py')
 save('config.json',{'window':[2016,2023],'train':[2016,2019],'validation':[2020],'buffer':[2021],'test':[2022,2023],'date_match':'explicit annual wording, actual period preferred; candidate only','annual_gap_days':[330,400],'fitted_operations':False})
 try:
  inputs=[ROOT/'Compustat/g7watsrhufsgtynw.dta',ROOT/'SEC 10-K ITEM 7/index.csv',ROOT/'Compustat/data_description.txt',ROOT/'SEC 10-K ITEM 7/data_description.txt',ROOT/'analysis/资本投资选题最终方案.md',Path(__file__),PROJECT/'src/cleaning.py']
  save('fingerprints.json',{str(p.relative_to(ROOT)):sha(p) for p in inputs})
  cols=['gvkey','cik','conm','datadate','apdedate','fyear','fyr','fdate','pdate','sic','sich','datafmt','indfmt','consol','curcd','at','capx','ni','sale','che','lt']
  raw=pd.read_stata(inputs[0],columns=cols,convert_categoricals=False)
  fin,excluded=panel(raw);fin.to_parquet(run/'financial_panel.parquet',index=False);excluded.to_parquet(run/'financial_excluded.parquet',index=False)
  log(f'原始财务{len(raw)}行；范围内{len(fin)}行；隔离{len(excluded)}行。主Y{fin.d_capex_at.notna().sum()}，辅助Y{fin.droa.notna().sum()}。')
  idx=pd.read_csv(inputs[1],dtype=str);idx['year']=idx.year.astype(int);idx=idx[idx.year.between(2016,2023)].copy()
  bygv={k:v for k,v in fin.groupby('gvkey')};bycik={k:v for k,v in fin.dropna(subset=['cik']).groupby('cik')}
  records=[];evidence=[];batch=[];(run/'texts').mkdir()
  for n,row in enumerate(idx.itertuples(index=False)):
   g=normalize_identifier(row.gvkey,6);c=normalize_identifier(row.cik);acc=row.accession_number_nodash
   rec={'accession_key':acc,'index_gvkey':g,'cik':c,'filing_year':row.year,'filename':row.filename,'filing_date':None,'form_type':None,'strict_training_ready':False}
   try:
    if Path(row.filename).name!=row.filename or not re.fullmatch(r'\d+_\d{18}\.txt',row.filename):raise ValueError('invalid filename')
    p=ROOT/'SEC 10-K ITEM 7'/str(row.year)/row.filename;b=p.read_bytes();rawtext=b.decode('utf-8');clean,quality=clean_text(rawtext)
    rec.update(quality);rec['raw_sha256']=hashlib.sha256(b).hexdigest();rec['clean_sha256']=hashlib.sha256(clean.encode()).hexdigest()
    ev=extract(rawtext,row.year)
    evidence.extend({'accession_key':acc,**e} for e in ev)
    dates=sorted({e['date'] for e in ev if not e['prospective_wording']})
    # Most recent explicit historical annual date is a candidate, never inferred from folder year.
    date=pd.Timestamp(dates[-1]) if dates else None;rec['text_period_candidate']=date
    scope=bygv.get(g) if g else bycik.get(c)
    rec['match_status']='no_annual_date_evidence' if date is None else 'company_outside_scope'
    if date is not None and scope is not None:
     matches=scope[scope.period_end.eq(date)]
     rec['match_status']='period_not_in_financial_scope'
     if len(matches)==1:
      f=matches.iloc[0]
      if pd.notna(f.cik) and f.cik!=c:rec['match_status']='company_identifier_conflict'
      else:
       rec.update({'matched_gvkey':f.gvkey,'matched_fyear':int(f.fyear),'matched_period_source':f.period_source,'match_status':'period_candidate_requires_review'})
     elif len(matches)>1:rec['match_status']='ambiguous_company_period'
    rec['read_status']='ok';batch.append({'accession_key':acc,'text_natural':clean})
   except (OSError,ValueError,UnicodeError) as e:rec.update(read_status='error',match_status='read_error',error=str(e))
   records.append(rec)
   if len(batch)>=256:
    pd.DataFrame(batch).to_parquet(run/'texts'/f'{n:06d}.parquet',index=False);batch=[]
   if (n+1)%4000==0:log(f'已处理原文{n+1}/{len(idx)}份。')
  if batch:pd.DataFrame(batch).to_parquet(run/'texts'/'last.parquet',index=False)
  audit=pd.DataFrame(records)
  audit['duplicate_clean_text']=audit.clean_sha256.notna()&audit.clean_sha256.duplicated(keep=False)
  audit['duplicate_company_period']=audit.matched_gvkey.notna()&audit.duplicated(['matched_gvkey','matched_fyear'],keep=False)
  audit['requires_text_review']=audit.quality_flags.fillna('').ne('')|audit.duplicate_clean_text|audit.read_status.ne('ok')
  audit.to_parquet(run/'filing_audit.parquet',index=False);pd.DataFrame(evidence).to_parquet(run/'period_evidence.parquet',index=False)
  candidates=audit[audit.match_status.eq('period_candidate_requires_review')].merge(fin,left_on=['matched_gvkey','matched_fyear'],right_on=['gvkey','fyear'],how='left',validate='many_to_one',suffixes=('','_financial'))
  candidates['proposed_split']=np.select([candidates.filing_year.le(2019),candidates.filing_year.eq(2020),candidates.filing_year.eq(2021)],['train','validation','buffer'],default='test')
  candidates['review_status']='unreviewed';candidates['label_public_date']=pd.NaT
  candidates.to_parquet(run/'candidate_sample.parquet',index=False)
  candidates[candidates.d_capex_at.notna()].to_parquet(run/'main_task_candidates.parquet',index=False)
  candidates[candidates.droa.notna()].to_parquet(run/'aux_task_candidates.parquet',index=False)
  sample=candidates.sample(min(100,len(candidates)),random_state=20261023)[['accession_key','filename','filing_year','matched_gvkey','matched_fyear','text_period_candidate','matched_period_source','duplicate_company_period','requires_text_review']]
  evdf=pd.DataFrame(evidence);contexts=evdf.groupby('accession_key').context.agg(lambda x:'\n---\n'.join(dict.fromkeys(x)))
  sample=sample.join(contexts,on='accession_key')
  for col in ['reviewer_1_period','reviewer_2_period','form_type_verified','filing_date_verified','review_notes']:sample[col]=''
  sample.to_csv(run/'period_review_100.csv',index=False)
  checks={'financial_count_conserved':len(fin)+len(excluded)==len(raw),'financial_keys_unique':not fin.duplicated(['gvkey','fyear']).any(),'all_window_texts_accounted':len(audit)==len(idx),'accessions_unique':not audit.accession_key.duplicated().any(),'missing_main_y_not_imputed':fin.loc[fin.d_capex_at.isna(),'d_capex_at_up'].isna().all(),'no_certified_training_rows':not audit.strict_training_ready.any()}
  save('validation.json',{k:bool(v) for k,v in checks.items()});assert all(checks.values())
  summary={'financial_rows':len(fin),'text_rows':len(audit),'match_status':audit.match_status.value_counts().to_dict(),'period_candidates':len(candidates),'main_y_candidates':int(candidates.d_capex_at.notna().sum()),'aux_y_candidates':int(candidates.droa.notna().sum()),'candidate_duplicate_company_period':int(candidates.duplicate_company_period.sum()),'candidate_text_review':int(candidates.requires_text_review.sum()),'candidate_actual_period_matches':int(candidates.matched_period_source.eq('apdedate').sum()),'strict_training_ready':False,'fitted_operations':False,'limitations':'Annual date rules produce unreviewed candidates; filing day and form type unavailable. Do not use proposed splits for fitting until date availability and period review pass.'}
  save('summary.json',summary);log('完成：'+json.dumps(summary,ensure_ascii=False))
  save('output_sha256.json',{str(p.relative_to(run)):sha(p) for p in run.rglob('*') if p.is_file()})
  save('status.json',{'status':'completed_candidate_preprocessing','training_ready':False})
  (PROJECT/'independent_preprocessing/latest_run.json').write_text(json.dumps({'run_dir':str(run)},indent=2))
 except Exception:
  log(traceback.format_exc());save('status.json',{'status':'failed'});raise
 finally:
  with (PROJECT/'docs/操作日志.md').open('a') as f:f.write(f'\n## {stamp}：独立原始数据预处理\n\n命令：`/private/tmp/fintech-quality-venv/bin/python formal_project/independent_preprocessing/run.py`。代码、配置、输入指纹、逐步数量、失败与未完成事项见本操作日志中`{run.name}`各步记录及run内JSON。不修改原始数据；不使用第一版输出；仅复用经阅读的确定性文本清洗函数。候选证据尚不等同人工确认或可训练样本。\n')
if __name__=='__main__':main()
