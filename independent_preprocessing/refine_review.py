"""Rebuild financial scope and isolate dates supported only by prior-report references."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,re,traceback
import pandas as pd
import numpy as np
from run import PROJECT,ROOT,panel,sha

def prior_reference(context):
 return bool(re.search(r'annual report|form\s+10[\s-]*k',context,re.I) and re.search(r'which\s+was\s+filed|previously\s+filed|incorporated\s+by\s+reference|refer\s+to',context,re.I))

def main():
 source=Path(json.loads((PROJECT/'independent_preprocessing/latest_run.json').read_text())['run_dir'])
 previous_review=Path(json.loads((PROJECT/'independent_preprocessing/latest_review.json').read_text())['run_dir'])
 if not (previous_review/'diagnostic_review.csv').exists():
  previous_review=Path(json.loads((previous_review/'provenance.json').read_text())['prior_review_run'])
 r=PROJECT/'runs'/('scope_period_refined_'+datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%dT%H%M%S_%f'));r.mkdir()
 def save(name,obj):(r/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str))
 def log(msg):
  with (PROJECT/'docs/操作日志.md').open('a') as f:f.write('\n'+msg+'\n')
  print(msg,flush=True)
 log(f'## 2026-10-23：历史行业及旧报告引用修正\n\nrun：`{r.name}`。实际命令：`/private/tmp/fintech-quality-venv/bin/python formal_project/independent_preprocessing/refine_review.py`。原始财务重读，现有文本/证据按哈希复用；所有说明追加既有MD。')
 try:
  files=[ROOT/'Compustat/g7watsrhufsgtynw.dta',source/'filing_audit.parquet',source/'period_evidence.parquet',previous_review/'diagnostic_review.csv',Path(__file__),Path(__file__).with_name('run.py')]
  save('provenance.json',{'source_run':str(source),'prior_review_run':str(previous_review),'sha256':{str(p):sha(p) for p in files}})
  manifest=json.loads((source/'output_sha256.json').read_text())
  for p in files[1:3]:assert sha(p)==manifest[p.name]
  save('config.json',{'historical_sic':'fallback only when sich missing; explicit out-of-scope codes excluded','prior_report_reference':'quarantine if all nonprospective candidate-date evidence references another report','no_fit':True})
  cols=['gvkey','cik','conm','datadate','apdedate','fyear','fyr','fdate','pdate','sic','sich','datafmt','indfmt','consol','curcd','at','capx','ni','sale','che','lt']
  raw=pd.read_stata(files[0],columns=cols,convert_categoricals=False);fin,excluded=panel(raw)
  fin.to_parquet(r/'financial_panel.parquet',index=False);excluded.to_parquet(r/'financial_excluded.parquet',index=False)
  a=pd.read_parquet(files[1]);a=a[a.match_status.eq('period_candidate_requires_review')].copy()
  c=a.merge(fin,left_on=['matched_gvkey','matched_fyear'],right_on=['gvkey','fyear'],how='left',suffixes=('','_financial'),validate='many_to_one',indicator=True)
  e=pd.read_parquet(files[2]);e['date']=pd.to_datetime(e.date)
  e=e.merge(a[['accession_key','text_period_candidate']],on='accession_key',validate='many_to_one')
  target=e[e.date.eq(e.text_period_candidate)&~e.prospective_wording].drop_duplicates(['accession_key','offset','date']).copy()
  target['prior_report_reference']=target.context.map(prior_reference)
  target.to_parquet(r/'target_period_evidence.parquet',index=False)
  counts=target.groupby('accession_key').prior_report_reference.agg(total_evidence='size',reference_evidence='sum')
  c=c.join(counts,on='accession_key');c['reference_only_period']=c.total_evidence.eq(c.reference_evidence)
  c['financial_scope_pass']=c['_merge'].eq('both');c=c.drop(columns='_merge')
  # Scope changes affect neighboring financial labels too; new Y comes only from rebuilt panel.
  c['assistant_text_scope_review']=c.accession_key.eq('000082705220000026')
  c['review_status']='pending_human_period_and_metadata_review';c['strict_training_ready']=False
  c['proposed_split']=np.select([c.filing_year.le(2019),c.filing_year.eq(2020),c.filing_year.eq(2021)],['train','validation','buffer'],default='test')
  c['label_public_date']=pd.NaT
  flow=[];pool=c.copy()
  for col,name in [('financial_scope_pass','require_current_financial_scope'),('reference_only_period','exclude_reference_only_dates'),('duplicate_company_period','exclude_unresolved_duplicate_groups'),('requires_text_review','exclude_existing_text_flags'),('assistant_text_scope_review','exclude_assistant_text_scope_flag')]:
   before=len(pool);mask=pool[col] if col=='financial_scope_pass' else ~pool[col];pool=pool[mask].copy();flow.append({'step':name,'before':before,'removed':before-len(pool),'after':len(pool)})
  generic=pool.copy();before=len(pool);pool=pool[pool.d_capex_at.notna()].copy();flow.append({'step':'require_main_y','before':before,'removed':before-len(pool),'after':len(pool)})
  c.to_parquet(r/'candidate_review_audit.parquet',index=False);pool.to_parquet(r/'main_review_pool.parquet',index=False);generic[generic.droa.notna()].to_parquet(r/'aux_review_pool.parquet',index=False)
  c[c.reference_only_period].to_csv(r/'prior_reference_quarantine.csv',index=False);pd.DataFrame(flow).to_csv(r/'sample_flow.csv',index=False)
  oldfin=pd.read_parquet(source/'financial_panel.parquet');removed=oldfin[~oldfin.source_row.isin(fin.source_row)];removed.to_csv(r/'financial_scope_removed.csv',index=False)
  # Record actual assistant excerpt review, not fictional human annotators or full document certification.
  sample=pd.read_csv(previous_review/'diagnostic_review.csv',dtype=str)
  notes={6:('text_scope_requires_review','开头为目录，日期证据另含财务报表附注；需核验MD&A边界，暂隔离。'),7:('company_scope_rejected','原文明确blank check company；历史sich=9995被当前sic=3672覆盖，修正后排除。'),12:('period_candidate_rejected','原文2020 COMPARED WITH 2019；唯一完整2019日期引用已提交的旧年报，不能作本篇期末。')}
  for j in sample.index:
   decision,note=notes.get(j,('excerpt_supports_period','已阅读所列最多三段原句，支持所列公司期间；未核验正式申报日、表格类型或全篇语义。'))
   sample.loc[j,'assistant_review']=decision;sample.loc[j,'assistant_notes']=note
  sample.to_csv(r/'diagnostic_review_completed.csv',index=False)
  checks={'financial_count_conserved':len(fin)+len(excluded)==len(raw),'historical_out_of_scope_not_overridden':not fin.sich.notna().mul(~fin.sich.between(100,8999)).any(),'all_input_candidates_accounted':len(c)==len(a),'review_pool_unique':not pool.duplicated(['matched_gvkey','matched_fyear']).any(),'reference_only_excluded':not pool.reference_only_period.any(),'main_y_present':pool.d_capex_at.notna().all(),'not_certified':not pool.strict_training_ready.any(),'known_wrong_reference_excluded':not pool.accession_key.eq('000001823021000063').any(),'known_blank_check_excluded':not pool.accession_key.eq('000141057822000335').any()}
  save('validation.json',{k:bool(v) for k,v in checks.items()});assert all(checks.values())
  summary={'original_candidates':len(c),'financial_rows':len(fin),'financial_rows_removed_vs_previous':len(removed),'financial_main_y':int(fin.d_capex_at.notna().sum()),'financial_aux_y':int(fin.droa.notna().sum()),'candidate_scope_exclusions':int((~c.financial_scope_pass).sum()),'reference_only_candidates':int(c.reference_only_period.sum()),'main_review_pool':len(pool),'aux_review_pool':int(generic.droa.notna().sum()),'assistant_review_counts':sample.assistant_review.value_counts().to_dict(),'automatic_checks':len(checks),'training_ready':False}
  save('summary.json',summary);save('status.json',{'status':'completed_scope_and_period_review','training_ready':False})
  save('output_sha256.json',{p.name:sha(p) for p in r.iterdir() if p.is_file()})
  (PROJECT/'independent_preprocessing/latest_review.json').write_text(json.dumps({'run_dir':str(r)},indent=2))
  log('完成：'+json.dumps(summary,ensure_ascii=False)+'。新财务面板重新按完整财年构造Y；尚缺真实申报日、修订身份和独立人工复核，未拟合任何参数。')
  with (PROJECT/'docs/独立预处理结果.md').open('a') as f:f.write(f'\n## 2026-10-23追加：抽查发现问题后的范围修正\n\n本轮run：`{r.name}`。24份分层诊断样本中，21份原句支持候选期间，1份引用旧年报而误配，1份文本范围需再核验，1份当年为空白支票公司且被错误行业回退纳入。该抽样特意覆盖薄弱证据，不能用21/24估计总体正确率；也不是两名组员独立标注。\n\n修正：历史sich有值时不被当前行业覆盖；历史缺失才回退。重新读取原始财务表并重建前后年及Y，而非仅删除当年候选。仅靠旧年报引用支撑的期间候选隔离，不猜测替代期间。\n\n实际结果：财务范围{len(fin):,}行，比上一轮少{len(removed):,}行；现有候选中{int((~c.financial_scope_pass).sum()):,}份不再符合财务范围，{int(c.reference_only_period.sum()):,}份只有旧报告引用证据（两类可能重叠）。完成顺序筛选后，主任务期间复核池{len(pool):,}份、辅助任务{int(generic.droa.notna().sum()):,}份。**仍不是可训练样本。**\n\n本轮9项一致性校验通过，修改后的6项规则测试通过；输出含重新构造的财务面板、候选审计、逐步筛选表、旧报告引用隔离表和助手24份原句复核记录。两名人工复核栏保持空白。所有结果说明追加既有MD，没有新建Markdown。\n')
 except Exception:
  save('status.json',{'status':'failed','traceback':traceback.format_exc()});log(traceback.format_exc());raise
if __name__=='__main__':main()
