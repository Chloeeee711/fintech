"""Evidence/duplicate review stage. Append only to existing Markdown documents."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,sys,traceback,re
import numpy as np
import pandas as pd
from run import PROJECT,sha
SOURCE=Path(json.loads((PROJECT/'independent_preprocessing/latest_run.json').read_text())['run_dir'])
r=PROJECT/'runs'/('period_review_'+datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y%m%dT%H%M%S_%f'));r.mkdir()
def save(name,data): (r/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str))
def log(msg):
 with (PROJECT/'docs/操作日志.md').open('a') as f:f.write('\n'+msg+'\n')
 print(msg,flush=True)
log(f'## 2026-10-23：期间证据与重复样本审计\n\nrun：`{r.name}`。实际命令：`/private/tmp/fintech-quality-venv/bin/python formal_project/independent_preprocessing/review_candidates.py`。输入：`{SOURCE.name}`。按用户最新要求，本轮不新建Markdown；每步写入现有操作日志，数据输出独立保存。')
try:
 inputs=[SOURCE/'candidate_sample.parquet',SOURCE/'period_evidence.parquet',SOURCE/'filing_audit.parquet']
 save('provenance.json',{'source_run':str(SOURCE),'inputs':{str(p):sha(p) for p in inputs},'code_sha256':sha(Path(__file__)),'python':sys.version})
 save('config.json',{'evidence_rule':'exact candidate-date evidence, nonprospective, unique normalized offsets','duplicate_policy':'quarantine whole company-fiscal-year group until form/date verified','human_review':'assistant evidence check distinct from two human annotators','fit_operations':False})
 c=pd.read_parquet(inputs[0]);e=pd.read_parquet(inputs[1]);a=pd.read_parquet(inputs[2])
 e['date']=pd.to_datetime(e.date)
 e=e.merge(c[['accession_key','text_period_candidate']],on='accession_key',how='inner',validate='many_to_one')
 target=e[e.date.eq(e.text_period_candidate)&~e.prospective_wording].drop_duplicates(['accession_key','offset','date']).copy()
 target['results_context']=target.context.str.contains(r'results of operations|net (?:income|loss)|revenues?|net sales|consolidated financial|fiscal (?:year|201\d|202\d)',case=False,regex=True)
 target['third_party_context']=target.context.str.contains(r'acquired|acquisition|predecessor|successor|pro forma|investee|subsidiar',case=False,regex=True)
 stats=target.groupby('accession_key').agg(period_evidence_locations=('offset','nunique'),results_evidence_locations=('results_context','sum'),third_party_context_locations=('third_party_context','sum'))
 c=c.join(stats,on='accession_key');assert c.period_evidence_locations.notna().all()
 c['evidence_priority']=np.select([c.period_evidence_locations.eq(1),c.results_evidence_locations.eq(0),c.third_party_context_locations.gt(0)],['single_location','no_results_context','third_party_wording'],default='repeated_results_evidence')
 groups=[]
 for (g,y),part in c.groupby(['gvkey','fyear']):
  if len(part)<2:continue
  groups.append({'gvkey':g,'fyear':int(y),'rows':len(part),'distinct_text_hashes':part.clean_sha256.nunique(),'filing_year_min':int(part.filing_year.min()),'filing_year_max':int(part.filing_year.max()),'group_type':'identical_text' if part.clean_sha256.nunique()==1 else 'different_texts','accessions':'|'.join(part.accession_key),'resolution':'quarantined_pending_form_and_filing_date'})
 dup=pd.DataFrame(groups);dup.to_csv(r/'duplicate_groups.csv',index=False)
 # Sequential funnel; no arbitrary deletion of one member of an unresolved duplicate group.
 steps=[];work=c.copy()
 for name,condition in [('exclude_unresolved_company_period_duplicates',~work.duplicate_company_period),('exclude_text_quality_flags',~c.requires_text_review),('require_main_y',c.d_capex_at.notna())]:
  before=len(work);work=work.loc[condition.reindex(work.index).fillna(False)].copy();steps.append({'step':name,'before':before,'after':len(work),'removed':before-len(work)})
 work['sample_status']='eligible_for_period_review_not_training';work.to_parquet(r/'main_review_pool.parquet',index=False)
 c['sample_status']=np.where(c.duplicate_company_period,'duplicate_group_quarantine',np.where(c.requires_text_review,'text_quality_quarantine','period_review_pending'))
 c.to_parquet(r/'candidate_review_audit.parquet',index=False)
 c[c.sample_status.ne('period_review_pending')].to_parquet(r/'quarantine.parquet',index=False)
 pd.DataFrame(steps).to_csv(r/'sample_flow.csv',index=False)
 c.groupby(['filing_year','evidence_priority']).size().rename('rows').to_csv(r/'evidence_by_year.csv')
 # Diagnostic sample stratified by priority, not an unbiased accuracy evaluation.
 selections=[]
 for key,part in c.groupby('evidence_priority'):
  selections.append(part.sample(min(6,len(part)),random_state=20261023))
 sample=pd.concat(selections).sort_values(['evidence_priority','accession_key'])
 excerpts=target.groupby('accession_key').context.agg(lambda x:'\n---\n'.join(list(dict.fromkeys(x))[:3]))
 sample=sample[['accession_key','filename','filing_year','gvkey','fyear','period_end','conm','evidence_priority','duplicate_company_period','requires_text_review','period_evidence_locations']].join(excerpts,on='accession_key')
 sample['assistant_review']='pending';sample['assistant_notes']='';sample['human_reviewer_1']='';sample['human_reviewer_2']='';sample.to_csv(r/'diagnostic_review.csv',index=False)
 checks={'candidate_count_preserved':len(c)==26343,'all_candidates_have_matching_raw_evidence':bool(c.period_evidence_locations.ge(1).all()),'review_pool_company_year_unique':not work.duplicated(['gvkey','fyear']).any(),'review_pool_main_y_complete':bool(work.d_capex_at.notna().all()),'review_pool_no_text_flags':not work.requires_text_review.any(),'review_pool_no_certified_rows':not work.strict_training_ready.any(),'duplicate_group_row_count_agrees':int(dup.rows.sum())==int(c.duplicate_company_period.sum())}
 save('validation.json',checks);assert all(checks.values())
 summary={'input_candidates':len(c),'duplicate_groups':len(dup),'duplicate_rows':int(dup.rows.sum()),'duplicate_group_types':dup.group_type.value_counts().to_dict(),'evidence_priority_counts':c.evidence_priority.value_counts().to_dict(),'main_review_pool':len(work),'diagnostic_sample':len(sample),'checks_passed':sum(checks.values()),'training_ready':False,'fit_executed':False}
 save('summary.json',summary);save('status.json',{'status':'completed_evidence_audit','training_ready':False});save('output_sha256.json',{p.name:sha(p) for p in r.iterdir() if p.is_file()})
 (PROJECT/'independent_preprocessing/latest_review.json').write_text(json.dumps({'run_dir':str(r)},indent=2))
 log('完成本轮证据统计与重复分组：'+json.dumps(summary,ensure_ascii=False)+'。重复组整组隔离，未挑选一份假定原始10-K；无质量标记仅表示进入期间复核池，不表示已复核或已可训练。')
 with (PROJECT/'docs/独立预处理结果.md').open('a') as f:
  f.write(f'\n## 2026-10-23追加：期间证据和重复组审计\n\n独立run：`{r.name}`。已有{len(c):,}份候选均能回查到对应日期的非预测措辞证据；这不是语义准确率证明。发现{len(dup):,}个重复公司年度组，涉及{int(dup.rows.sum()):,}行；按整组隔离处理，未以编号或文本长度擅自选择。再排除文本复核标记、要求主Y有效后，得到{len(work):,}份期间复核池，尚未通过人工确认与日期可得性检查。\n\n证据分层：`{summary["evidence_priority_counts"]}`。固定种子每类最多6份，共{len(sample)}份诊断抽查，刻意覆盖薄弱证据，因此不能把通过比例当作总体准确率。两个组员复核栏仍为空。7项自动一致性检查通过。\n\n输出：`duplicate_groups.csv`、`candidate_review_audit.parquet`、`main_review_pool.parquet`、`sample_flow.csv`、`diagnostic_review.csv`及JSON校验/指纹，均在该run。未新增Markdown文件。\n')
except Exception:
 save('status.json',{'status':'failed','traceback':traceback.format_exc()});log(traceback.format_exc());raise
