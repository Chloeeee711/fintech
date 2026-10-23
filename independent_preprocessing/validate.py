from pathlib import Path
import json,sys,hashlib
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'formal_project'
r=Path(json.loads((P/'independent_preprocessing/latest_run.json').read_text())['run_dir'])
d=pd.read_parquet(r/'financial_panel.parquet');a=pd.read_parquet(r/'filing_audit.parquet');c=pd.read_parquet(r/'candidate_sample.parquet')
raw=pd.read_stata(ROOT/'Compustat/g7watsrhufsgtynw.dta',columns=['gvkey','fyear','datadate','apdedate','at','capx','ni'],convert_categoricals=False)
checks={}
for y in ['d_capex_at','droa']:
 errors=[]
 for row in d[d[y].notna()].sample(min(500,d[y].notna().sum()),random_state=36).itertuples():
  q=raw[raw.gvkey.eq(row.gvkey)&raw.fyear.isin([row.fyear-1,row.fyear,row.fyear+1])].set_index('fyear')
  # Source may contain rows outside accepted financial format; disambiguate by exact source_row below.
  ids=d[d.gvkey.eq(row.gvkey)&d.fyear.isin([row.fyear-1,row.fyear,row.fyear+1])].set_index('fyear').source_row
  q=raw.loc[ids.values].set_index('fyear');p,t,n=[q.loc[x] for x in [row.fyear-1,row.fyear,row.fyear+1]]
  expected=n.capx/t['at']-t.capx/p['at'] if y=='d_capex_at' else n.ni/((t['at']+n['at'])/2)-t.ni/((p['at']+t['at'])/2)
  errors.append(np.isclose(expected,getattr(row,y),rtol=1e-12,atol=1e-12))
 checks[y+'_500_source_recalculations']=all(errors)
checks['candidate_dates_equal_effective_period']=bool(c.text_period_candidate.eq(c.period_end).all())
checks['actual_period_preferred']=bool(d.loc[d.apdedate.notna(),'period_end'].eq(d.loc[d.apdedate.notna(),'apdedate']).all())
checks['all_dates_remain_uninvented']=bool(a.filing_date.isna().all())
checks['no_candidate_certified']=bool((~c.strict_training_ready).all())
checks['no_label_date_invented']=bool(c.label_public_date.isna().all())
textdir=Path(json.loads((r/'reused_artifacts.json').read_text())['text_directory']) if (r/'reused_artifacts.json').exists() else r/'texts'
checks['all_text_part_rows_accounted']=sum(len(pd.read_parquet(p,columns=['accession_key'])) for p in textdir.glob('*.parquet'))==int(a.read_status.eq('ok').sum())
report={'checks':checks,'passed':sum(checks.values()),'total':len(checks)}
(r/'independent_validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2));assert all(checks.values())
