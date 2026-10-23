from pathlib import Path
import json
import pandas as pd
import numpy as np
from run import PROJECT,ROOT,sha
r=Path(json.loads((PROJECT/'independent_preprocessing/latest_review.json').read_text())['run_dir'])
d=pd.read_parquet(r/'financial_panel.parquet');pool=pd.read_parquet(r/'main_review_pool.parquet')
raw=pd.read_stata(ROOT/'Compustat/g7watsrhufsgtynw.dta',columns=['gvkey','fyear','datadate','apdedate','at','capx','ni','sich','sic'],convert_categoricals=False)
lookup=d.set_index(['gvkey','fyear']).source_row
checks={}
for y in ['d_capex_at','droa']:
 tests=[]
 for row in d[d[y].notna()].sample(500,random_state=71).itertuples():
  ids=[lookup.loc[(row.gvkey,row.fyear+k)] for k in [-1,0,1]]
  p,t,n=[raw.loc[j] for j in ids]
  result=n.capx/t['at']-t.capx/p['at'] if y=='d_capex_at' else n.ni/((t['at']+n['at'])/2)-t.ni/((p['at']+t['at'])/2)
  tests.append(np.isclose(result,getattr(row,y),rtol=1e-12,atol=1e-12))
 checks[y+'_500_source_checks']=all(tests)
source=raw.loc[d.source_row]
checks['no_explicit_invalid_historical_sic']=bool((source.sich.isna()|source.sich.between(100,8999)).all())
checks['main_pool_no_reference_only']=not pool.reference_only_period.any()
checks['missing_y_not_imputed']=bool(d.loc[d.d_capex_at.isna(),'d_capex_at_up'].isna().all())
checks['company_year_unique']=not pool.duplicated(['gvkey','fyear']).any()
checks['filing_date_not_invented']=bool(pool.filing_date.isna().all())
checks['raw_financial_fingerprint_matches']=sha(ROOT/'Compustat/g7watsrhufsgtynw.dta')==json.loads((r/'provenance.json').read_text())['sha256'][str(ROOT/'Compustat/g7watsrhufsgtynw.dta')]
checks={k:bool(v) for k,v in checks.items()}
(r/'independent_validation.json').write_text(json.dumps({'checks':checks,'passed':sum(checks.values()),'total':len(checks),'validator_sha256':sha(Path(__file__))},indent=2))
print(json.dumps(checks,indent=2));assert all(checks.values())
