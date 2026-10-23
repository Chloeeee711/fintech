from pathlib import Path
import pandas as pd, json, hashlib, datetime
root=Path(__file__).resolve().parents[3]
out=root/'formal_project/runs'/('raw_reaudit_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir()
a=root/'Compustat/g7watsrhufsgtynw.dta'
r=pd.read_stata(a,iterator=True); labels=r.variable_labels();r.close()
d=pd.read_stata(a,columns=['gvkey','cik','conm','datadate','apdedate','fyear','fyr','fdate','pdate'],convert_categoricals=False)
valid=d.apdedate.notna(); diff=valid & d.apdedate.ne(d.datadate)
i=pd.read_csv(root/'SEC 10-K ITEM 7/index.csv',dtype=str)
examples=d[d.cik.str.lstrip('0').isin(['20','1961'])]
examples.to_csv(out/'opened_companies.csv',index=False)
d.loc[diff].head(100).to_csv(out/'period_difference_examples.csv',index=False)
result={'annual_rows':len(d),'annual_columns':len(labels),'actual_period_nonmissing':int(valid.sum()),'actual_period_differs_datadate':int(diff.sum()),'cik_nonblank':int(d.cik.str.strip().ne('').sum()),'index_rows':len(i),'index_columns':list(i.columns),'date_labels':{k:labels[k] for k in ['datadate','apdedate','fyear','fyr','fdate','pdate']},'examples':examples.head(10).astype(str).to_dict('records')}
(out/'findings.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
files=[a,Path(__file__),root/'SEC 10-K ITEM 7/index.csv',root/'Compustat/data_description.txt',root/'SEC 10-K ITEM 7/data_description.txt']
(out/'fingerprints.json').write_text(json.dumps({str(p.relative_to(root)):sha(p) for p in files},indent=2))
print(out);print(json.dumps(result,ensure_ascii=False,indent=2))
