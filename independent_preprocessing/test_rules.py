import unittest
import pandas as pd
import numpy as np
from run import extract, panel
class Rules(unittest.TestCase):
 def test_explicit_actual_period(self):
  e=extract('All references to 2009, 2008 and 2007 mean the fiscal years ended January 2, 2010, January 3, 2009 and December 29, 2007.',2010)
  self.assertEqual({x['date'] for x in e},{'2010-01-02','2009-01-03','2007-12-29'})
 def test_subsequent_event_excluded(self):
  e=extract('Subsequent to the year ended December 31, 2022, on February 1, 2023 we closed a sale.',2023)
  self.assertEqual({x['date'] for x in e},{'2022-12-31'})
 def test_outlook_with_ended(self):
  for text in ['We anticipate our revenues for the year ended June 30, 2023 to reach 5 million.', 'The margin for the year ended June 30, 2023 is expected to be 60 percent.']:
   self.assertTrue(all(e['prospective_wording'] for e in extract(text,2023)))
 def test_reference_evidence(self):
  from refine_review import prior_reference
  self.assertTrue(prior_reference('Annual Report on Form 10-K for the fiscal year ended December 31, 2019, which was filed on February 19, 2020.'))
  self.assertFalse(prior_reference('Our revenue for the year ended December 31, 2020 increased.'))
 def test_no_guess(self):self.assertEqual(extract('Revenue improved in 2019. Capital expenditures expected in 2020.',2020),[])
 def test_forecast_not_historical(self):
  self.assertTrue(all(e['prospective_wording'] for e in extract('We expect spending for the year ending December 31, 2020.',2020)))
 def test_panel(self):
  rows=[]
  for year,assets,capx,ni in [(2018,100.,5.,10.),(2019,120.,12.,22.),(2020,140.,24.,26.)]:
   rows.append(dict(gvkey='1',cik='10',datadate=pd.Timestamp(year,12,31),apdedate=pd.Timestamp(year,12,31),fyear=year,sich=2000,sic='2000',datafmt='STD',indfmt='INDL',consol='C',curcd='USD',at=assets,capx=capx,ni=ni,sale=100.,che=10.,lt=30.))
  raw=pd.DataFrame(rows);d,_=panel(raw);r=d[d.fyear.eq(2019)].iloc[0]
  self.assertAlmostEqual(r.d_capex_at,24/120-12/100)
  self.assertAlmostEqual(r.droa,26/130-22/110)
  self.assertTrue(pd.isna(d[d.fyear.eq(2018)].iloc[0].d_capex_at))
  bad=raw.copy();bad.loc[bad.fyear.eq(2020),'sich']=9995
  scoped,rejected=panel(bad);self.assertEqual(len(rejected),1);self.assertTrue(scoped.d_capex_at.isna().all())
  gap,_=panel(raw[raw.fyear.ne(2019)]);self.assertTrue(gap.d_capex_at.isna().all())
  raw.loc[raw.fyear.eq(2020),'capx']=-1;d,_=panel(raw)
  self.assertTrue(d.d_capex_at.isna().all())
  self.assertTrue(d.d_capex_at_up.isna().all())
if __name__=='__main__':unittest.main()
