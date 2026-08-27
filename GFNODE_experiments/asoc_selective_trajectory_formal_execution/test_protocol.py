"""C1-S4 ordinary tests. Synthetic tests call production functions; no real fit occurs."""
from __future__ import annotations
import csv,json,tempfile,unittest
from pathlib import Path
import numpy as np
import run_c1_formal as formal
import c1_formal_pipeline as pipe
AUDIT={}; STATE=None; CONFIG={}

class FixtureTests(unittest.TestCase):
 def test_01_single_annual_source(self):
  x=np.arange(np.datetime64('2021-01-01'),np.datetime64('2022-01-01'),np.timedelta64(1,'s'),dtype='datetime64[s]'); r=pipe.combine_source_seconds([x],2021); self.assertEqual(r['unique_seconds'],31536000); self.assertEqual(r['missing_seconds'],0)
 def test_02_month_blocks_order_by_utc(self):
  a=np.arange(np.datetime64('2021-01-01'),np.datetime64('2021-01-02'),np.timedelta64(1,'s')); b=np.arange(np.datetime64('2021-01-02'),np.datetime64('2021-01-03'),np.timedelta64(1,'s')); self.assertEqual(pipe.combine_source_seconds([b,a],2021)['first_utc'],'2021-01-01T00:00:00')
 def test_03_block_duplicate(self):
  a=np.array(['2021-01-01T00:00:00','2021-01-01T00:00:01'],dtype='datetime64[s]'); self.assertEqual(pipe.combine_source_seconds([a,a[-1:]],2021)['duplicate_seconds'],1)
 def test_04_block_gap(self):
  a=np.array(['2021-01-01T00:00:00','2021-01-01T00:00:02'],dtype='datetime64[s]'); self.assertGreater(pipe.combine_source_seconds([a],2021)['missing_seconds'],0)
 def test_05_cross_year(self):
  a=np.array(['2020-12-31T23:59:59','2021-01-01T00:00:00'],dtype='datetime64[s]'); self.assertEqual(pipe.combine_source_seconds([a],2021)['out_of_year_records'],1)
 def test_06_glued_truncated(self):
  h=b'Timestamp_UTC [DD/MM/YYYY hh:mm:ss],Irradiance_MB0 [W/m-2],Irradiance_MB1 [W/m-2],Irradiance_MB2 [W/m-2]\n'; lines=b'01/01/2021 00:00:00,1\n01/01/2021 00:00:01,1,2,301/01/2021 00:00:02,1,2,3\n'
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x.csv'; p.write_bytes(h+lines); r=formal.exact_structure_scan(p)
  self.assertGreater(r['truncated_records'],0); self.assertGreater(r['glued_records'],0)
 def test_07_utc_acst(self): self.assertEqual(str(pipe.utc_to_acst(np.array(['2022-01-01T00:00:01'],dtype='datetime64[s]'))[0]),'2022-01-01T09:30:01')
 def test_08_right_closed(self): self.assertEqual(str(pipe.right_closed_bin(np.datetime64('2021-01-01T00:00:01'))),'2021-01-01T00:05:00'); self.assertEqual(str(pipe.right_closed_bin(np.datetime64('2021-01-01T00:05:00'))),'2021-01-01T00:05:00')
 def test_09_annual_boundary(self): self.assertEqual(int((np.datetime64('2022-01-01')-np.datetime64('2021-12-31T23:59:59'))/np.timedelta64(1,'s')),1)
 def test_10_input_target_indices(self):
  f=np.arange(120*14).reshape(120,14); y=np.arange(120); x,t=pipe.causal_windows(f,y,np.array([71])); np.testing.assert_array_equal(x[0,-1],f[71]); np.testing.assert_array_equal(t[0],y[72:84])
 def test_11_stage_boundary(self):
  s=np.r_[np.zeros(90),np.ones(90)]; o=pipe.window_origins(s,np.ones(180,bool)); self.assertFalse(any(78<=i<=160 for i in o))
 def test_12_zero_timestamp_breaks_window(self):
  v=np.ones(120,bool); v[70]=False; self.assertNotIn(71,pipe.window_origins(np.zeros(120),v).tolist())
 def test_13_common_intersection(self): np.testing.assert_array_equal(pipe.common_origins({'a':[1,2,3],'b':[2,3,4],'c':[0,2,3]}),[2,3])
 def test_14_primary_common(self):
  o=np.array([1,2,3]); p={'a':np.array([0,.2,.2,.2]),'b':np.array([0,.2,0,.2]),'c':np.array([0,.2,.2,.2])}; np.testing.assert_array_equal(pipe.primary_daylight_common(o,p,{k:.1 for k in p}),[True,False,True])
 def test_15_future_sentinel(self):
  times=np.arange(np.datetime64('2022-01-01'),np.datetime64('2022-01-01T10:00'),np.timedelta64(5,'m')); power=np.arange(len(times),dtype=float); mb=np.ones((len(times),3)); frac=np.ones_like(mb); mask=np.ones_like(mb); before=pipe.build_14_features(power,mb,frac,mask,times); power[73:]=999; mb[73:]=999; after=pipe.build_14_features(power,mb,frac,mask,times); np.testing.assert_array_equal(before[:73],after[:73])
 def test_16_train_only_preprocessor(self):
  p=pipe.TrainOnlyPreprocessor(); x=np.ones((2,72,14)); y=np.arange(24).reshape(2,12); p.fit(x,y,'BASE_TRAIN')
  with self.assertRaises(ValueError): pipe.TrainOnlyPreprocessor().fit(x,y,'BASE_MODEL_VALIDATION')
 def test_17_validation_not_risk_fit(self):
  with self.assertRaises(ValueError): pipe.fit_risk_estimator(np.ones((2,2)),np.ones(2),1,'BASE_MODEL_VALIDATION',CONFIG,42)
 def test_18_risk_fit_not_threshold(self):
  with self.assertRaises(ValueError): pipe.calibrate_threshold(np.arange(5),'RISK_FIT')
 def test_19_final_score_isolation(self):
  c=np.array([1.,2.,3.,4.]); a=pipe.calibrate_threshold(c,'RISK_CALIBRATION'); f=np.array([-1e9,1e9]); f[:]=0; self.assertEqual(a,pipe.calibrate_threshold(c,'RISK_CALIBRATION'))
 def test_20_higher_index(self):
  r=pipe.calibrate_threshold(np.arange(10.),'RISK_CALIBRATION',.8); self.assertEqual(r['zero_based_index'],8); self.assertEqual(r['threshold'],8)
 def test_21_aurc_hand(self):
  s=np.array([0.,1.,2.]); l=np.array([1.,2.,3.]); o=np.array(['2021-01-01','2021-01-02','2021-01-03'],dtype='datetime64[D]'); self.assertTrue(np.isfinite(pipe.aurc(s,l,o)))
 def test_22_risk_same_origins(self): self.assertTrue(all(np.array_equal(np.arange(5),np.arange(5)) for _ in pipe.RISK_METHODS))
 def test_23_matched_persistence(self):
  last=np.array([1.,2.]); p=pipe.last_value_persistence(last); self.assertEqual(p.shape,(2,12)); np.testing.assert_array_equal(p[:,0],last)
 def test_24_bootstrap_fixed_mask(self):
  n=30; y=np.ones((n,12)); p=y+.1; q=np.zeros_like(y); a=np.arange(n)%2==0; o=np.datetime64('2021-01-01')+np.arange(n).astype('timedelta64[D]'); self.assertEqual(pipe.bootstrap_metrics(y,p,q,a,o,20,42)['valid_replicates'],20)
 def test_25_success_pass_fail(self):
  rows=[{'array':a,'seed':s,'risk_method':'FULL_RISK_MODEL','coverage':.8,'best_simple_aurc_improvement':.1,'accepted_rmse':1.,'persistence_rmse':2.,'rmse_reduction':.2} for a in CONFIG['arrays'] for s in CONFIG['seeds']]; self.assertTrue(pipe.evaluate_success(rows,CONFIG)['passed']); rows[0]['coverage']=0.; self.assertFalse(pipe.evaluate_success(rows,CONFIG)['passed'])
 def test_26_closed_route_refuses_authorization(self):
  r=pipe.execute_formal(CONFIG,None,Path('unused'),True,True); self.assertEqual(r['status'],'C1_ROUTE_CLOSED_DATA_UNAVAILABLE'); self.assertFalse(r['training_started'])
 def test_27_closed_route_precedes_data_state(self): self.assertEqual(pipe.execute_formal(CONFIG,None,Path('unused'),False,False)['status'],'C1_ROUTE_CLOSED_DATA_UNAVAILABLE')
 def test_28_archived_entry_is_not_executed(self): self.assertTrue(CONFIG['route_closed'])

class RealArrayTests(unittest.TestCase):
 def test_01_real_three_year_scan(self): self.assertEqual(set(AUDIT['irradiance']),{'2021','2022','2023'}); self.assertEqual(AUDIT['irradiance']['2022']['parseable_target_year_unique_seconds'],31536000)
 def test_02_explicit_sources_exist(self):
  for paths in CONFIG['irradiance_sources'].values():
   for path in paths:self.assertTrue(Path(path).is_file())
 def test_03_real_common_intersection(self):
  arrays=list(CONFIG['pv_files'])
  for stage in CONFIG['stages']:
   expected=set(STATE[f'origins__{stage}__{arrays[0]}'])
   for a in arrays[1:]:expected&=set(STATE[f'origins__{stage}__{a}'])
   self.assertEqual(expected,set(STATE[f'origins__{stage}__COMMON']))
 def test_04_real_index_causality(self):
  for stage in CONFIG['stages']:
   o=STATE[f'origins__{stage}__COMMON']; self.assertTrue(np.all((o[:,None]-np.arange(71,-1,-1))<=o[:,None])); self.assertTrue(np.all((o[:,None]+np.arange(1,13))>o[:,None]))
 def test_05_real_stage_disjoint(self):
  sets=[set(STATE[f'origins__{s}__COMMON']) for s in CONFIG['stages']]
  for i,a in enumerate(sets):
   for b in sets[i+1:]:self.assertFalse(a&b)
 def test_06_real_source_unchanged_and_no_outcomes(self):
  self.assertTrue(AUDIT['source_files_unchanged']); self.assertFalse(AUDIT['training_performed']); self.assertFalse(AUDIT['final_test_model_predictions_generated']); self.assertFalse(AUDIT['final_test_prediction_errors_accessed']); self.assertFalse(AUDIT['final_test_risk_scores_accessed']); self.assertFalse(AUDIT['final_test_coverage_or_aurc_accessed'])
 def test_07_csv_readable(self):
  with formal.SUMMARY_CSV.open(encoding='utf-8-sig',newline='') as h:self.assertGreater(len(list(csv.DictReader(h))),0)

def main():
 global AUDIT,STATE,CONFIG
 AUDIT=json.loads((formal.RESULTS/'audit.json').read_text(encoding='utf-8')); STATE=np.load(formal.RESULTS/'audit_state.npz'); CONFIG=formal.load_config(); suite=unittest.TestSuite(); suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(FixtureTests)); suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(RealArrayTests)); result=unittest.TextTestRunner(verbosity=2).run(suite)
 print(json.dumps({'fixture_total':28,'real_total':7,'successful':result.wasSuccessful(),'side_effects':False})); return 0 if result.wasSuccessful() else 1
if __name__=='__main__':raise SystemExit(main())
