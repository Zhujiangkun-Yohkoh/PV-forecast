"""Executable arrays + real-source checks; no skipped tests or fitted estimators."""
import argparse
import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import audit_multisite_data as a

PATHS=None

class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=a.config()
        cls.result=a.audit(PATHS)
        cls.forward=a.forward_checks()
        cls.index=pd.date_range('2017-01-01',periods=600,freq='5min')
        cls.frame=pd.DataFrame({'power':np.arange(600,dtype=float),'temperature':20.,'ghi':100.},index=cls.index)
        cls.bounds=['2017-01-01 00:00:00',str(cls.index[-1])]

    def test_01_fixed_est_no_dst(self):
        idx=a.fixed_est(['2017-01-01 12:00:00-05:00','2017-07-01 12:00:00-05:00'])
        self.assertEqual([t.utcoffset().total_seconds() for t in idx],[-18000,-18000])
        with self.assertRaises(ValueError):a.fixed_est(['2017-07-01 12:00:00-04:00'])

    def test_02_real_files_headers(self):
        r=self.result['nist'];self.assertEqual(len(r['inventory']),365);self.assertEqual(len(r['header']),100)
        self.assertTrue(all(x['rows']>0 for x in r['inventory']))

    def test_03_exact_missing_minutes(self):
        actual=self.result['nist']['time']['missing_times']
        expected=[str(t) for t in pd.date_range('2017-10-20 23:59:00',periods=5,freq='min',tz=a.EST)]
        self.assertEqual(actual,expected)

    def test_04_five_distinct_minutes(self):
        f=pd.DataFrame({'power':[1.,2,3,4,5]},index=pd.date_range('2017-01-01',periods=5,freq='min'))
        out=a.aggregate_nist(f);self.assertEqual(out.iloc[0,0],3.);self.assertEqual(out.index[0],pd.Timestamp('2017-01-01 00:05'))
        with self.assertRaises(ValueError):a.aggregate_nist(pd.concat([f,f.iloc[:1]]).sort_index())

    def test_05_no_partial_means(self):
        f=pd.DataFrame({'power':[1.,2,np.nan,4,5],'temperature':[10.,20,30,40,50]},index=pd.date_range('2017-01-01',periods=5,freq='min'))
        out=a.aggregate_nist(f);self.assertTrue(np.isnan(out.power.iloc[0]));self.assertEqual(out.temperature.iloc[0],30)
        self.assertTrue(a.aggregate_nist(f.drop(f.index[2])).isna().all().all())

    def test_06_no_Wm2_conversion(self):
        self.assertFalse(self.cfg['nist']['convert_mV']);self.assertFalse(self.result['nist']['mV_present'])
        self.assertTrue(self.result['nist']['Wm2_direct_in_csv'])
        f=pd.DataFrame({'ghi':[100.]*5},index=pd.date_range('2017-01-01',periods=5,freq='min'))
        self.assertEqual(a.aggregate_nist(f).ghi.iloc[0],100.)

    def test_07_sentinels_not_in_target(self):
        r=self.result['nist']['fields']
        self.assertEqual(r['InvPAC_kW_Avg']['candidate_minus999'],4712)
        self.assertEqual(r['PwrMtrP_kW_Avg']['candidate_minus999'],0)
        for site in ('nist','yulara'):
            for c in self.cfg[site]['fields']:
                self.assertEqual(self.result[site]['fields'][c]['candidate_minus999'],0)
                self.assertEqual(self.result[site]['fields'][c]['candidate_minus7999'],0)

    def test_08_meter_selected(self):
        self.assertEqual(self.cfg['nist']['fields'][0],'PwrMtrP_kW_Avg')
        self.assertNotIn('InvPAC_kW_Avg',self.cfg['nist']['fields'])
        self.assertLess(self.result['nist']['fields']['PwrMtrP_kW_Avg']['quantiles']['0'] ,0)

    def test_09_no_rounding(self):
        self.assertEqual(len(self.result['yulara']['excluded']),2)
        idx=pd.to_datetime(['2017-01-01 00:00','2017-01-01 00:01','2017-01-01 00:05'])
        f=pd.DataFrame({'power':[1.,999.,2.]},index=idx);out,ex=a.regular_yulara(f)
        self.assertEqual(ex.power.iloc[0],999.);self.assertEqual(out.power.iloc[0],1.)
        self.assertEqual(out.power.iloc[1],2.)

    def test_10_gap_not_stitched(self):
        f=self.frame.iloc[:5].drop(self.index[2]);out,ex=a.regular_yulara(f)
        self.assertTrue(out.loc[self.index[3]].isna().all())
        with self.assertRaises(ValueError):a.eligible(self.frame.drop(self.index[100]),12,self.bounds)

    def test_11_real_missing_masks_and_order(self):
        raw=np.array([[1,np.nan,3],[np.inf,2,-4.]])
        filled=np.array([[1,2,3],[0,2,-4.]])
        out=a.augment_seven(raw,filled,[1,0])
        np.testing.assert_array_equal(out,[[1,2,3,0,1,0,1],[0,2,-4,1,0,0,0]])

    def test_12_labels_not_imputed(self):
        raw=np.array([1,np.nan,-.1]);out,valid=a.label_values(raw)
        np.testing.assert_array_equal(out,raw);np.testing.assert_array_equal(valid,[True,False,True])

    def test_13_all_fit_guards(self):
        class Recorder:
            def __init__(self):self.calls=[]
            def fit(self,x):self.calls.append(np.asarray(x).copy());return self
        for name in ('knn','if','feature_scaler','target_scaler'):
            stub=Recorder();a.train_only_fit(stub,[1,2],'train')
            for split in ('validation','test'):
                with self.assertRaises(ValueError):a.train_only_fit(stub,[999],split)
            self.assertEqual(len(stub.calls),1);np.testing.assert_array_equal(stub.calls[0],[1,2])

    def test_14_availability(self):
        f,o,t=a.eligible(self.frame,12,self.bounds)
        inputs=o[:,None]-np.arange(71,-1,-1)
        self.assertTrue((f.index.to_numpy()[inputs]<=f.index.to_numpy()[o,None]).all())
        self.assertTrue((f.index.to_numpy()[inputs]-np.timedelta64(5,'m')>=np.datetime64(self.bounds[0])).all())

    def test_15_targets_after_origin(self):
        f,o,t=a.eligible(self.frame,144,self.bounds)
        np.testing.assert_array_equal(f.index.to_numpy()[t[:,0]]-f.index.to_numpy()[o],np.full(len(o),np.timedelta64(5,'m')))

    def test_16_no_cross_split(self):
        bounds=[str(self.index[288]),str(self.index[-1])];f,o,t=a.eligible(self.frame,12,bounds)
        self.assertTrue((f.index.to_numpy()[t]<=np.datetime64(bounds[1])).all())
        self.assertTrue((f.index.to_numpy()[o-71]-np.timedelta64(5,'m')>=np.datetime64(bounds[0])).all())

    def test_17_daily_timestamp_join(self):
        power=self.frame.power.drop(self.index[3]);targets=self.index[[291,292]].to_numpy().reshape(1,2)
        out=a.daily_lookup(power,targets)
        self.assertTrue(np.isnan(out[0,0]));self.assertEqual(out[0,1],4)

    def test_18_all_methods_point_intersection(self):
        y=np.array([[1.,2,3]]);daily=np.array([[1.,np.nan,3]])
        methods=[np.ones((1,3)) for _ in range(5)];methods[3][0,2]=np.nan
        masks=a.shared_mask(y,daily,methods,np.ones((1,3),bool))
        self.assertEqual(len(masks),6)
        for mask in masks:np.testing.assert_array_equal(mask,[[True,False,False]])

    def test_19_daylight_train_only(self):
        cfg={'splits':{'train':[str(self.index[0]),str(self.index[99])]}}
        modified=self.frame.copy();modified.loc[self.index[100]:,'power']=1e12
        self.assertEqual(a.threshold_from_train(self.frame,cfg),.99)
        self.assertEqual(a.threshold_from_train(modified,cfg),.99)

    def test_20_models_forward_and_parameter_use(self):
        self.assertEqual(len(self.forward['checks']),4)
        for r in self.forward['checks']:
            self.assertEqual(r['outputs'],[[1,144],[2,144]])
            self.assertEqual(r['inactive_parameter_tensors'],[])
            self.assertGreater(r['parameter_tensors_checked'],0)

    def test_21_random_17_state_rejected(self):
        for r in self.forward['checks']:self.assertTrue(r['synthetic_17_state_strict_rejected'])

    def test_22_test_sentinel_invariance(self):
        cfg=copy.deepcopy(self.cfg);cfg['test_score_sentinel']=1e99
        np.testing.assert_equal(cfg['run_matrix'],self.cfg['run_matrix'])
        np.testing.assert_equal(cfg['nist'],self.cfg['nist'])
        # Actual data mutation outside Train cannot influence threshold or Train energy diagnosis.
        idx=pd.date_range('2017-08-31 23:40',periods=40,freq='min')
        f=pd.DataFrame({'PwrMtrP_kW_Avg':np.arange(40,dtype=float),'Pyra1_Wm2_Avg':np.arange(40,dtype=float)*2,
                        'PwrMtrErec_kWh_Max':np.arange(40,dtype=float),'PwrMtrEdel_kWh_Max':0.},index=idx)
        original=a.energy_diagnostics(f);changed=f.copy();changed.loc['2017-09-01':]=1e20
        self.assertEqual(original,a.energy_diagnostics(changed))
        x=pd.DataFrame({'power':[1.,2,np.nan,4,5]},index=idx[:5]);z=x.copy();z.attrs['test_score']=-1e30
        pd.testing.assert_frame_equal(a.aggregate_nist(x),a.aggregate_nist(z))

    def test_23_runtime_training_guards_no_grad_or_files(self):
        for r in self.forward['checks']:self.assertTrue(r['weights_restored_and_gradients_none'])
        for ext in ('*.pt','*.pth','*.npz'):self.assertEqual(list(a.HERE.glob(ext)),[])
        self.assertFalse(self.cfg['authorization_next_round'])

    def test_24_sources_unchanged(self):
        self.assertTrue(self.result['raw_size_mtime_unchanged'])
        y,n,files=a.validate_paths(PATHS)
        self.assertEqual(a.stats(y),{k:self.result['yulara'][k] for k in ('size','mtime_ns')})
        for f,r in zip(files,self.result['nist']['inventory']):self.assertEqual(a.stats(f),{k:r[k] for k in ('size','mtime_ns')})

    def test_25_counts_and_horizon_eligibility(self):
        for site,rows in self.result['counts'].items():
            self.assertEqual(len(rows),48)
            for r in rows:self.assertGreater(r['forecast_origin_count'],0)
        f=self.frame.copy();f.loc[self.index[110],'power']=np.nan
        _,o12,_=a.eligible(f,12,self.bounds);_,o48,_=a.eligible(f,48,self.bounds)
        self.assertIn(80,o12);self.assertNotIn(80,o48)

    def test_26_run_matrix_and_primary_model(self):
        matrix=self.cfg['run_matrix'];self.assertEqual(len(matrix),24)
        self.assertEqual(len({r['run_id'] for r in matrix}),24)
        self.assertEqual(self.cfg['primary_model'],'INVERTED_VARIATE_TRAJECTORY')
        self.assertEqual({r['seed'] for r in matrix},{42,43,44})

    def test_27_numeric_types_and_negative_retention(self):
        v,d=a.numeric(pd.Series(['','oops','inf','-inf','-999','-.1','0','3']))
        self.assertEqual((d['null'],d['non_numeric'],d['positive_inf'],d['negative_inf']),(1,1,1,1))
        self.assertEqual(d['candidate_minus999'],1);self.assertEqual(v.iloc[5],-.1)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--paths',required=True);args=parser.parse_args();PATHS=args.paths
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ProtocolTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if hasattr(ProtocolTests,'forward'):
        print('FORWARD_REPORT='+json.dumps(ProtocolTests.forward))
    print(f'M1_TESTS run={result.testsRun} failures={len(result.failures)} errors={len(result.errors)} skipped={len(result.skipped)}')
    sys.exit(0 if result.wasSuccessful() and not result.skipped else 1)
