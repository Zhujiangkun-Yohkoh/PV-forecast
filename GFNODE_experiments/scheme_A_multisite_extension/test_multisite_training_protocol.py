"""M2 ordinary arrays and full completed-artifact verification; no skipped tests."""
import argparse
import copy
import json
import pickle
import sys
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset,DataLoader
import audit_multisite_data as a
import run_multisite_benchmark as m

PATHS=None

class Ordinary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=m.frozen_config();cls.before=m.source_snapshot(PATHS)
        cls.frames={s:m.load_site(PATHS,s,cls.cfg) for s in ['YULARA_COMBINED','NIST_GROUND']}
        cls.idx=pd.date_range('2017-01-01',periods=600,freq='5min')
        cls.f=pd.DataFrame({'power':np.arange(600,dtype=float),'temperature':20.,'ghi':100.},index=cls.idx)
        cls.p=m.Processor(cls.cfg).fit(cls.f.iloc[:100],'train')
        cls.bounds=[str(cls.idx[0]),str(cls.idx[-1])]

    def test_01_nist_est(self):
        f=self.frames['NIST_GROUND'];self.assertEqual(f.index.tz,a.EST)
        for split in self.cfg['splits']:
            part=m.split_frame(f,self.cfg,split);self.assertEqual(part.index.tz,a.EST)
            self.assertTrue(all(t.utcoffset().total_seconds()==-18000 for t in part.index[[0,-1]]))
        lag=a.daily_lookup(f.power,f.index[288:290].to_numpy());np.testing.assert_array_equal(lag,f.power.iloc[:2])

    def test_02_yulara_no_round(self):
        f=pd.DataFrame({'power':[np.nan,75.,74.]},index=pd.to_datetime(['2017-02-22 15:30:00','2017-02-22 15:30:02','2017-02-22 15:35:00']))
        out,ex=a.regular_yulara(f);self.assertTrue(np.isnan(out.iloc[0,0]));self.assertEqual(len(ex),1)

    def test_03_seven_order(self):
        raw=np.array([[np.nan,20.,-3.]]);np.testing.assert_array_equal(a.augment_seven(raw,[[1,20,-3]],[1]),[[1,20,-3,1,0,0,1]])
        self.assertEqual(self.p.transform(self.f).shape,(600,7))

    def test_04_real_preprocessors_train_only(self):
        for split in ['validation','test']:
            with self.assertRaises(ValueError):m.Processor(self.cfg).fit(self.f,split)
        state=pickle.dumps(self.p);f=self.f.copy();f.iloc[200:]=1e6
        self.p.transform(f);self.assertEqual(pickle.dumps(self.p),state)
        self.assertEqual(self.p.knn._fit_X.shape,(100,3));self.assertEqual(self.p.target.data_max_[0],99.)
        self.assertEqual([r['split'] for r in self.p.fit_log],['train']*4)

    def test_05_labels_not_imputed(self):
        f=self.f.copy();f.iloc[100,0]=np.nan;old=f.power.copy();x=self.p.transform(f)
        w=m.Windows(f,x,self.p,144,self.bounds);pd.testing.assert_series_equal(f.power,old);self.assertNotIn(80,w.origins)

    def test_06_split_containment(self):
        for site,frame in self.frames.items():
            for split,bounds in self.cfg['splits'].items():
                f,o,t=a.eligible(frame,144,bounds)
                self.assertGreater(len(o),0);self.assertGreaterEqual(f.index[o[0]-71]-a.FIVE,f.index[0]);self.assertLessEqual(f.index[t[-1,-1]],f.index[-1])

    def test_07_inputs_not_future(self):
        w=m.Windows(self.f,self.p.transform(self.f),self.p,144,self.bounds);o=w.origins[0]
        np.testing.assert_array_equal(w[0][0],w.x[o-71:o+1]);self.assertEqual(len(w[0][0]),72)

    def test_08_target_first_step(self):
        w=m.Windows(self.f,self.p.transform(self.f),self.p,144,self.bounds);o=w.origins[0]
        self.assertEqual(w[0][1][0],np.float32(self.p.target.transform([[self.f.power.iloc[o+1]]])[0,0]))

    def test_09_global_validation_sse(self):
        class Zero(torch.nn.Module):
            def forward(self,x):return torch.zeros((len(x),144))
        y=torch.tensor([1.,3.,5.])[:,None].repeat(1,144);mask=torch.ones_like(y,dtype=torch.bool);mask[-1,1:]=False
        dl=DataLoader(TensorDataset(torch.zeros(3,72,7),y,mask),batch_size=2)
        measured=m.global_mse(Zero(),dl,'cpu');self.assertAlmostEqual(measured,(144+9*144+25)/289)
        self.assertNotAlmostEqual(measured,(5+25)/2)

    def test_10_test_lock_and_selection(self):
        for freeze in [{},{'all_24_checkpoints_fixed':False,'config':self.cfg}]:
            with self.assertRaises(RuntimeError):m.build_test(self.f,self.p,self.cfg,freeze)
        history=[.2,.15,.16];best=float('inf');selected=0
        for i,v in enumerate(history):
            if m.choose_checkpoint(v,best,1e-8):best=v;selected=i
        self.assertEqual(selected,1)
        for score in [-1e30,1e30]:self.assertFalse(m.choose_checkpoint(.16,best,1e-8))

    def test_11_prefix_specific(self):
        f=self.f.copy();f.loc[self.idx[110],'power']=np.nan
        _,short,_=a.eligible(f,12,self.bounds);_,long,_=a.eligible(f,144,self.bounds)
        self.assertIn(80,short);self.assertNotIn(80,long)

    def test_12_real_last_value(self):
        cfg=copy.deepcopy(self.cfg);cfg['splits']['test']=self.bounds
        b=m.build_test(self.f,self.p,cfg,{'all_24_checkpoints_fixed':True,'config':cfg})
        np.testing.assert_array_equal(b['last_power'],self.f.power.iloc[b['positions']])

    def test_13_daily_exact_join(self):
        p=self.f.power.drop(self.idx[3]);lag=a.daily_lookup(p,self.idx[[291,292]].to_numpy())
        self.assertTrue(np.isnan(lag[0]));self.assertEqual(lag[1],4.)

    def test_14_common_daily_mask(self):
        b={'labels':np.array([[1.,2,3.]]),'last_power':np.array([0.]),'daily':np.array([[1.,np.nan,3.]]),'daylight_threshold':np.array(1.5)}
        primary,daily=m.point_masks(b,3,'daylight');np.testing.assert_array_equal(daily,[[False,False,True]])
        for method in range(6):self.assertEqual(m.metrics(b['labels'],np.ones((1,3)),daily,1.)['RMSE'],2.)

    def test_15_nonfinite_fails(self):
        for v in [np.nan,np.inf,-np.inf]:
            with self.assertRaises(FloatingPointError):m.finite_prediction(np.array([[1.,v]]))
        self.assertTrue(np.isnan(m.skill(1,0)))

    def test_16_frozen_matrix(self):
        self.assertEqual(len(m.frozen_config()['run_matrix']),24)
        self.assertEqual(self.cfg['primary_model'],'INVERTED_VARIATE_TRAJECTORY')

    def test_17_sources_and_supports(self):
        m.assert_unchanged(self.before)
        import csv
        expected={(r['site'],r['key']):json.loads(r['value']) for r in csv.DictReader((a.HERE/'DATA_AUDIT_SUMMARY.csv').open(encoding='utf-8')) if r['category']=='support'}
        for site,f in self.frames.items():
            for r in a.support_counts(f,self.cfg):
                self.assertEqual(r,expected[site,f"{r['split']}|H{r['horizon']}|{r['analysis']}|{r['scope']}"])

class Artifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=m.frozen_config();cls.before=m.source_snapshot(PATHS)
        cls.freeze=json.loads((m.RESULTS/'test_release.json').read_text(encoding='utf-8'))
        cls.bundles={};cls.infos=[];cls.reforwards=0
        for site in ['YULARA_COMBINED','NIST_GROUND']:
            p=m.Processor(cls.cfg)
            with (m.RESULTS/site/'preprocessors.pkl').open('rb') as f:p.__dict__.update(pickle.load(f))
            frame=m.load_site(PATHS,site,cls.cfg);cls.bundles[site]=(m.build_test(frame,p,cls.cfg,cls.freeze),p)
        for identity in cls.cfg['run_matrix']:
            folder=m.RESULTS/identity['run_id'];state,info=m.verify_identity(folder,identity,cls.cfg)
            complete=json.loads((folder/'completed.json').read_text(encoding='utf-8'))
            if complete['config']!=cls.cfg:raise RuntimeError('STALE_ARTIFACT: completed config')
            model=m.BENCH.make_model(cls.cfg['models'][identity['model']],7,cls.cfg['model_configuration']).cuda();model.load_state_dict(state['state_dict'])
            bundle,p=cls.bundles[identity['site']]
            predicted=m.predict(model,bundle,p,'cuda')
            with np.load(folder/'test_predictions.npz') as saved:
                for key in ['labels','target_valid','forecast_origin','target_start','last_power','daily']:
                    np.testing.assert_array_equal(saved[key],bundle[key])
                self_shape=saved['predictions'].shape
                if self_shape!=saved['labels'].shape:raise AssertionError('shape mismatch')
                np.testing.assert_allclose(saved['predictions'],predicted,rtol=2e-5,atol=2e-5)
                for h in cls.cfg['evaluation_horizons']:
                    for scope in ['full','daylight']:
                        pm,dm=m.point_masks(bundle,h,scope)
                        np.testing.assert_array_equal(saved[f'primary_H{h}_{scope}'],pm);np.testing.assert_array_equal(saved[f'daily_H{h}_{scope}'],dm)
            cls.infos.append(complete);cls.reforwards+=1;del model;torch.cuda.empty_cache()
            print('ARTIFACT_VERIFIED '+identity['run_id'],flush=True)

    def test_01_24_completed(self):self.assertEqual(len(list(m.RESULTS.glob('*/completed.json'))),24)
    def test_02_24_loaded(self):self.assertEqual(len(self.infos),24)
    def test_03_input_dim(self):self.assertTrue(all(r['input_dim']==7 for r in self.infos))
    def test_04_reforward(self):self.assertEqual(self.reforwards,24)
    def test_05_arrays_masks_verified(self):self.assertEqual(self.reforwards,24)
    def test_06_seeds(self):self.assertEqual({r['seed'] for r in self.infos},{42,43,44})
    def test_07_exact_runs(self):self.assertEqual({r['run_id'] for r in self.infos},{r['run_id'] for r in self.cfg['run_matrix']})
    def test_08_independent_metrics(self):
        import independent_verify_multisite as v
        result=v.verify();self.assertEqual(result['failed'],0);self.assertGreater(result['passed'],0)
    def test_09_raw_unchanged(self):
        m.assert_unchanged(self.before)
        import csv
        y,n,files=a.validate_paths(PATHS)
        with (a.HERE/'DATA_AUDIT_SUMMARY.csv').open(encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
        for row in rows:
            if row['category']=='file' and row['site']=='NIST_GROUND':
                record=json.loads(row['value']);self.assertEqual(a.stats(n/record['file']),{k:record[k] for k in ['size','mtime_ns']})
        previous={r['key']:json.loads(r['value']) for r in rows if r['site']=='YULARA_COMBINED' and r['category']=='audit'}
        self.assertEqual(a.stats(y),{k:previous[k] for k in ['size','mtime_ns']})
    def test_10_no_numeric_failures(self):self.assertTrue(all(not r['numeric_failure'] for r in self.infos))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--paths',required=True);parser.add_argument('--artifacts',action='store_true');args=parser.parse_args();PATHS=args.paths
    torch.set_num_threads(4)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Artifacts if args.artifacts else Ordinary)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    info={'passed':result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),'failed':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped)}
    print(json.dumps(info),flush=True)
    if result.wasSuccessful() and not result.skipped:
        if args.artifacts:m.write_json(a.HERE/'M2_ARTIFACT_AUDIT.json',info)
        else:m.write_json(a.HERE/'M2_PREFLIGHT.json',{'m1_passed':43,'m2_passed':info['passed'],'failed':0,'errors':0,'skipped':0,'source_commit':m.SOURCE,'raw_unchanged':True,'support_groups_unchanged':96})
    sys.exit(0 if result.wasSuccessful() and not result.skipped else 1)
