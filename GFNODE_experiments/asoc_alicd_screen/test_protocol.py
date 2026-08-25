import inspect,sys,unittest
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import run_alicd_screen as R

class ProtocolTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.c=R.cfg();cls.d,cls.ds,cls.ld,cls.center,cls.scale=R.prepare(cls.c);R.seed_all(42);cls.m=R.ALICD(len(cls.d['base_cols']),cls.c);cls.batch=next(iter(cls.ld['train']));cls.x,cls.y,cls.y0,cls.o=cls.batch;cls.out=cls.m(cls.x[:8],cls.y0[:8])
 def test_01_delta_shape(self):self.assertEqual(tuple(self.out['delta_raw'].shape),(8,12))
 def test_02_anchor_shape(self):self.assertEqual(tuple(self.out['anchor_raw'].shape),(8,3))
 def test_03_trajectory_shape(self):self.assertEqual(tuple(self.out['trajectory'].shape),(8,12))
 def test_04_C_structure(self):self.assertEqual(tuple(self.m.C.shape),(12,12));self.assertTrue(torch.equal(self.m.C,torch.tril(torch.ones_like(self.m.C))))
 def test_05_S_indices(self):self.assertTrue(torch.equal(self.m.S.argmax(1),torch.tensor([2,5,11])))
 def test_06_A_shape(self):self.assertEqual(tuple(self.m.A.shape),(3,12))
 def test_07_A_full_rank(self):self.assertEqual(int(torch.linalg.matrix_rank(self.m.A)),3)
 def test_08_projection_stable(self):self.assertTrue(torch.isfinite(self.m.projection).all());self.assertTrue(torch.allclose(self.m.A@self.m.projection,torch.eye(3,dtype=torch.float64),atol=1e-12))
 def test_09_projected_constraint(self):
  lhs=self.out['delta_projected']@self.m.A.to(self.out['delta_projected']).T;rhs=self.out['anchor_raw']-self.y0[:8,None];self.assertTrue(torch.allclose(lhs,rhs,atol=2e-5))
 def test_10_trajectory_anchor_identity(self):self.assertTrue(torch.allclose(self.out['trajectory'][:,[2,5,11]],self.out['anchor_raw'],atol=2e-5))
 def test_11_projection_differentiable(self):self.assertTrue(self.out['trajectory'].grad_fn is not None)
 def test_12_heads_finite_gradient(self):
  self.m.zero_grad();R.losses(self.out,self.y[:8],self.y0[:8],self.c)[-1].backward();self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in list(self.m.delta_head.parameters())+list(self.m.anchor_head.parameters())))
 def test_13_change_and_low_change_loss_finite(self):
  for y in (self.y[:8],self.y0[:8,None].repeat(1,12)):
   out=self.m(self.x[:8],self.y0[:8]);self.assertTrue(all(torch.isfinite(v) for v in R.losses(out,y,self.y0[:8],self.c)))
 def test_14_y0_is_origin_power(self):
  expected=(self.d['power'][self.o.numpy()]-self.center)/self.scale;self.assertTrue(np.allclose(self.y0.numpy(),expected))
 def test_15_y0_target_scaler(self):self.assertFalse(np.allclose(self.y0.numpy(),self.d['scaled_features'][self.o.numpy(),0]));self.assertTrue(np.allclose(self.y0.numpy()*self.scale+self.center,self.d['power'][self.o.numpy()],atol=1e-6))
 def test_16_first_delta(self):self.assertTrue(torch.allclose(R.target_delta(self.y,self.y0)[:,0],self.y[:,0]-self.y0))
 def test_17_later_delta(self):self.assertTrue(torch.allclose(R.target_delta(self.y,self.y0)[:,1:],self.y[:,1:]-self.y[:,:-1]))
 def test_18_train_only_scaler(self):
  self.assertAlmostEqual(self.center,float(self.d['target_center']));self.assertAlmostEqual(self.scale,float(self.d['target_scale']));self.assertEqual(self.c['splits']['train'][1],self.c['splits']['validation'][0])
 def test_19_windows_do_not_cross_split(self):
  times=self.d['times'];
  for s in ('train','validation','test'):
   o=self.d[f'{s}_origins'];self.assertTrue(np.all(times[o]-times[o-71]==71*300_000_000_000));self.assertTrue(np.all(times[o+12]-times[o]==12*300_000_000_000))
 def test_20_inputs_end_at_origin(self):
  for s in ('train','validation','test'):
   ds=self.ds[s];i=min(10,len(ds)-1);x,_,_,o=ds[i];self.assertEqual(int(o),int(ds.origins[i]));self.assertEqual(x.shape[0],72)
 def test_21_no_test_loader_signature(self):self.assertNotIn('test',inspect.signature(R.train_model).parameters)
 def test_22_validation_checkpoint_only(self):self.assertEqual(self.c['training']['checkpoint_metric'],'validation_total_loss');self.assertEqual(self.c['loss_weights'],{'trajectory':1.0,'anchor':0.2,'increment':0.1})
 def test_23_fair_labels_timestamps_origins_masks(self):
  base=np.load(R.resolve(self.c['trajectory_results'])/'42'/'test_predictions.npz');o=self.d['test_origins'];y=np.stack([self.d['power'][i+1:i+13] for i in o]);self.assertTrue(np.array_equal(base['labels'],y));self.assertTrue(np.array_equal(base['forecast_origin_timestamp_ns'],self.d['times'][o]));self.assertEqual(base['daylight_mask'].shape,y.shape);p=R.RESULTS/'ALICD'/'42'/'test_predictions.npz';
  if p.exists():
   a=np.load(p);self.assertTrue(np.array_equal(a['labels'],base['labels']));self.assertTrue(np.array_equal(a['forecast_origin_timestamp_ns'],base['forecast_origin_timestamp_ns']));self.assertTrue(np.array_equal(a['origins'],o))
 def test_24_forward_backward_smoke(self):
  m=R.ALICD(len(self.d['base_cols']),self.c);out=m(self.x[:4],self.y0[:4]);loss=R.losses(out,self.y[:4],self.y0[:4],self.c)[-1];loss.backward();self.assertTrue(torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
 def test_25_single_batch_overfit(self):
  R.seed_all(7);m=R.ALICD(len(self.d['base_cols']),self.c);opt=torch.optim.AdamW(m.parameters(),lr=.003);x,y,y0=self.x[:16],self.y[:16],self.y0[:16];start=float(R.losses(m(x,y0),y,y0,self.c)[-1]);
  for _ in range(80):opt.zero_grad();loss=R.losses(m(x,y0),y,y0,self.c)[-1];loss.backward();opt.step()
  end=float(R.losses(m(x,y0),y,y0,self.c)[-1]);self.assertLess(end,start*.35)

if __name__=='__main__':unittest.main(verbosity=2)
