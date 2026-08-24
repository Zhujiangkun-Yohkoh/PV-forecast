import inspect, sys, unittest
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
import run_frozen_hazard_screen as R

class ProtocolTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.c=R.cfg(); cls.d,cls.ds,cls.ld,cls.center,cls.scale,cls.threshold=R.make_data(cls.c); cls.seed=42
  ck=torch.load(R.resolve(cls.c['trajectory_results'])/'42'/'best_validation.pt',map_location='cpu',weights_only=True); cls.state=ck['state_dict']; cls.model=R.FrozenTrajectoryHazard(len(cls.d['base_cols']),cls.c,cls.state); cls.batch=next(iter(cls.ld['train'])); cls.x=cls.batch[0]; cls.old=np.load(R.artifact('TRAJECTORY_ONLY',42,cls.c))
 def test_01_checkpoint_loads(self): self.assertIn('net.0.weight',self.state)
 def test_02_backbone_frozen(self): self.assertTrue(all(not p.requires_grad for p in self.model.backbone.parameters()))
 def test_03_power_head_frozen(self): self.assertTrue(all(not p.requires_grad for p in self.model.power_head.parameters()))
 def test_04_hazard_trainable(self): self.assertTrue(all(p.requires_grad for p in self.model.hazard_head.parameters()))
 def test_05_optimizer_hazard_only(self):
  opt=torch.optim.AdamW(self.model.hazard_head.parameters()); self.assertEqual({id(p) for g in opt.param_groups for p in g['params']},{id(p) for p in self.model.hazard_head.parameters()})
 def test_06_backbone_unchanged_step(self):
  snap=R.snapshot_frozen(self.model); opt=torch.optim.AdamW(self.model.hazard_head.parameters()); _,h=self.model(self.x); loss=R.JOINT.hazard_nll(h,self.batch[3],self.batch[4]); loss.backward();opt.step();self.assertTrue(R.frozen_unchanged(self.model,snap))
 def test_07_power_head_unchanged_step(self): self.test_06_backbone_unchanged_step()
 def test_08_buffers_unchanged(self): self.assertTrue(R.frozen_unchanged(self.model,R.snapshot_frozen(self.model)))
 def test_09_backbone_eval_during_train(self): self.model.train();self.assertFalse(self.model.backbone.training)
 def test_10_hazard_train_during_train(self): self.model.train();self.assertTrue(self.model.hazard_head.training)
 def test_11_batch_power_identity(self):
  base=R.INFO.ModernTCN(len(self.d['base_cols']),self.c);base.load_state_dict(self.state);base.eval();self.model.eval();self.assertTrue(torch.equal(base(self.x),self.model(self.x)[0]))
 def test_12_complete_power_identity(self):
  for seed in self.c['seeds']:
   p=R.RESULTS/'FROZEN_TRAJECTORY_HAZARD'/str(seed)/'test_predictions.npz'; self.assertTrue(p.exists())
   old=np.load(R.artifact('TRAJECTORY_ONLY',seed,self.c)); self.assertLessEqual(float(np.max(np.abs(np.load(p)['power_predictions']-old['predictions']))),2e-6)
 def test_13_hazard_shape(self): self.assertEqual(tuple(self.model(self.x[:3])[1].shape),(3,12,3))
 def test_14_softmax_sum(self): self.assertTrue(torch.allclose(self.model(self.x[:3])[1].softmax(-1).sum(-1),torch.ones(3,12),atol=1e-6))
 def test_15_event_distribution_sum(self):
  q=self.model(self.x[:3])[1].softmax(-1);surv=torch.ones(3);total=torch.zeros(3)
  for k in range(12): total+=surv*(q[:,k,1]+q[:,k,2]);surv*=q[:,k,0]
  self.assertTrue(torch.allclose(total+surv,torch.ones(3),atol=1e-5))
 def test_16_event_batch_finite(self):
  logits=torch.randn(4,12,3,requires_grad=True);loss=R.JOINT.hazard_nll(logits,torch.tensor([0,3,8,11]),torch.tensor([1,2,1,2]));loss.backward();self.assertTrue(torch.isfinite(loss) and torch.isfinite(logits.grad).all())
 def test_17_no_event_batch_finite(self):
  logits=torch.randn(4,12,3,requires_grad=True);loss=R.JOINT.hazard_nll(logits,torch.full((4,),-1),torch.zeros(4,dtype=torch.long));loss.backward();self.assertTrue(torch.isfinite(loss) and torch.isfinite(logits.grad).all())
 def test_18_frozen_grad_none(self):
  self.model.zero_grad(set_to_none=True);_,h=self.model(self.x[:4]);R.JOINT.hazard_nll(h,self.batch[3][:4],self.batch[4][:4]).backward();self.assertTrue(all(p.grad is None for p in list(self.model.backbone.parameters())+list(self.model.power_head.parameters())))
 def test_19_hazard_grad_finite(self):
  self.model.zero_grad(set_to_none=True);_,h=self.model(self.x[:4]);R.JOINT.hazard_nll(h,self.batch[3][:4],self.batch[4][:4]).backward();self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.hazard_head.parameters()))
 def test_20_no_test_loader_signature(self): self.assertNotIn('test',inspect.signature(R.train_hazard).parameters)
 def test_21_validation_checkpoint_metric(self): self.assertEqual(self.c['training']['checkpoint_metric'],'validation_first_event_nll')
 def test_22_fair_artifacts(self):
  for name in ('STEP_MULTITASK','STANDARD_ONSET_HAZARD','POWER_ANCHORED_HAZARD'):
   a=np.load(R.artifact(name,42,self.c));self.assertTrue(np.array_equal(a['labels'],self.old['labels']));self.assertTrue(np.array_equal(a['forecast_origin_timestamp_ns'],self.old['forecast_origin_timestamp_ns']))
 def test_23_train_threshold_fixed(self): self.assertAlmostEqual(self.threshold,float(self.d['ramp_threshold']))
 def test_24_inputs_end_at_origin(self):
  times=self.d['times'];
  for s in ('train','validation','test'):
   o=self.d[f'{s}_origins'];self.assertTrue(np.all(times[o]-times[o-71]==71*300*10**9))

if __name__=='__main__': unittest.main(verbosity=2)
