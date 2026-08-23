import inspect,unittest,torch
from run_experiment import JointEarlyFusionModernTCN,SharedPrivateModernTCN,masked_loss
class T(unittest.TestCase):
 def test_shapes_gradients_and_masks(self):
  w=torch.randn(2,72,6);p=torch.randn(2,3,72,3);y=torch.randn(2,3,144);m=torch.ones_like(y,dtype=torch.bool);m[:,1]=False
  for C in (JointEarlyFusionModernTCN,SharedPrivateModernTCN):
   q=C();o=q(w,p);self.assertEqual(tuple(o.shape),(2,3,144));self.assertTrue(torch.isfinite(masked_loss(o,y,m)));z=masked_loss(o,y,torch.ones_like(m));z.backward();self.assertTrue(all(h.weight.grad is not None and h.weight.grad.abs().sum()>0 for h in q.heads))
 def test_no_input_mean_and_private_interface(self):
  self.assertEqual(JointEarlyFusionModernTCN().backbone.n[0].in_channels,15);self.assertEqual(SharedPrivateModernTCN().shared.n[0].in_channels,6);self.assertEqual(SharedPrivateModernTCN().private[0].n[0].in_channels,3)
 def test_single_technology_gradient_routing(self):
  q=SharedPrivateModernTCN();w=torch.randn(2,72,6);p=torch.randn(2,3,72,3);q(w,p)[:,0].sum().backward()
  self.assertTrue(any(x.grad is not None and x.grad.abs().sum()>0 for x in q.shared.parameters()))
  self.assertTrue(any(x.grad is not None and x.grad.abs().sum()>0 for x in q.private[0].parameters()))
  self.assertTrue(all(x.grad is None or x.grad.abs().sum()==0 for branch in q.private[1:] for x in branch.parameters()))
 def test_train_signature_has_no_test_loader(self):
  from run_experiment import train
  names=set(inspect.signature(train).parameters)
  self.assertFalse({'test','test_loader','te'}.intersection(names))
if __name__=='__main__':unittest.main()
