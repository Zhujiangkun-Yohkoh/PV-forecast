import unittest,torch
from run_experiment import JointEarlyFusionModernTCN,SharedPrivateModernTCN,masked_loss
class T(unittest.TestCase):
 def test_shapes_gradients_and_masks(self):
  w=torch.randn(2,72,6);p=torch.randn(2,3,72,3);y=torch.randn(2,3,144);m=torch.ones_like(y,dtype=torch.bool);m[:,1]=False
  for C in (JointEarlyFusionModernTCN,SharedPrivateModernTCN):
   q=C();o=q(w,p);self.assertEqual(tuple(o.shape),(2,3,144));self.assertTrue(torch.isfinite(masked_loss(o,y,m)));z=masked_loss(o,y,torch.ones_like(m));z.backward();self.assertTrue(all(h.weight.grad is not None and h.weight.grad.abs().sum()>0 for h in q.heads))
 def test_no_input_mean_and_private_interface(self):
  self.assertEqual(JointEarlyFusionModernTCN().backbone.n[0].in_channels,15);self.assertEqual(SharedPrivateModernTCN().shared.n[0].in_channels,6);self.assertEqual(SharedPrivateModernTCN().private[0].n[0].in_channels,3)
if __name__=='__main__':unittest.main()
