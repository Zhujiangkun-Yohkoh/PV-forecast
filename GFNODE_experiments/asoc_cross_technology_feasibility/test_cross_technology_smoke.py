import unittest, torch
from cross_technology import SharedModernTCN
class Smoke(unittest.TestCase):
 def test_three_head_h144_backward(self):
  m=SharedModernTCN(15);x=torch.randn(2,72,15);y=torch.randn(2,3,144);o=m(x);self.assertEqual(tuple(o.shape),(2,3,144));((o-y)**2).mean().backward();self.assertTrue(any(p.grad is not None for p in m.parameters()))
if __name__=='__main__':unittest.main()
