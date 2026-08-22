import inspect, unittest, torch
from benchmark import cfg, model, train
class TestBenchmark(unittest.TestCase):
 def test_outputs_are_h144(self):
  c=cfg(); x=torch.randn(2,72,17)
  for name in ('Discrete Candidate','iTransformer','PatchTST','ModernTCN'):
   self.assertEqual(tuple(model(name,17,c)(x).shape),(2,144))
 def test_train_has_no_test_loader(self): self.assertNotIn('test',inspect.signature(train).parameters)
if __name__=='__main__': unittest.main()
