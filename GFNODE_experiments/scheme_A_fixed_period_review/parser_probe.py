"""Read-only test of CSV parser precision as a replay discrepancy source."""
from frozen_evaluation import *
if __name__=='__main__':
 c=json.loads(Path(sys.argv[1]).read_text(encoding='utf8'));site='Sanyo';f=next(Path(c['alice_results']).glob('Depthwise*Sanyo*42/test_H144.npz'));z=read(f);state=torch.load(f.parent/'best_validation.pt',weights_only=False,map_location='cpu');records=[];reader=pd.read_csv;b.set_seed(42)
 for precision in ['high','legacy','round_trip']:
  proc=load(Path(c['new_results'])/site/'ridge_preprocessor.pkl');proc.data_path=Path(c['alice_raw'][site])
  with patch.object(pd,'read_csv',lambda *a,**k:reader(*a,**(k|{'float_precision':precision}))):proc.load()
  part=proc.split_raw('test');x=transform_alice(proc,part);pos=part.index.get_indexer(pd.to_datetime(z['forecast_origin']));model=b.make_model('Depthwise convolutional TCN',17,proc.cfg).cuda();model.load_state_dict(state['state_dict']);scaled,_=b.predict_scaled(model,x[pos[:,None]+np.arange(-71,1)],256,torch.device('cuda'));p=b.inverse_target(proc,scaled);r=dict(parser=precision,max_abs=float(abs(p-z['predictions']).max()),passed=bool(np.allclose(p,z['predictions'],rtol=2e-5,atol=2e-5)));records.append(r);print(r,flush=True)
 savejson(OUT/'CSV_PARSER_REPLAY_DIAGNOSTIC.json',records)
