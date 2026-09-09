"""Diagnose batch/backend numerical replay, never relax scoring from model errors."""
from frozen_evaluation import *
if __name__=='__main__':
 c=json.loads(Path(sys.argv[1]).read_text(encoding='utf8'));site='Sanyo';proc=load(Path(c['new_results'])/site/'ridge_preprocessor.pkl');proc.data_path=Path(c['alice_raw'][site]);proc.load();part=proc.split_raw('test');x=transform_alice(proc,part);f=next(Path(c['alice_results']).glob('Depthwise*Sanyo*42/test_H144.npz'));z=read(f);o=pd.to_datetime(z['forecast_origin']);pos=part.index.get_indexer(o);state=torch.load(f.parent/'best_validation.pt',map_location='cpu',weights_only=False);b.set_seed(42);model=b.make_model('Depthwise convolutional TCN',17,proc.cfg);model.load_state_dict(state['state_dict']);rows=[]
 proc.transform();original=proc.evaluation_windows['test'];print('Original window equality',np.array_equal(original.x,x[pos[:,None]+np.arange(-71,1)]),original.x.strides,flush=True)
 for device,tf32 in [('cuda',True),('cuda',False),('cpu',False)]:
  torch.backends.cudnn.allow_tf32=tf32;model.to(device);scaled,_=b.predict_scaled(model,x[pos[:,None]+np.arange(-71,1)],256,torch.device(device));p=b.inverse_target(proc,scaled);e=abs(p-z['predictions']);row=dict(device=device,cudnn_tf32=tf32,max_abs=float(e.max()),median=float(np.median(e)),passed_frozen_tolerance=bool(np.allclose(p,z['predictions'],rtol=2e-5,atol=2e-5)));rows.append(row);print(row,flush=True)
 savejson(OUT/'REPLAY_BACKEND_DIAGNOSTIC.json',rows)
 torch.backends.cudnn.allow_tf32=True;model.cuda();scaled,_=b.predict_scaled(model,original.x,256,torch.device('cuda'));p=b.inverse_target(proc,scaled);print('Exact original window-builder max',float(abs(p-z['predictions']).max()),np.allclose(p,z['predictions'],rtol=2e-5,atol=2e-5),flush=True)
 e=abs(p-z['predictions']);bad=(~np.isclose(p,z['predictions'],rtol=2e-5,atol=2e-5)).any(1);print('Affected origins',bad.sum(),'of',len(pos),'first',o[bad][:8],flush=True)
 for matmul in [True,False]:
  torch.backends.cuda.matmul.allow_tf32=matmul
  for batch in [128,512]:
   scaled,_=b.predict_scaled(model,original.x,batch,torch.device('cuda'));p=b.inverse_target(proc,scaled);print('matmul',matmul,'batch',batch,'max',float(abs(p-z['predictions']).max()),'pass',np.allclose(p,z['predictions'],rtol=2e-5,atol=2e-5),flush=True)
