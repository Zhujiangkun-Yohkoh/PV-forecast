"""Five fixed representative GPU batch probes, no parser/CPU/version search."""
from pathlib import Path
import sys,json,argparse,platform
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'scheme_A_fixed_period_review'))
import frozen_evaluation as f
from frozen_evaluation import np,pd,torch,b,load,deny,ExitStack,patch,KNNImputer,IsolationForest,MinMaxScaler,threadpool_limits,set_config
from analysis_saved import read,R
p=argparse.ArgumentParser();p.add_argument('--paths',required=True);a=p.parse_args();c=json.loads(Path(a.paths).read_text(encoding='utf8'));torch.set_num_threads(4);set_config(working_memory=64)
cases=[('Qcells','Depthwise convolutional TCN',44),('Sanyo','Depthwise convolutional TCN',42),('Sanyo','Inverted-variate Transformer',43),('Sanyo','Inverted-variate Transformer',44),('Qcells','Inverted-variate Transformer',42)]
rows=[];identity=[];cache={}
with ExitStack() as stack,threadpool_limits(limits=4):
 for cls,name in [(KNNImputer,'fit'),(IsolationForest,'fit'),(MinMaxScaler,'fit'),(torch.Tensor,'backward'),(torch,'save')]:stack.enter_context(patch.object(cls,name,deny))
 for site,name,seed in cases:
  if site not in cache:
   proc=load(Path(c['new_results'])/site/'ridge_preprocessor.pkl');proc.data_path=Path(c['alice_raw'][site]);proc.load();part=proc.split_raw('test');x=f.transform_alice(proc,part);cache[site]=(proc,part,x)
  proc,part,x=cache[site];folder=next(z.parent for z in Path(c['alice_results']).glob('*/completed.json') if (lambda j:j['dataset']==site and j['model']==name and j['seed']==seed)(json.loads(z.read_text(encoding='utf8'))));old=read(folder/'test_H144.npz');key=('Inverted' if 'Inverted' in name else name)+str(seed);new=read(Path(c['destination'])/site/(key.replace(' ','_')+'.npz'));oo=pd.DatetimeIndex(pd.to_datetime(old['forecast_origin']));no=pd.DatetimeIndex(pd.to_datetime(new['forecast_origin']));lookup=no.get_indexer(oo);d=abs(new['predictions'][lookup]-old['predictions']);mi,mj=np.unravel_index(d.argmax(),d.shape);mixed_i=lookup[mi];oldpos=part.index.get_indexer(oo);newpos=part.index.get_indexer(no)
  state=torch.load(folder/'best_validation.pt',map_location='cpu',weights_only=False);b.set_seed(seed);model=b.make_model(name,17,proc.cfg).cuda().eval();loaded=model.load_state_dict(state['state_dict'],strict=True)
  identity.append(dict(site=site,model=name,seed=seed,checkpoint=folder.name,state_keys=list(state),strict_missing=list(loaded.missing_keys),strict_unexpected=list(loaded.unexpected_keys),eval_not_training=not model.training,input_dtype=str(x.dtype),input_channels=list(proc.feature_columns)+['missing:'+v for v in proc.feature_columns]+['isolation_forest'],imputer_fit_shape=list(proc.knn._fit_X.shape),feature_scaler_scale=proc.feature_scaler.scale_.tolist(),target_scale=proc.target_scaler.scale_.tolist(),target_min=proc.target_scaler.min_.tolist(),processor_identity='later saved Ridge Train-only processor; no separately saved historical neural object for direct comparison',original_neural_KNN_IF_scaler_state_available=False,parameter_count=sum(q.numel() for q in model.parameters())))
  plans=[('original_batch',oldpos[(mi//256)*256:(mi//256+1)*256],mi%256,mi),('mixed_batch',newpos[(mixed_i//256)*256:(mixed_i//256+1)*256],mixed_i%256,mi),('singleton',oldpos[mi:mi+1],0,mi),('original_last_incomplete',oldpos[(len(oldpos)//256)*256:],-1,len(oldpos)-1),('mixed_last_incomplete',newpos[(len(newpos)//256)*256:],None,None)]
  for mode,pos,sel,ref_i in plans:
   if not len(pos):continue
   with torch.inference_mode():scaled=model(torch.from_numpy(x[pos[:,None]+np.arange(-71,1)]).cuda()).cpu().numpy()
   pred=proc.target_scaler.inverse_transform(scaled.reshape(-1,1)).reshape(scaled.shape).astype(np.float32)
   if sel is None:
    sel=-1;timestamp=part.index[pos[sel]];expected=new['predictions'][newpos.tolist().index(pos[sel])];source='prior mixed-origin array'
   else:timestamp=oo[ref_i];expected=old['predictions'][ref_i];source='original frozen prediction'
   delta=pred[sel].astype(float)-expected.astype(float);rows.append(dict(site=site,model=name,seed=seed,mode=mode,batch_size=len(pos),origin=str(timestamp),reference=source,max_difference_kW=float(abs(delta).max()),max_difference_W=float(abs(delta).max()*1000),RMS_difference_W=float(np.sqrt(np.mean(delta*delta))*1000),passes_original_tolerance=bool(np.allclose(pred[sel],expected,rtol=2e-5,atol=2e-5)),focus_lead_min=(mj+1)*5,focus_prediction_kW=float(pred[sel,mj]),focus_reference_kW=float(expected[mj])))
  del model,state;torch.cuda.empty_cache();print('Representative batch probe',site,name,seed,'complete',flush=True)
info=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(0),cudnn_tf32=torch.backends.cudnn.allow_tf32,matmul_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,float32_matmul_precision=torch.get_float32_matmul_precision())
(R/'CURRENT_FORWARD_ENVIRONMENT.json').write_text(json.dumps(info,indent=2),encoding='utf8');(R/'REPRESENTATIVE_IDENTITY.json').write_text(json.dumps(identity,indent=2),encoding='utf8');pd.DataFrame(rows).to_csv(R/'batch_probe.csv',index=False)
