"""Descriptive new-period input exposure under saved Ridge transforms; no fit."""
from frozen_evaluation import *
if __name__=='__main__':
 c=json.loads(Path(sys.argv[1]).read_text(encoding='utf8'));patterns=[];ranges=[];set_config(working_memory=64)
 with ExitStack() as stack:
  for cls in [KNNImputer,MinMaxScaler,IsolationForest]:stack.enter_context(patch.object(cls,'fit',deny))
  for site in ['YULARA_COMBINED','NIST_GROUND']:
   f=new_external_frame(c,site);p=load(Path(c['new_results'])/site/'ridge_preprocessor.pkl');x=p.transform(f);idx=f.index;pos=np.flatnonzero((idx>=pd.Timestamp('2018-04-01',tz=idx.tz))&(idx<=pd.Timestamp('2018-06-30 23:55',tz=idx.tz)));windows=x[pos[:,None]+np.arange(-71,1)];rawmask=~np.isfinite(f[m.a.FIELDS].to_numpy(float))
   for j,name in enumerate(m.a.FIELDS):
    counts=rawmask[pos[:,None]+np.arange(-71,1),j].sum(1)
    for label,keep in [('none',counts==0),('some',(counts>0)&(counts<72)),('all72',counts==72)]:patterns.append(dict(site=site,field=name,pattern=label,origins=int(keep.sum()),candidate_origins=len(pos),fraction=float(keep.mean())))
   flat=windows.reshape(len(pos),-1).astype(float);coef=read(Path(c['expanded_results'][site])/'Ridge_coefficients.npz');standardized=(flat-coef['feature_mean'])/coef['feature_scale']
   for j,name in enumerate(m.a.config()['input_channels']):
    for lag in [0,71]:
     k=(71-lag)*7+j;v=standardized[:,k];ranges.append(dict(site=site,channel=name,lag=lag,minimum=float(v.min()),p01=float(np.quantile(v,.01)),median=float(np.median(v)),p99=float(np.quantile(v,.99)),maximum=float(v.max()),saved_train_mean=float(coef['feature_mean'][k]),saved_train_scale=float(coef['feature_scale'][k])))
 pd.DataFrame(patterns).to_csv(OUT/'fixed_2018_history_missing_patterns.csv',index=False);pd.DataFrame(ranges).to_csv(OUT/'fixed_2018_Ridge_standardized_ranges.csv',index=False);print('Saved-transform exposure recorded; no fitting')
