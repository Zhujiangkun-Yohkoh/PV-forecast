"""Build four vector figures from immutable Stage-2 evidence."""
from pathlib import Path
import numpy as np
import pandas as pd
from reportlab.lib.colors import Color, HexColor, black, white
from reportlab.pdfgen.canvas import Canvas

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
EVIDENCE=ROOT/"GFNODE_experiments"/"clean_deterministic_manuscript_stage2_evidence"
OUT=HERE/"figures"; OUT.mkdir(parents=True,exist_ok=True)
MODELS=["PERSISTENCE_LAST","Discrete Candidate","iTransformer","PatchTST","ModernTCN"]
DISPLAY={"PERSISTENCE_LAST":"Last-value persistence","Discrete Candidate":"Discrete candidate","iTransformer":"iTransformer","PatchTST":"PatchTST","ModernTCN":"ModernTCN"}
COLORS={"PERSISTENCE_LAST":"#000000","Discrete Candidate":"#E69F00","iTransformer":"#56B4E9","PatchTST":"#009E73","ModernTCN":"#CC79A7"}

def text(c,x,y,s,size=8,anchor="start",bold=False,color=black):
    c.setFillColor(color); c.setFont("Helvetica-Bold" if bold else "Helvetica",size)
    (c.drawCentredString if anchor=="middle" else c.drawRightString if anchor=="end" else c.drawString)(x,y,s)

def marker(c,x,y,model,r=3):
    c.setFillColor(HexColor(COLORS[model])); c.setStrokeColor(black)
    if model=="PatchTST":
        p=c.beginPath(); p.moveTo(x,y+r); p.lineTo(x-r,y-r); p.lineTo(x+r,y-r); p.close(); c.drawPath(p,fill=1,stroke=1)
    elif model=="ModernTCN":
        p=c.beginPath(); p.moveTo(x,y+r); p.lineTo(x-r,y); p.lineTo(x,y-r); p.lineTo(x+r,y); p.close(); c.drawPath(p,fill=1,stroke=1)
    else: c.circle(x,y,r,fill=1,stroke=1)

def load():
    m=pd.read_csv(EVIDENCE/"FINAL_METRICS_LONG.csv"); e=pd.read_csv(EVIDENCE/"FINAL_EFFICIENCY.csv")
    assert len(m)==3696 and set(MODELS).issubset(set(m.model)); return m,e

def figure_protocol():
    c=Canvas(str(OUT/"fig1_leakage_free_protocol.pdf"),pagesize=(510,225))
    boxes=[(15,166,110,"Raw 5-min timeline\nnight retained","#E5E5E5"),(145,166,105,"Train\nfit preprocessing","#D9EAD3"),(275,166,105,"Validation\nearly stopping only","#FFF2CC"),(405,166,90,"Test\nuntouched evaluation","#CFE2F3")]
    for x,y,w,label,color in boxes:
        c.setFillColor(HexColor(color)); c.setStrokeColor(black); c.rect(x,y,w,38,fill=1,stroke=1)
        for i,line in enumerate(label.split("\n")): text(c,x+w/2,y+23-i*11,line,7.5,"middle",i==0)
    for x0,x1 in [(125,145),(250,275),(380,405)]: c.line(x0,185,x1,185); c.line(x1-4,188,x1,185); c.line(x1-4,182,x1,185)
    lower=[(48,"KNN, Isolation Forest, and scalers\nfit on Train only"),(195,"Windows formed within each split\nlookback 72; output 144"),(342,"One H144 trajectory supplies\nH12/H48/H96/H144 prefixes")]
    for x,label in lower:
        c.setFillColor(white); c.rect(x,93,120,42,fill=1,stroke=1)
        for i,line in enumerate(label.split("\n")): text(c,x+60,116-i*12,line,7,"middle")
    text(c,255,60,"No window crosses a split; Test does not select preprocessing, hyperparameters, or checkpoints.",7.5,"middle")
    text(c,255,37,"Metrics use common H144 origins and masks; daylight is an evaluation-only target mask.",7.5,"middle")
    text(c,255,212,"Leakage-free temporal evaluation protocol",9,"middle",True); c.showPage(); c.save()

def panel(c,q,x,y,w,h,title,ymax):
    left,bottom=x+32,y+24; pw,ph=w-38,h-42; c.setStrokeColor(HexColor("#BBBBBB")); c.rect(left,bottom,pw,ph,fill=0,stroke=1)
    for val in np.linspace(0,ymax,4):
        yy=bottom+ph*val/ymax; c.setStrokeColor(HexColor("#DDDDDD")); c.line(left,yy,left+pw,yy); text(c,left-4,yy-2,f"{val:.2f}",5.5,"end")
    for hs in [12,48,96,144]: text(c,left+pw*(hs-12)/132,bottom-10,str(hs),5.5,"middle")
    for model in MODELS:
        z=q[q.model==model].sort_values("horizon_steps"); assert len(z)==4; pts=[(left+pw*(r.horizon_steps-12)/132,bottom+ph*r.value/ymax) for _,r in z.iterrows()]
        c.setStrokeColor(HexColor(COLORS[model])); c.setLineWidth(1.1)
        for a,b in zip(pts,pts[1:]): c.line(a[0],a[1],b[0],b[1])
        for xx,yy in pts: marker(c,xx,yy,model,2)
    text(c,x+w/2,y+h-12,title,7,"middle",True)

def figure_curves(m):
    c=Canvas(str(OUT/"fig2_multihorizon_nrmse.pdf"),pagesize=(510,330)); q=m[(m.metric=="range_nRMSE")&m.statistic.isin(["mean","deterministic"])&m.model.isin(MODELS)]
    for col,dataset in enumerate(["Sanyo","Hanwha","Qcells"]):
        for row,(scope,name) in enumerate([("regular_full_timeline","Full timeline"),("daylight","Daylight")]): panel(c,q[(q.dataset==dataset)&(q.scope==scope)],5+col*168,157-row*145,163,140,f"{dataset} -- {name}",.65 if row else .48)
    for i,model in enumerate(MODELS):
        x=45+(i%3)*150; y=20-(i//3)*12; marker(c,x,y+2,model,2.5); text(c,x+8,y,DISPLAY[model],6.5)
    text(c,255,314,"Test Train-range nRMSE by horizon, array, and scope",9,"middle",True); text(c,255,4,"Forecast horizon (5-min steps)",6.5,"middle"); c.showPage(); c.save()

def figure_rank_heatmap(m):
    neural=["Discrete Candidate","iTransformer","PatchTST","ModernTCN"]; q=m[(m.metric=="RMSE")&(m.statistic=="mean")&m.model.isin(neural)].copy(); q["r"]=q.groupby(["dataset","horizon_steps","scope"])["value"].rank()
    labels=[]; rows=[]
    for dataset in ["Sanyo","Hanwha","Qcells"]:
        for scope,short in [("regular_full_timeline","Full"),("daylight","Day")]:
            for h in [12,48,96,144]:
                z=q[(q.dataset==dataset)&(q.scope==scope)&(q.horizon_steps==h)].set_index("model"); labels.append(f"{dataset} {short} H{h}"); rows.append([z.loc[x,"r"] for x in neural])
    c=Canvas(str(OUT/"fig3_rank_heatmap.pdf"),pagesize=(410,500)); x0,cw,ch=138,58,17.8; text(c,205,482,"RMSE rank among neural models (1 = best)",9,"middle",True)
    for j,model in enumerate(neural): text(c,x0+j*cw+cw/2,458,DISPLAY[model],6,"middle")
    for i,(label,row) in enumerate(zip(labels,rows)):
        y=445-i*ch; text(c,x0-5,y+5,label,6,"end")
        for j,val in enumerate(row):
            t=(val-1)/3; c.setFillColor(Color(.15+.55*t,.55-.35*t,.75-.45*t)); c.setStrokeColor(white); c.rect(x0+j*cw,y,cw,ch,fill=1,stroke=1); text(c,x0+j*cw+cw/2,y+5,f"{val:.0f}",7,"middle",True,white)
    c.showPage(); c.save()

def figure_pareto(m,e):
    neural=["Discrete Candidate","iTransformer","PatchTST","ModernTCN"]; q=m[(m.metric=="range_nRMSE")&(m.statistic=="mean")&m.model.isin(neural)]; macro=q.groupby("model",as_index=False).value.mean().rename(columns={"value":"error"}); p=macro.merge(e[e.model.isin(neural)].drop_duplicates("model")[["model","latency_median_ms","parameter_count"]],on="model"); persistence=m[(m.metric=="range_nRMSE")&(m.statistic=="deterministic")&(m.model=="PERSISTENCE_LAST")].value.mean()
    c=Canvas(str(OUT/"fig4_accuracy_efficiency.pdf"),pagesize=(390,280)); x0,y0,w,h=55,45,300,185; c.rect(x0,y0,w,h,fill=0,stroke=1); xmin,xmax=np.log10(.3),np.log10(40); ymin,ymax=.11,max(.31,persistence*1.05)
    py=y0+h*(persistence-ymin)/(ymax-ymin); c.setDash(4,3); c.line(x0,py,x0+w,py); c.setDash(); text(c,x0+w-3,py+4,"Last-value persistence",6,"end")
    for _,r in p.iterrows():
        xx=x0+w*(np.log10(r.latency_median_ms)-xmin)/(xmax-xmin); yy=y0+h*(r.error-ymin)/(ymax-ymin); marker(c,xx,yy,r.model,3+min(3,r.parameter_count/250000)); text(c,xx+5,yy+5,DISPLAY[r.model],6)
    for tick in [.3,1,3,10,30]: text(c,x0+w*(np.log10(tick)-xmin)/(xmax-xmin),y0-12,str(tick),6,"middle")
    for tick in [.12,.16,.20,.24,.28]: text(c,x0-6,y0+h*(tick-ymin)/(ymax-ymin)-2,f"{tick:.2f}",6,"end")
    text(c,195,257,"Accuracy--efficiency trade-off",9,"middle",True); text(c,205,20,"Median GPU latency per sample (ms, log scale)",7,"middle"); c.saveState(); c.translate(15,137); c.rotate(90); text(c,0,0,"Macro mean Train-range nRMSE",7,"middle"); c.restoreState(); text(c,205,7,"Marker area scales approximately with parameter count",6,"middle"); c.showPage(); c.save()

def main():
    m,e=load(); figure_protocol(); figure_curves(m); figure_rank_heatmap(m); figure_pareto(m,e)
    expected=["fig1_leakage_free_protocol.pdf","fig2_multihorizon_nrmse.pdf","fig3_rank_heatmap.pdf","fig4_accuracy_efficiency.pdf"]; assert sorted(p.name for p in OUT.glob("*.pdf"))==expected; print("Generated:",", ".join(expected))
if __name__=="__main__": main()
