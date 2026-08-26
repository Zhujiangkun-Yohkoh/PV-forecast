"""Regenerate four vector figures from corrected Scheme-A evidence."""
from pathlib import Path
import math
import numpy as np
import pandas as pd
from reportlab.lib.colors import Color, HexColor, black, white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EVIDENCE = ROOT / "GFNODE_experiments" / "scheme_A_submission_correction" / "corrected_metrics.csv"
OUT = HERE / "figures"
OUT.mkdir(parents=True, exist_ok=True)
pdfmetrics.registerFont(TTFont("Arial", r"C:\Windows\Fonts\arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", r"C:\Windows\Fonts\arialbd.ttf"))

MODELS = ["Last-value Persistence", "Discrete recurrent decoder", "Inverted-variate Transformer", "Joint-patch Transformer", "Depthwise convolutional TCN"]
NEURAL = MODELS[1:]
SHORT = dict(zip(MODELS, ["Last-value persistence", "Discrete recurrent", "Inverted-variate", "Joint-patch", "Depthwise TCN"]))
COLORS = dict(zip(MODELS, ["#000000", "#E69F00", "#0072B2", "#009E73", "#CC79A7"]))

def text(c, x, y, value, size=8, anchor="start", bold=False, color=black):
    c.setFillColor(color); c.setFont("Arial-Bold" if bold else "Arial", size)
    fn = c.drawCentredString if anchor == "middle" else c.drawRightString if anchor == "end" else c.drawString
    fn(x, y, str(value))

def marker(c, x, y, model, radius=3):
    c.setFillColor(HexColor(COLORS[model])); c.setStrokeColor(black)
    if model == MODELS[3]:
        p=c.beginPath(); p.moveTo(x,y+radius); p.lineTo(x-radius,y-radius); p.lineTo(x+radius,y-radius); p.close(); c.drawPath(p,fill=1,stroke=1)
    elif model == MODELS[4]:
        p=c.beginPath(); p.moveTo(x,y+radius); p.lineTo(x-radius,y); p.lineTo(x,y-radius); p.lineTo(x+radius,y); p.close(); c.drawPath(p,fill=1,stroke=1)
    else: c.circle(x,y,radius,fill=1,stroke=1)

def load():
    d=pd.read_csv(EVIDENCE)
    assert len(d)==6648 and set(MODELS).issubset(d.model)
    assert {12,48,96,144}.issubset(set(d.horizon_steps.dropna().astype(int)))
    return d

def figure_protocol():
    c=Canvas(str(OUT/"fig1_leakage_free_protocol.pdf"),pagesize=(510,238))
    boxes=[(12,180,112,"Raw 5-min timeline\nnight retained","#E5E5E5"),(142,180,105,"Train\nfit preprocessing","#D9EAD3"),(270,180,108,"Validation\nselect checkpoints","#FFF2CC"),(402,180,96,"Test\nevaluation only","#CFE2F3")]
    for x,y,w,label,color in boxes:
        c.setFillColor(HexColor(color)); c.setStrokeColor(black); c.rect(x,y,w,38,fill=1,stroke=1)
        for i,line in enumerate(label.split("\n")): text(c,x+w/2,y+23-i*11,line,7.5,"middle",i==0)
    for x0,x1 in [(124,142),(247,270),(378,402)]: c.line(x0,199,x1,199); c.line(x1-4,202,x1,199); c.line(x1-4,196,x1,199)
    lower=[(15,"KNN, Isolation Forest, and\nall scalers fit on Train only"),(137,"17 channels: 8 causal values,\n8 masks, 1 anomaly flag"),(259,"Windows built inside each split;\nlookback 72, output 144"),(381,"Horizon-specific valid origins;\nH144-common sensitivity")]
    for x,label in lower:
        c.setFillColor(white); c.rect(x,104,114,44,fill=1,stroke=1)
        for i,line in enumerate(label.split("\n")): text(c,x+57,129-i*12,line,6.8,"middle")
    text(c,255,75,"Past Active Power is causal input; future Active Power is target only.",7.5,"middle")
    text(c,255,55,"Validation global MSE = total SSE / valid targets; Test is excluded from fitting, training, and checkpoint selection.",7.1,"middle")
    text(c,255,27,"Leakage-aware and sample-matched temporal evaluation protocol",9.5,"middle",True)
    c.showPage(); c.save()

def panel(c,q,x,y,w,h,title,ymax):
    left,bottom,pw,ph=x+32,y+24,w-39,h-43; c.setStrokeColor(HexColor("#888888")); c.rect(left,bottom,pw,ph,fill=0,stroke=1)
    for val in np.linspace(0,ymax,4):
        yy=bottom+ph*val/ymax; c.setStrokeColor(HexColor("#DDDDDD")); c.line(left,yy,left+pw,yy); text(c,left-4,yy-2,f"{val:.2f}",5.5,"end")
    for hs in [12,48,96,144]: text(c,left+pw*(hs-12)/132,bottom-10,hs,5.5,"middle")
    for model in MODELS:
        z=q[q.model==model].sort_values("horizon_steps"); assert len(z)==4
        pts=[(left+pw*(r.horizon_steps-12)/132,bottom+ph*r.value/ymax) for _,r in z.iterrows()]
        c.setStrokeColor(HexColor(COLORS[model])); c.setLineWidth(1.15)
        if model==MODELS[0]: c.setDash(4,2)
        for a,b in zip(pts,pts[1:]): c.line(a[0],a[1],b[0],b[1])
        c.setDash()
        for xx,yy in pts: marker(c,xx,yy,model,2.1)
    text(c,x+w/2,y+h-12,title,7,"middle",True)

def figure_curves(d):
    q=d[(d.analysis=="primary_horizon_specific")&(d.metric=="range_nRMSE")&d.model.isin(MODELS)&d.statistic.isin(["mean","deterministic"])]
    c=Canvas(str(OUT/"fig2_multihorizon_nrmse.pdf"),pagesize=(510,342))
    for col,dataset in enumerate(["Sanyo","Hanwha","Qcells"]):
        for row,(scope,label) in enumerate([("regular_full_timeline","Full timeline"),("daylight","Daylight")]): panel(c,q[(q.dataset==dataset)&(q.scope==scope)],4+col*169,166-row*147,164,142,f"{dataset} -- {label}",.66 if row else .46)
    for i,model in enumerate(MODELS):
        x,y=28+(i%3)*162,21-(i//3)*12; marker(c,x,y+2,model,2.4); text(c,x+8,y,SHORT[model],6.3)
    text(c,255,326,"Primary Test Train-range nRMSE on horizon-specific valid origins",9.3,"middle",True); text(c,255,5,"Forecast horizon (5-min steps)",6.5,"middle")
    c.showPage(); c.save()

def figure_rank_heatmap(d):
    q=d[(d.analysis=="primary_horizon_specific")&(d.metric=="RMSE")&d.statistic.isin(["mean","deterministic"])&d.model.isin(MODELS)].copy()
    q["rank"]=q.groupby(["dataset","horizon_steps","scope"])["value"].rank(method="average")
    labels,rows=[],[]
    for dataset in ["Sanyo","Hanwha","Qcells"]:
        for scope,short in [("regular_full_timeline","Full"),("daylight","Day")]:
            for h in [12,48,96,144]:
                z=q[(q.dataset==dataset)&(q.scope==scope)&(q.horizon_steps==h)].set_index("model"); labels.append(f"{dataset} {short} H{h}"); rows.append([z.loc[m,"rank"] for m in MODELS])
    c=Canvas(str(OUT/"fig3_rank_heatmap.pdf"),pagesize=(480,510)); x0,cw,ch=132,67,18
    text(c,240,493,"Primary RMSE rank including Last-value Persistence (1 = best)",9.2,"middle",True)
    for j,model in enumerate(MODELS): text(c,x0+j*cw+cw/2,466,SHORT[model],5.6,"middle")
    for i,(label,row) in enumerate(zip(labels,rows)):
        y=451-i*ch; text(c,x0-5,y+5,label,6.1,"end")
        for j,val in enumerate(row):
            t=(val-1)/4; c.setFillColor(Color(.13+.60*t,.54-.31*t,.75-.43*t)); c.setStrokeColor(white); c.rect(x0+j*cw,y,cw,ch,fill=1,stroke=1); text(c,x0+j*cw+cw/2,y+5,f"{val:.0f}",7,"middle",True,white)
    c.showPage(); c.save()

def figure_pareto(d):
    err=d[(d.analysis=="primary_horizon_specific")&(d.metric=="range_nRMSE")&(d.statistic=="mean")&d.model.isin(NEURAL)].groupby("model",as_index=False).value.mean().rename(columns={"value":"error"})
    meta=d[d.analysis=="run_metadata"]
    lat=meta[meta.metric=="latency_mean_ms"].groupby("model",as_index=False).value.mean().rename(columns={"value":"latency"})
    par=meta[meta.metric=="parameter_count"].groupby("model",as_index=False).value.mean().rename(columns={"value":"params"})
    p=err.merge(lat,on="model").merge(par,on="model")
    persistence=d[(d.analysis=="primary_horizon_specific")&(d.metric=="range_nRMSE")&(d.statistic=="deterministic")&(d.model==MODELS[0])].value.mean()
    c=Canvas(str(OUT/"fig4_accuracy_efficiency.pdf"),pagesize=(420,300)); x0,y0,w,h=62,52,320,200; c.rect(x0,y0,w,h,fill=0,stroke=1); xmin,xmax=math.log10(.35),math.log10(45); ymin,ymax=.10,.35
    py=y0+h*(persistence-ymin)/(ymax-ymin); c.setDash(4,3); c.line(x0,py,x0+w,py); c.setDash(); text(c,x0+w-3,py+4,"Last-value persistence macro error",6.3,"end")
    maxp=p.params.max()
    for _,r in p.iterrows():
        xx=x0+w*(math.log10(r.latency)-xmin)/(xmax-xmin); yy=y0+h*(r.error-ymin)/(ymax-ymin); radius=3+6*math.sqrt(r.params/maxp)
        marker(c,xx,yy,r.model,radius); dx=-7 if r.model==MODELS[1] else 7; text(c,xx+dx,yy+radius+3,SHORT[r.model],6.2,"end" if dx<0 else "start")
    for tick in [.4,1,3,10,30]: text(c,x0+w*(math.log10(tick)-xmin)/(xmax-xmin),y0-13,tick,6,"middle")
    for tick in [.12,.16,.20,.24,.28,.32]: text(c,x0-7,y0+h*(tick-ymin)/(ymax-ymin)-2,f"{tick:.2f}",6,"end")
    text(c,210,280,"Corrected accuracy--efficiency trade-off",9.4,"middle",True); text(c,222,25,"Mean batch-one GPU latency (ms, log scale)",7,"middle")
    c.saveState(); c.translate(16,154); c.rotate(90); text(c,0,0,"Macro mean Train-range nRMSE",7,"middle"); c.restoreState(); text(c,222,9,"Marker area is exactly proportional to parameter count",6.2,"middle")
    c.showPage(); c.save()

def main():
    d=load(); figure_protocol(); figure_curves(d); figure_rank_heatmap(d); figure_pareto(d)
    expected=["fig1_leakage_free_protocol.pdf","fig2_multihorizon_nrmse.pdf","fig3_rank_heatmap.pdf","fig4_accuracy_efficiency.pdf"]
    assert sorted(p.name for p in OUT.glob("*.pdf"))==expected; print("Generated:",", ".join(expected))

if __name__=="__main__": main()
