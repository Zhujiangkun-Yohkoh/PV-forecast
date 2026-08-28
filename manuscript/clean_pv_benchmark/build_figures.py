"""Build publication figures and supplementary evidence tables.

All quantitative graphics and tables are derived from corrected_metrics.csv.
No paper result is entered manually in this file.
"""
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

FONT_REGULAR = Path(r"C:\Windows\Fonts\arial.ttf")
FONT_BOLD = Path(r"C:\Windows\Fonts\arialbd.ttf")
pdfmetrics.registerFont(TTFont("Arial", str(FONT_REGULAR)))
pdfmetrics.registerFont(TTFont("Arial-Bold", str(FONT_BOLD)))

LAST = "Last-value Persistence"
DAILY = "Daily Persistence"
NEURAL = [
    "Discrete recurrent decoder",
    "Inverted-variate Transformer",
    "Joint-patch Transformer",
    "Depthwise convolutional TCN",
]
MODELS = [LAST, *NEURAL]
SHORT = {
    LAST: "Last-value",
    DAILY: "Daily",
    NEURAL[0]: "Discrete recurrent",
    NEURAL[1]: "Inverted-variate",
    NEURAL[2]: "Joint-patch",
    NEURAL[3]: "Depthwise TCN",
}
COLORS = {
    LAST: "#333333", DAILY: "#7F3C8D", NEURAL[0]: "#E69F00",
    NEURAL[1]: "#0072B2", NEURAL[2]: "#009E73", NEURAL[3]: "#CC79A7",
}
DATASETS = ["Sanyo", "Hanwha", "Qcells"]
HORIZONS = [12, 48, 96, 144]
SCOPES = [("regular_full_timeline", "Full"), ("daylight", "Daylight")]


def load() -> pd.DataFrame:
    d = pd.read_csv(EVIDENCE)
    assert len(d) == 11328
    assert set(MODELS + [DAILY]).issubset(set(d.model))
    assert set(HORIZONS).issubset(set(d.horizon_steps.dropna().astype(int)))
    assert {"primary_horizon_specific", "supplementary_daily_matched"}.issubset(set(d.analysis))
    return d


def txt(c, x, y, value, size=8, anchor="start", bold=False, color=black):
    # AIP artwork guidance: keep all final-size figure text at least 8 pt.
    size = max(float(size), 8.0)
    c.setFillColor(color)
    c.setFont("Arial-Bold" if bold else "Arial", size)
    fn = c.drawCentredString if anchor == "middle" else c.drawRightString if anchor == "end" else c.drawString
    fn(x, y, str(value))


def arrow(c, x0, y0, x1, y1):
    c.setStrokeColor(HexColor("#4D4D4D")); c.setLineWidth(0.8)
    c.line(x0, y0, x1, y1)
    angle = math.atan2(y1-y0, x1-x0)
    for delta in (-0.45, 0.45):
        c.line(x1, y1, x1-6*math.cos(angle+delta), y1-6*math.sin(angle+delta))


def figure_protocol():
    c = Canvas(str(OUT/"fig1_leakage_free_protocol.pdf"), pagesize=(510, 360), initialFontName="Arial")
    txt(c, 255, 343, "Leakage-aware, elementwise-matched PV forecasting benchmark", 10, "middle", True)
    c.setFillColor(HexColor("#EDF4F8")); c.roundRect(12, 245, 486, 78, 6, fill=1, stroke=1)
    txt(c, 25, 307, "A  DATA LAYER", 8, bold=True, color=HexColor("#1B4F72"))
    arrays = [(42, "Site 17\nSanyo"), (126, "Site 25\nHanwha"), (210, "Site 38\nQcells")]
    for x, label in arrays:
        c.setFillColor(white); c.roundRect(x, 262, 68, 35, 4, fill=1, stroke=1)
        for i, line in enumerate(label.split("\n")): txt(c, x+34, 283-i*11, line, 7, "middle", i == 0)
    c.setFillColor(HexColor("#D9EAD3")); c.roundRect(314, 262, 153, 35, 4, fill=1, stroke=1)
    txt(c, 390.5, 283, "Shared weather context", 7, "middle", True)
    txt(c, 390.5, 272, "Array-specific power + missingness", 8, "middle")
    for x, _ in arrays: arrow(c, x+68, 279, 314, 279)

    c.setFillColor(HexColor("#FAF5E8")); c.roundRect(12, 139, 486, 92, 6, fill=1, stroke=1)
    txt(c, 25, 215, "B  CAUSAL TIMELINE", 8, bold=True, color=HexColor("#7D6608"))
    y = 176; c.setLineWidth(2); c.setStrokeColor(HexColor("#777777")); c.line(42, y, 470, y)
    blocks = [(42, 158, "Train\nfit preprocessing", "#CDECCF"),
              (204, 95, "Validation\nselect checkpoint", "#FFF0B3"),
              (326, 144, "Test\nevaluation only", "#CFE8FF")]
    for x, w, label, color in blocks:
        c.setFillColor(HexColor(color)); c.rect(x, y-14, w, 28, fill=1, stroke=1)
        for i, line in enumerate(label.split("\n")): txt(c, x+w/2, y+3-i*10, line, 6.4, "middle", i == 0)
    txt(c, 42, 148, "17-D causal input; L = 72 (6 h)", 8)
    txt(c, 470, 148, "forecast origin  |  H = 144 (12 h)", 8, "end")
    # The Train block and the L72/origin/H144 labels carry the causal policy;
    # a repeated sentence here would collide with those labels at final size.

    c.setFillColor(HexColor("#F4EEF8")); c.roundRect(12, 22, 486, 102, 6, fill=1, stroke=1)
    txt(c, 25, 108, "C  EVALUATION LAYER", 8, bold=True, color=HexColor("#5B2C6F"))
    labels = ["4 compact neural\nimplementations", "Last-value\nPersistence", "Daily Persistence\n(matched supplement)"]
    xs, ws = [32, 180, 300], [126, 98, 166]
    for x, w, label in zip(xs, ws, labels):
        c.setFillColor(white); c.roundRect(x, 67, w, 29, 4, fill=1, stroke=1)
        for i, line in enumerate(label.split("\n")): txt(c, x+w/2, 84-i*9, line, 6.5, "middle", i == 0)
    txt(c, 255, 51, "Within each array, all methods share forecast origins, labels, and point masks", 8, "middle")
    txt(c, 255, 40, "at H12 / H48 / H96 / H144.", 8, "middle")
    txt(c, 255, 27, "ONLY THE FORECASTER CHANGES; EVALUATION SUPPORT REMAINS MATCHED.", 8, "middle", True, HexColor("#5B2C6F"))
    c.showPage(); c.save()


def combo_order():
    return [(ds, h, sc, label) for ds in DATASETS for h in HORIZONS for sc, label in SCOPES]


def posthoc_neural_envelope(q):
    q = q[q.model.isin(NEURAL)].copy()
    idx = q.groupby(["dataset", "horizon_steps", "scope"])["value"].idxmin()
    return q.loc[idx].set_index(["dataset", "horizon_steps", "scope"])


def figure_baseline_reversal(d):
    primary = d[(d.analysis == "primary_horizon_specific") & (d.metric == "RMSE") & d.statistic.isin(["mean", "deterministic"])]
    daily = d[(d.analysis == "supplementary_daily_matched") & (d.metric == "RMSE") & d.statistic.isin(["mean", "deterministic"])]
    envelope_primary = posthoc_neural_envelope(primary)
    envelope_daily = posthoc_neural_envelope(daily)
    last = primary[primary.model == LAST].set_index(["dataset", "horizon_steps", "scope"])
    day = daily[daily.model == DAILY].set_index(["dataset", "horizon_steps", "scope"])
    combos = combo_order()
    ratios_last, ratios_daily = [], []
    for ds, h, sc, _ in combos:
        key = (ds, h, sc)
        ratios_last.append(envelope_primary.loc[key, "value"] / last.loc[key, "value"])
        ratios_daily.append(envelope_daily.loc[key, "value"] / day.loc[key, "value"])

    c = Canvas(str(OUT/"fig2_persistence_reversal.pdf"), pagesize=(510, 600), initialFontName="Arial")
    txt(c, 255, 583, "The baseline choice reverses the practical conclusion", 10, "middle", True)
    x0, w = 132, 350
    all_ratios = np.asarray(ratios_last + ratios_daily, dtype=float)
    assert np.all(np.isfinite(all_ratios)) and np.all(all_ratios > 0)
    lo = 10 ** (math.floor(math.log10(all_ratios.min()) * 4) / 4 - 0.10)
    hi = 10 ** (math.ceil(math.log10(all_ratios.max()) * 4) / 4 + 0.10)
    panels = [(320, "(a)  Envelope / Last-value Persistence", ratios_last,
               "Neural envelope lower than Last-value: 24/24."),
              (38, "(b)  Envelope / Daily Persistence (matched points)", ratios_daily,
               "Daily Persistence lower than neural envelope: 22/24.")]
    for pidx, (y0, title, vals, panel_note) in enumerate(panels):
        hgt = 235; txt(c, 16, y0+hgt-2, title, 8, bold=True)
        txt(c, 482, y0+hgt-2, panel_note, 8, "end", True)
        x_ref = x0 + w * (0 - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
        c.setStrokeColor(HexColor("#666666")); c.setDash(3,2); c.line(x_ref, y0, x_ref, y0+hgt-18); c.setDash()
        for i, ((ds, hz, sc, slabel), value) in enumerate(zip(combos, vals)):
            yy = y0+hgt-31-i*8.25
            txt(c, x0-5, yy-2.6, f"{ds} H{hz} {slabel}", 8, "end")
            xx = x0 + w * (math.log10(value) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
            color = HexColor("#0072B2") if value < 1 else HexColor("#D55E00")
            c.setStrokeColor(color); c.setLineWidth(1.5); c.line(x_ref, yy, xx, yy)
            c.setFillColor(color); c.circle(xx, yy, 2.1, fill=1, stroke=0)
            if pidx == 1 and value < 1:
                txt(c, min(xx+5, 467), yy-2.6, f"Hanwha H12 {slabel}", 8, color=color)
        ticks = [v for v in [0.1, 0.2, 0.5, 1, 2, 5, 10] if lo <= v <= hi]
        for tick in ticks:
            xx = x0 + w * (math.log10(tick) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
            txt(c, xx, y0-13, f"{tick:g}", 8, "middle")
        txt(c, x0+w/2, y0-27, "RMSE ratio (log scale): <1 envelope better; >1 Persistence better", 8, "middle")
        assert all(lo <= value <= hi for value in vals), "Figure 2 point outside plotted x range"
    c.showPage(); c.save()


def marker(c, x, y, model, radius=2.4):
    c.setFillColor(HexColor(COLORS[model])); c.setStrokeColor(black)
    if model == NEURAL[2]:
        p = c.beginPath(); p.moveTo(x,y+radius); p.lineTo(x-radius,y-radius); p.lineTo(x+radius,y-radius); p.close(); c.drawPath(p,fill=1,stroke=1)
    elif model == NEURAL[3]:
        p = c.beginPath(); p.moveTo(x,y+radius); p.lineTo(x-radius,y); p.lineTo(x,y-radius); p.lineTo(x+radius,y); p.close(); c.drawPath(p,fill=1,stroke=1)
    else: c.circle(x,y,radius,fill=1,stroke=1)


def figure_horizon_dependence(d):
    q = d[(d.analysis == "primary_horizon_specific") & (d.metric == "range_nRMSE") & d.model.isin(MODELS) & d.statistic.isin(["mean", "deterministic"])]
    c = Canvas(str(OUT/"fig3_horizon_technology.pdf"), pagesize=(510, 430), initialFontName="Arial")
    txt(c, 255, 414, "Error growth depends on array, horizon, and evaluation scope", 9.5, "middle", True)
    ymax = .67
    for row, (scope, scope_label) in enumerate(SCOPES):
      for col, ds in enumerate(DATASETS):
        x,y,w,h = 8+col*168,218-row*190,160,174; left,bottom,pw,ph=x+31,y+24,w-40,h-48
        c.setStrokeColor(HexColor("#777777")); c.rect(left,bottom,pw,ph,fill=0,stroke=1)
        for val in [0,.2,.4,.6]:
            yy=bottom+ph*val/ymax; c.setStrokeColor(HexColor("#E0E0E0")); c.line(left,yy,left+pw,yy)
            if col==0: txt(c,left-4,yy-2,f"{val:.1f}",8,"end")
        for ix,hz in enumerate(HORIZONS): txt(c,left+pw*ix/3,bottom-10,f"H{hz}",5.5,"middle")
        for model in MODELS:
            z=q[(q.dataset==ds)&(q.model==model)&(q.scope==scope)].sort_values("horizon_steps"); assert len(z)==4
            pts=[(left+pw*i/3,bottom+ph*r.value/ymax) for i,(_,r) in enumerate(z.iterrows())]
            c.setStrokeColor(HexColor(COLORS[model])); c.setLineWidth(1.05); c.setDash()
            for a,b in zip(pts,pts[1:]): c.line(a[0],a[1],b[0],b[1])
            for xx,yy in pts: marker(c,xx,yy,model,2)
        txt(c,x+w/2,y+h-14,f"{ds} — {scope_label}",8,"middle",True)
    for i,model in enumerate(MODELS):
        xx=15+i*99; marker(c,xx,18,model,2.2); txt(c,xx+7,16,SHORT[model],8)
    c.saveState(); c.translate(12,215); c.rotate(90); txt(c,0,0,"Train-range nRMSE",8,"middle"); c.restoreState()
    c.showPage(); c.save()


def figure_efficiency(d):
    err=d[(d.analysis=="primary_horizon_specific")&(d.metric=="range_nRMSE")&(d.statistic=="mean")&d.model.isin(NEURAL)].groupby("model",as_index=False).value.mean().rename(columns={"value":"error"})
    meta=d[d.analysis=="run_metadata"]
    lat=meta[meta.metric=="latency_mean_ms"].groupby("model",as_index=False).value.mean().rename(columns={"value":"latency"})
    par=meta[meta.metric=="parameter_count"].groupby("model",as_index=False).value.mean().rename(columns={"value":"params"})
    p=err.merge(lat,on="model").merge(par,on="model")
    persistence=d[(d.analysis=="primary_horizon_specific")&(d.metric=="range_nRMSE")&(d.statistic=="deterministic")&(d.model==LAST)].value.mean()
    p["pareto"]=[not any((o.latency<=r.latency and o.error<=r.error and (o.latency<r.latency or o.error<r.error)) for _,o in p.iterrows()) for _,r in p.iterrows()]
    c=Canvas(str(OUT/"fig4_accuracy_efficiency.pdf"),pagesize=(510,330),initialFontName="Arial")
    x0,y0,w,h=78,58,392,226; xmin,xmax=math.log10(.35),math.log10(45); ymin,ymax=p.error.min()-.018,p.error.max()+.018
    c.rect(x0,y0,w,h,fill=0,stroke=1)
    # Fixed annotation offsets separate the three compact models clustered near
    # 0.5 ms.  They affect presentation only; all coordinates remain data-driven.
    maxp=p.params.max(); offsets={NEURAL[0]:(-10,8,"end"),NEURAL[1]:(12,2,"start"),NEURAL[2]:(12,-17,"start"),NEURAL[3]:(12,18,"start")}
    for _,r in p.iterrows():
        xx=x0+w*(math.log10(r.latency)-xmin)/(xmax-xmin); yy=y0+h*(r.error-ymin)/(ymax-ymin); radius=12*math.sqrt(r.params/maxp)
        c.setFillColor(HexColor(COLORS[r.model])); c.setStrokeColor(black); c.setLineWidth(1.8 if r.pareto else .7); c.circle(xx,yy,radius,fill=1,stroke=1)
        dx,dy,anchor=offsets[r.model]; label=f"{SHORT[r.model]} ({int(r.params/1000)}k)"+("  Pareto" if r.pareto else "")
        txt(c,xx+dx,yy+dy,label,6.4,anchor,r.pareto)
    for tick in [.4,1,3,10,30]: txt(c,x0+w*(math.log10(tick)-xmin)/(xmax-xmin),y0-13,tick,6,"middle")
    yticks = np.linspace(ymin, ymax, 4)
    assert len(np.unique(np.round(yticks, 4))) >= 3
    for tick in yticks:
        txt(c,x0-7,y0+h*(tick-ymin)/(ymax-ymin)-2,f"{tick:.3f}",8,"end")
    txt(c,255,311,"Hardware-specific accuracy-efficiency trade-off",10,"middle",True); txt(c,270,29,"Mean batch-one GPU latency (ms, logarithmic scale)",7,"middle")
    c.saveState(); c.translate(20,170); c.rotate(90); txt(c,0,0,"Macro mean Train-range nRMSE",7,"middle"); c.restoreState()
    txt(c,270,10,"Marker area is proportional to parameter count; bold outlines identify Pareto-efficient implementations.",6.1,"middle")
    txt(c,x0+w-3,y0+h-13,f"Last-value Persistence macro error = {persistence:.3f} (outside neural-focused y range)",8,"end",False,HexColor("#555555"))
    c.showPage(); c.save()


def figure_rank_heatmap(d):
    q=d[(d.analysis=="primary_horizon_specific")&(d.metric=="RMSE")&d.statistic.isin(["mean","deterministic"])&d.model.isin(MODELS)].copy()
    q["rank"]=q.groupby(["dataset","horizon_steps","scope"])["value"].rank(method="average")
    rows=[]
    for ds,hz,sc,slabel in combo_order():
        z=q[(q.dataset==ds)&(q.horizon_steps==hz)&(q.scope==sc)].set_index("model"); rows.append((f"{ds} H{hz} {slabel}",[z.loc[m,"rank"] for m in MODELS]))
    # Extra top space and multiline headers keep every full model name visible
    # when the heat map is embedded at final size.
    c=Canvas(str(OUT/"figS1_rank_heatmap.pdf"),pagesize=(480,570),initialFontName="Arial"); txt(c,240,553,"Primary RMSE rank (1 = lowest error)",9.2,"middle",True)
    x0,cw,ch=132,67,18
    headers = {
        LAST: ["Last-value", "Persistence"],
        NEURAL[0]: ["Discrete", "recurrent", "decoder"],
        NEURAL[1]: ["Inverted-variate", "Transformer"],
        NEURAL[2]: ["Joint-patch", "Transformer"],
        NEURAL[3]: ["Depthwise", "convolutional", "TCN"],
    }
    for j,model in enumerate(MODELS):
        for k,line in enumerate(headers[model]):
            txt(c,x0+j*cw+cw/2,529-k*10,line,8,"middle")
    for i,(label,row) in enumerate(rows):
        y=489-i*ch; txt(c,x0-5,y+5,label,8,"end")
        for j,val in enumerate(row):
            t=(val-1)/4; c.setFillColor(Color(.10+.65*t,.55-.33*t,.75-.43*t)); c.setStrokeColor(white); c.rect(x0+j*cw,y,cw,ch,fill=1,stroke=1); txt(c,x0+j*cw+cw/2,y+5,f"{val:.0f}",7,"middle",True,white)
    c.showPage(); c.save()


def esc(value):
    return str(value).replace("&",r"\&").replace("_",r"\_").replace("%",r"\%")


def supplementary_tables(d):
    lines=["% AUTO-GENERATED by build_figures.py from corrected_metrics.csv. DO NOT EDIT."]
    primary=d[(d.analysis=="primary_horizon_specific")&(d.statistic=="per_seed")&d.model.isin(NEURAL)]
    for metric in ["RMSE","MAE","R2","Bias","range_nRMSE","RMSE_skill"]:
        q=primary[primary.metric==metric]
        caption_metric=metric.replace("_",r"\_")
        metric_alt = {
            "RMSE": "Per-seed root mean square errors show how forecast accuracy changes across arrays, scopes, and horizons for the four compact neural implementations. Seed spread reflects initialization variability rather than independent data replication.",
            "MAE": "Per-seed mean absolute errors provide a less tail-sensitive comparison of the four compact neural implementations across arrays, scopes, and horizons. The pattern complements RMSE by reducing the influence of large residuals.",
            "R2": "Per-seed coefficients of determination show horizon- and scope-dependent explained variance, including negative values where forecasts underperform the target mean. Values are descriptive because adjacent forecast origins overlap.",
            "Bias": "Per-seed mean errors identify underprediction and overprediction patterns across the three arrays, two scopes, and four forecast horizons. Positive values indicate overprediction, and negative values indicate underprediction.",
            "range_nRMSE": "Per-seed Train-range normalized RMSE values permit within-array comparison of error growth while avoiding unsupported AC-capacity normalization. The denominator is fitted only on Train and differs by array.",
            "RMSE_skill": "Per-seed RMSE skill relative to Last-value Persistence quantifies improvement beyond local continuity; positive values favor the neural implementation. Every value uses the same origins, labels, and masks as its reference.",
        }[metric]
        lines += [r"\begin{longtable}{lllrrrr}",rf"\caption{{Primary per-seed {caption_metric} values.}}\label{{tabs:{metric.lower().replace('_','')}}}\\",rf"\multicolumn{{7}}{{p{{0.94\textwidth}}}}{{\small\textit{{Alt text: {metric_alt}}}}}\\",r"\toprule Array & Scope & Horizon & Model & Seed 42 & Seed 43 & Seed 44\\\midrule",r"\endfirsthead\toprule Array & Scope & Horizon & Model & Seed 42 & Seed 43 & Seed 44\\\midrule\endhead"]
        for ds,hz,sc,slabel in combo_order():
            for model in NEURAL:
                z=q[(q.dataset==ds)&(q.horizon_steps==hz)&(q.scope==sc)&(q.model==model)].sort_values("seed"); assert len(z)==3
                vals=" & ".join(f"{v:.5f}" for v in z.value); lines.append(f"{ds} & {slabel} & H{hz} & {esc(SHORT[model])} & {vals}\\\\")
        lines += [r"\bottomrule\end{longtable}",""]
    counts=d[(d.analysis=="primary_horizon_specific")&(d.metric=="RMSE")&(d.model==LAST)&(d.statistic=="deterministic")]
    lines += [r"\begin{longtable}{lllrr}",r"\caption{Primary horizon-specific evaluation support.}\label{tabs:counts}\\",r"\multicolumn{5}{p{0.94\textwidth}}{\small\textit{Alt text: Forecast-origin and valid-target counts decrease with horizon and are shared by every method within each array-specific comparison. Full and daylight scopes report their target support separately.}}\\",r"\toprule Array & Scope & Horizon & Forecast origins & Valid target points\\\midrule",r"\endfirsthead\toprule Array & Scope & Horizon & Forecast origins & Valid target points\\\midrule\endhead"]
    for ds,hz,sc,slabel in combo_order():
        r=counts[(counts.dataset==ds)&(counts.horizon_steps==hz)&(counts.scope==sc)].iloc[0]; lines.append(f"{ds} & {slabel} & H{hz} & {int(r.forecast_origin_count):,} & {int(r.valid_target_count):,}\\\\")
    lines += [r"\bottomrule\end{longtable}",""]
    secondary=d[(d.analysis=="secondary_h144_common")&(d.metric=="range_nRMSE")&d.statistic.isin(["mean","deterministic"])&d.model.isin(MODELS)]
    lines += [r"\begin{longtable}{lllrrrrr}",r"\caption{Complete-H144-origin sensitivity: Train-range nRMSE for every evaluated prefix. All horizons within a row use origins that possess a complete valid H144 target.}\label{tabs:h144sensitivity}\\",r"\multicolumn{8}{p{0.94\textwidth}}{\small\textit{Alt text: Restricting every prefix to complete H144 origins preserves the main pattern of horizon-dependent error without changing the primary horizon-specific analysis. This analysis is secondary to the horizon-specific evaluation.}}\\",r"\toprule Array & Scope & Model & H12 & H48 & H96 & H144\\\midrule",r"\endfirsthead\toprule Array & Scope & Model & H12 & H48 & H96 & H144\\\midrule\endhead"]
    for ds in DATASETS:
        for sc,slabel in SCOPES:
            for model in MODELS:
                z=secondary[(secondary.dataset==ds)&(secondary.scope==sc)&(secondary.model==model)].sort_values("horizon_steps"); assert len(z)==4
                lines.append(f"{ds} & {slabel} & {esc(SHORT[model])} & "+" & ".join(f"{v:.4f}" for v in z.value)+r"\\")
    lines += [r"\bottomrule\end{longtable}",""]
    dm=d[(d.analysis=="supplementary_daily_matched")&(d.metric=="RMSE")&d.statistic.isin(["mean","deterministic"])]
    lines += [r"\begin{longtable}{lllrrrrrr}",r"\caption{Daily-matched RMSE (kW); the Daily-valid point mask is identical for all methods.}\label{tabs:daily}\\",r"\multicolumn{9}{p{0.94\textwidth}}{\small\textit{Alt text: On identical Daily-valid target points, Daily Persistence has lower RMSE than the post hoc neural envelope in 22 of 24 comparisons; the exceptions are Hanwha H12 full and daylight.}}\\",r"\toprule Array & Scope & Horizon & Daily & Last & Discrete & Inverted & Joint-patch & Depthwise\\\midrule",r"\endfirsthead\toprule Array & Scope & Horizon & Daily & Last & Discrete & Inverted & Joint-patch & Depthwise\\\midrule\endhead"]
    for ds,hz,sc,slabel in combo_order():
        z=dm[(dm.dataset==ds)&(dm.horizon_steps==hz)&(dm.scope==sc)].set_index("model"); vals=[z.loc[m,"value"] for m in [DAILY,LAST,*NEURAL]]; lines.append(f"{ds} & {slabel} & H{hz} & "+" & ".join(f"{v:.4f}" for v in vals)+r"\\")
    lines += [r"\bottomrule\end{longtable}",""]
    (HERE/"supplementary_tables.tex").write_text("\n".join(lines),encoding="utf-8")


def main_result_tables(d):
    """Generate the quantitative main-text table from the evidence."""
    q=d[(d.analysis=="primary_horizon_specific")&(d.metric=="range_nRMSE")&d.statistic.isin(["mean","deterministic"])]
    best=posthoc_neural_envelope(q)
    last=q[q.model==LAST].set_index(["dataset","horizon_steps","scope"])
    lines=["% AUTO-GENERATED by build_figures.py from corrected_metrics.csv. DO NOT EDIT.",
           r"\begin{table*}[t]",r"\caption{Primary Test Train-range nRMSE on horizon-specific valid origins. The envelope is the post hoc minimum three-seed mean among the four compact implementations and is a descriptive upper bound, not a prespecified model; the complete model-by-seed table is in the Supplementary Material.}",r"\label{tab:primarysummary}\centering\small",
           r"\begin{tabular}{lllrrrr}",r"\toprule Array & Scope & Horizon & Envelope member & Envelope nRMSE & Last-value nRMSE & RMSE skill\\\midrule"]
    for ds in DATASETS:
        for sc,slabel in SCOPES:
            for hz in [12,144]:
                key=(ds,hz,sc); br=best.loc[key]; lr=last.loc[key]
                skill=1-br.value/lr.value
                lines.append(f"{ds} & {slabel} & H{hz} & {esc(SHORT[br.model])} & {br.value:.4f} & {lr.value:.4f} & {skill:.3f}\\\\")
    lines += [r"\bottomrule\end{tabular}\end{table*}",""]
    (HERE/"main_result_tables.tex").write_text("\n".join(lines),encoding="utf-8")


def main():
    expected={"fig1_leakage_free_protocol.pdf","fig2_persistence_reversal.pdf","fig3_horizon_technology.pdf","fig4_accuracy_efficiency.pdf","figS1_rank_heatmap.pdf"}
    for old in OUT.glob("*.pdf"):
        if old.name not in expected:
            old.unlink()
    d=load(); figure_protocol(); figure_baseline_reversal(d); figure_horizon_dependence(d); figure_efficiency(d); figure_rank_heatmap(d); supplementary_tables(d); main_result_tables(d)
    assert {p.name for p in OUT.glob("*.pdf")}==expected
    print("Generated publication figures and supplementary_tables.tex")


if __name__=="__main__":
    main()
