"""One-time manuscript closeout authoring; tables/figures derive from saved results."""
from pathlib import Path
import re,json
H=Path(__file__).resolve().parent;ROOT=H.parents[1];P=ROOT/'manuscript/clean_pv_benchmark';F=P/'closeout_figures';p=P/'main.tex';s=p.read_text(encoding='utf8')
assert 'The restored-origin neural comparison is not promoted here' in s,'Do not rerun one-time authoring over the final manuscript'
abstract='''Photovoltaic forecast gains depend on the reference history and the targets being scored. We compare four compact neural implementations with persistence and linear references for five power systems at three sites, using six-hour histories and cumulative forecast windows up to twelve hours. Ridge contrasts separate recent-history learning from additional previous-day information. At Alice Springs, complete-window rejection spreads sparse negative measurements across many overlapping forecasts. Pointwise scoring restores these targets selectively: at Qcells, the verified Inverted model's twelve-hour full-target RMSE rises from 0.844 to 1.080 kW, while its higher error than Daily and lower error than Last-value remain. Hanwha retains the same reference ordering. Two external systems are evaluated with frozen models in October--December 2017 and a fixed April--June 2018 period under both complete-window and pointwise rules. NIST's Inverted-minus-Daily difference changes from positive to negative under either rule; Yulara's original and expanded Ridge fits also change order between periods. Paired temporal intervals qualify these effects, and disjoint power-range squared errors explain the changing aggregate balance. These comparisons distinguish reference information, scoring population and evaluation period, rather than identifying a universally preferable architecture.'''
s=re.sub(r'(?s)(\\begin\{abstract\}).*?(\\end\{abstract\})',lambda m:m.group(1)+'\n'+abstract+'\n'+m.group(2),s)
s=s.replace('The decisive diagnostic concerns missingness exposure.','The diagnostic evidence points to missingness exposure.').replace('Figure~\\ref{fig:ridgeintervals} now shows','Figure~\\ref{fig:ridgeintervals} shows').replace('At Qcells, the observed $-0.0071$ kW B-minus-Daily difference','At Qcells, the original-grid $-0.0071$ kW B-minus-Daily difference')
start=s.index('Each new-period comparison scores');end=s.index('\\section{Results}',start)
s=s[:start]+r'''For both external periods, a common candidate rule includes every five-minute origin in the stated calendar interval, with at least twenty-four hours of prior data and twelve hours of subsequent target buffer. The two scoring rules use either all 144 finite labels or each individually finite label, followed by identical finite-origin and Daily point intersections. Finite negative external power remains eligible. The October--December 2017 comparison therefore includes historical and January 1 target buffers; the original archived boundary counts and scores are retained separately. Both supports derive from the same candidates within each period. Methods are paired within a period; dates in different years are not paired.

For Alice, S1 scores finite nonnegative targets individually, whereas S2 scores finite raw targets. Both retain the frozen historical input and nonnegative Daily/Last-value rules. Original eligibility and newly restored origins are defined separately for each cumulative prefix. The new comparisons accept only methods reproducing their archived-origin predictions within the unchanged numerical criterion. Hanwha and Qcells each include all three Inverted seeds; Sanyo's accepted seed42 is reported separately without a fabricated three-seed mean. Joint-patch provides verified secondary comparisons. Remaining numerical sensitivities are documented rather than included in accepted new-support rankings. This changes the evaluation population while retaining the original training population and model weights.

''' .replace('\\n','\n')+s[end:]
# Move unchanged historical plots to electronic supplement, without redrawing data.
moved=[]
for stem in ['fig7_qcells_selection','fig2_alice_references','fig3_external_skills']:
 pattern=r'\\begin\{figure\*\}.*?\\end\{figure\*\}'
 blocks=list(re.finditer(pattern,s,re.S));block=next(m.group(0) for m in blocks if stem+'.pdf' in m.group(0));s=s.replace(block,'');moved.append(stem)
s=s.replace('Fig.~\\ref{fig:qcellsselection}','Supplementary Fig.~S15').replace('Figure~\\ref{fig:alice}','Supplementary Fig.~S16').replace('Fig.~\\ref{fig:external}','Supplementary Fig.~S17')
insert=s.index('\\subsection{Explaining the NIST full-target reversal}')
new=r'''\subsubsection{Frozen models on restored target support}
The verified Hanwha and Qcells Inverted models retain lower full-target error than Last-value but higher error than Daily and Ridge B under both pointwise rules (Fig.~\ref{fig:restored}). At twelve hours, Qcells Inverted mean RMSE changes from 0.844 kW on original support to 1.080 kW on S1; Daily changes from 0.244 to 0.275 kW. The Inverted-minus-Daily effect increases from 0.600 to 0.805 kW, with the S1 48-hour interval [0.654,0.986] kW. On newly restored origins alone the effect is 0.942 kW [0.742,1.196]. Hanwha instead changes from 0.805 to 0.769 kW, but its S1 difference from Daily remains positive at 0.586 kW [0.487,0.736]. S2 gives closely similar twelve-hour effects. The one-hour effects and all target ranges are retained, without assuming that restored targets follow the complete-window distribution.

Qcells twelve-hour Daily-matched support expands from 2996 origins and 431150 target pairs to 6786 origins and 958668 pairs under S1, with 959382 pairs under S2. The active share also changes. These are changes in the scored population, not improvements or degradation caused by updating a model. They do not establish how training on a different target mask would perform.

\begin{figure*}[!t]\centering
\includegraphics[width=0.97\textwidth]{closeout_figures/fig9_alice_support.pdf}
\caption{Verified three-seed Inverted minus Daily or expanded Ridge B on full targets at Hanwha and Qcells. Original support is prefix-specific; S1 scores nonnegative targets individually and S2 finite raw targets. Points average seed-specific RMSE differences and bars are 95\% paired 48 h block intervals. Negative favors Inverted. Sanyo has no accepted three-seed mean in this sensitivity.}
\label{fig:restored}\end{figure*}

'''
s=s[:insert]+new+s[insert:]
start=s.index('\\subsection{Same-site next-year fixed-period results}');end=s.index('\\section{Interpretation and limits}',start)
s=s[:start]+r'''\subsection{Period contrasts under the same scoring rules}
Figure~\ref{fig:nextyear} separates support changes from period contrasts using two rules in each period. At Yulara, all 2017 candidate trajectories have finite labels, so complete-H144 and pointwise results coincide. In 2018 their small differences do not alter the reference ordering. On unified complete support, original B minus Daily changes from +19.72 kW in 2017 to -1.54 kW in 2018; expanded B changes from -0.27 to +1.28 kW. The 2017 expanded-B interval [-1.85,1.67] kW crosses zero, whereas the 2018 interval [0.42,2.27] is positive. Original and expanded fits remain separately reported, not reselected using new-period scores.

At NIST, Inverted minus Daily is +2.70 kW [-1.03,6.47] under complete-H144 support in 2017 and -12.45 kW [-17.57,-7.01] in 2018. Pointwise support yields +2.54 kW [-1.00,6.10] and -12.84 kW [-17.40,-8.00], respectively. Thus the point-estimate reversal is present under either scoring rule. The 2017 crossing intervals do not establish a stable disadvantage, and intervals are conditional on each observed period. Original and expanded NIST B remain identical and lower than Daily in all four cells.

The unified 2017 candidate rule includes origins before the archive's first 06:00 origin and restores end-of-period target buffers. For example, NIST complete support contains 3158261 Daily-matched pairs, compared with 3128676 in the original archive. Neither original tables nor original prediction arrays are replaced. Separately reporting the archival replay and the unified-boundary four cells prevents a boundary change from being presented as model improvement.

The NIST Inverted-minus-Daily full-MSE balance also changes under either rule. With complete-H144 support, active and low-power contributions are -418.95 and +652.33 kW$^2$ in 2017, versus -1241.03 and +174.03 in 2018. The corresponding pointwise values are -411.05 and +632.34, versus -1275.75 and +130.09 kW$^2$. Active savings and low-power costs persist, but their magnitudes reverse the aggregate balance. These are additive per-seed squared-error differences averaged over seeds, not squared mean RMSEs or a diagnosis of specific operating events.

\begin{figure*}[!t]\centering
\includegraphics[width=0.97\textwidth]{closeout_figures/fig8_fixed_period.pdf}
\caption{Same-site four-cell comparisons at twelve hours: October--December 2017 and April--June 2018, each on complete-H144 and pointwise-finite support with unified buffers. Negative RMSE differences favor the named method over Daily. Bars are 95\% paired 48 h block intervals within each period; different years are not date-paired. Inverted averages three seed-specific effects; original and expanded Ridge B remain distinct. Scales differ between panels.}
\label{fig:nextyear}\end{figure*}

\clearpage
''' + s[end:]
s=s.replace('Period differences include model age, input exposure, operating conditions and scoring support, not an isolated seasonal effect.','Under either fixed scoring rule, period differences can include model age, input exposure and operating conditions; the four-cell comparison does not identify an isolated seasonal effect.')
s=s.replace('Qcells support counts further show how whole-window rejection can select a different target population.','At Hanwha and Qcells, restored pointwise targets change absolute errors while retaining the verified Inverted model\'s ordering against recent-value and daily references. Training-selection effects remain outside this frozen-weight sensitivity.')
s=s.replace('fixed_period_figures/','closeout_figures/').replace('\\begin{figure*}[p]','\\begin{figure*}[!t]')
p.write_text(s.rstrip()+'\n',encoding='utf8')
# Updated supplement front supersedes earlier whole-site block without erasing its Git history.
front=r'''\section{Numerical replay and accepted support sensitivity}
The additional support analysis distinguishes numerical agreement with an archived prediction from stability of a scored metric. Using unchanged relative and absolute tolerances of $2\times10^{-5}$, 16 of 36 Alice neural replays pass; all nine TCN and nine recurrent cases plus Sanyo Inverted43/44 remain outside tolerance. The maximum difference in the saved mixed-origin replay is 1.287 W, with maximum whole-output difference RMS 0.02032 W. On identical scored targets at one and twelve hours, across full, active and low-power ranges, the largest RMSE change is 0.000567 W and the largest MAE change is 0.000425 W. No compared historical-support mean ranking or Daily-effect direction changes. These small metric effects do not prove that the historical neural input state has been recovered.

For common labels and mask, $|\mathrm{RMSE}(p)-\mathrm{RMSE}(q)|\leq\sqrt{\mathrm{mean}(p-q)^2}$ and $|\mathrm{MAE}(p)-\mathrm{MAE}(q)|\leq\mathrm{mean}|p-q|$. The recorded bounds pass on the checked supports. They do not constrain newly restored origins; relative skill can also be sensitive when the reference error is close to zero.

Representative probes cover Qcells TCN44, Sanyo TCN42, Sanyo Inverted43/44 and Qcells Inverted42. Strict checkpoint loading has no missing or unexpected parameter keys. Models are in evaluation mode with float32 inputs and frozen target inverse transforms. Original versus mixed 256-origin batches do not resolve the selected Alice discrepancies; single-item batches can change their size. Neither the CPU nor a different batch output is assumed to be the truth. The current runtime is PyTorch 2.7.1+cu118, CUDA 11.8, cuDNN 90100 and RTX 3060 Laptop GPU, with deterministic cuDNN, cuDNN TF32 enabled and matrix TF32 disabled. Complete historical backend settings and the original separately saved neural KNN/IF/scaler state are unavailable. The current reconstruction uses the later saved Train-only Ridge processor. Its feature order and fitted-state summaries are recorded, without claiming identity with an unavailable historical object.

\subsection{Per-method acceptance}
Hanwha and Qcells include all three verified Inverted seeds in new-support primary contrasts; Sanyo includes seed42 separately, never an incomplete three-seed mean. Joint-patch is verified secondary evidence at all three arrays. Original/expanded Ridge A/B and deterministic references replay successfully. Other new-origin neural metrics are clearly marked diagnostic and excluded from accepted averages/rankings. Full accepted and diagnostic CSVs are separate. This method-specific scope supersedes the earlier all-site scoring block while preserving its original records and tolerance.

For each prefix, original eligibility requires all targets in that prefix finite and nonnegative. S1 requires each scored target nonnegative; S2 retains each finite raw value. New origins are relative to the corresponding prefix's original eligibility, not always H144. Inputs and nonnegative Daily/Last-value rules remain unchanged, and all compared methods share the same Daily point intersection. A label beyond Test is unscored but does not invalidate an available previous-day input. No full-window labels are imputed or missing rows joined together. Counts include origins, origin--lead pairs, unique physical target timestamps and active composition. Train support remains unchanged: Qcells neural Train/Validation counts are 12747/2999, distinct from Ridge's 12648/2977.

\section{Four-cell period and support definitions}
Both external periods use every calendar origin and historical/target buffers: October--December 2017 and April--June 2018. January 1 records provide the final twelve-hour target buffer for December 31 origins; they do not add a new origin month. Complete-H144 and pointwise-finite masks share the same candidate list, finite origin requirement and Daily point intersection. NIST remains fixed EST without DST; Yulara retains its provider-local coordinate and conservative availability shift. Finite negative external values remain eligible.

All 2018 predictions are reused. The 2017 supplemented candidates use the frozen 2017 processors and checkpoints with no fits. Archival-origin forward passes reproduce the original external predictions. A Yulara mixed-candidate batch probe has a maximum 0.03624 W difference outside the old tolerance for one seed. The explicit inference partition therefore preserves original chronological batching for newly computed archival origins, then computes additional origins separately. Both sets contain new forwards from the same checkpoint; historical saved predictions are not spliced into a replacement predictor. The failed mixed-batch diagnostic is retained. Original archive counts and metrics are reported separately from unified-buffer comparisons.

\subsection{Paired uncertainty and independent checks}
Forty-eight-hour nonoverlapping origin blocks are the primary scheme, with 24/72-hour sensitivity, 2000 draws and seed 20260908. Methods and actual neural seeds use the same draw within each comparison. SSE and point counts are summed before RMSE; neural-seed effects are averaged, not predictions. Periods are analyzed separately rather than pairing different calendar dates. Empty blocks and zero-support draws remain explicitly counted. These intervals condition on frozen predictions and the observed periods and do not remove all dependence or include retraining uncertainty.

The current saved-point audit independently reconstructs masks, time joins and metrics, while a separate block-count-matrix implementation recomputes interval effects without importing the production interval function. Earlier numerical audit totals remain historical. New archive verification tests file integrity and lightweight arithmetic, not complete training reproduction.

'''
(P/'supplementary_closeout.tex').write_text(front,encoding='utf8')
p=P/'supplementary.tex';t=p.read_text(encoding='utf8').replace('\\input{supplementary_fixed_period.tex}','\\input{supplementary_closeout.tex}');p.write_text(t,encoding='utf8')
p=P/'supplementary_diagnostic_figures.tex';t=p.read_text(encoding='utf8').replace('fixed_period_figures/','closeout_figures/')
for stem in moved:
 cap=(F/(stem+'_caption.txt')).read_text(encoding='utf8');alt=(F/(stem+'_alt.txt')).read_text(encoding='utf8')
 def esc(x):return x.replace('%','\\%').replace('_','\\_').replace('−','-')
 t+='\n\\clearpage\n\\begin{figure}[p]\\centering\n\\includegraphics[width=0.96\\textwidth]{closeout_figures/'+stem+'.pdf}\n\\caption{'+esc(cap)+'}\\par\\small\\textit{Alt text: '+esc(alt)+'}\\end{figure}\n'
p.write_text(t,encoding='utf8')
(P/'build_figures.py').write_text("from pathlib import Path\nimport runpy\nroot=Path(__file__).resolve().parents[2]\nrunpy.run_path(str(root/'GFNODE_experiments/scheme_A_support_closeout/build_figures.py'),run_name='__main__')\n",encoding='utf8')
(H/'DOCUMENT_COUNTS.json').write_text(json.dumps({'abstract_whitespace_words':len(abstract.split()),'main_figures':5,'supplement_figures':17,'method':'abstract source whitespace; no PDF full-text substitute'},indent=2),encoding='utf8')
print('Manuscript revised; abstract',len(abstract.split()),'words; five main figures')
