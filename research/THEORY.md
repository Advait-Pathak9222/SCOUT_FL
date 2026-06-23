# JEDI-FL / SCOUT-FL — Theoretical results (statements, proofs sketches, and how each is experimentally validated)

Notation: round $t$, global model $\mathbf w_t$, objective $F$, selected set $\mathcal S_t$ (budget $|\mathcal S_t|=K$),
step size $\eta$, smoothness $L$, stochastic-gradient variance bound $\sigma^2$, AirComp aggregation MSE
$\varepsilon_{\mathrm{agg}}(\mathcal S)=\sigma_z^2/(|\mathcal S|^2 P\min_{k\in\mathcal S} g_k)$, per-target Fisher matrix
$\mathbf J_{k}$, accumulated FIM $\mathbf J(\mathcal S)=\mathbf J_0+\sum_{k\in\mathcal S}\mathbf J_k$.

Each result lists the **validation hook** = the code that checks it numerically (so every claim is
experimentally reproducible, not asserted).

---

## P1 — Sensing log-det utility is monotone submodular
**Statement.** $f_{\mathrm{sense}}(\mathcal S)=\sum_m w_m[\log\det\mathbf J_m(\mathcal S)-\log\det\mathbf J_0]$ is monotone
non-decreasing and submodular for PSD $\mathbf J_k\succeq0$.
**Proof.** $\log\det$ is concave and operator-monotone on the PSD cone; for $A\preceq B$ and $X\succeq0$,
$\log\det(A+X)-\log\det A\ge\log\det(B+X)-\log\det B$ (Shamaiah–Banerjee–Vikalo, CDC 2010). Monotonicity:
adding a PSD term cannot decrease $\log\det$. Nonneg-weighted sum over targets preserves both. ∎
**Validation hook.** `analysis/verify_submodularity.verify_submodular` on `SensingUtility.value` → 0 violations
(`tests/test_objectives`, `tests/test_joint_information::test_decoupled_jedi_is_near_submodular`).

## P2 — Coverage and learning utilities are submodular; fairness is modular
$f_{\mathrm{cov}}$ (concave-saturating coverage of an aging map) and $f_{\mathrm{learn}}$ (facility-location over
gradient embeddings, DivFL / Balakrishnan et al. ICLR 2022) are monotone submodular; the participation/age
term is modular. **Validation:** same `verify_submodular` harness per utility.

## P4 — Greedy guarantee for the composite / JEDI objective
**Statement.** A nonneg-weighted sum of monotone submodular functions is monotone submodular, so greedy gives
$(1-1/e)$ (Nemhauser–Wolsey–Fisher 1978). For the **coupled** JEDI objective the AirComp factor
$\kappa(\varepsilon_{\mathrm{agg}}(\mathcal S))$ multiplies the learning block and breaks exact submodularity;
greedy then retains the weak-submodular guarantee $(1-e^{-\gamma})$ with submodularity ratio $\gamma$ (Das & Kempe 2011).
**Validation hook.** `analysis/verify_submodularity.submodularity_ratio` reports $\gamma_{\min}$: decoupled
$\gamma\approx1$ (submodular), coupled $\gamma\approx1$ empirically (mild coupling) ⇒ $(1-1/e)$ effectively holds
(`tests/test_joint_information`).

## P5 — CRB (A-optimal) is weakly submodular
$\mathrm{tr}\,\mathbf J(\mathcal S)^{-1}$ reduction is only weakly submodular; greedy gets $(1-e^{-\gamma})$ with
$\gamma$ characterizable in low-SNR regimes. Reported via the same `submodularity_ratio` on the CRB reduction.

## P6 — Convergence: AirComp MSE + selection control the per-round descent  *(the wireless-learning link)*
**Assumptions.** (A1) $L$-smooth $F$; (A2) unbiased local gradients, variance $\le\sigma^2$; (A3) unbiased AirComp
aggregate with $\mathbb E\|\hat{\mathbf g}_t-\bar{\mathbf g}_t\|^2\le\varepsilon_{\mathrm{agg}}(\mathcal S_t)$;
(A4, optional) PL constant $\mu$.
**Lemma (one-round descent).** For $\eta\le 1/L$,
$$\mathbb E[F(\mathbf w_{t+1})]-F(\mathbf w_t)\;\le\;-\eta\big(1-\tfrac{L\eta}{2}\big)\|\nabla F(\mathbf w_t)\|^2
\;+\;\tfrac{L\eta^2}{2}\Big(\tfrac{\sigma^2}{|\mathcal S_t|}+\varepsilon_{\mathrm{agg}}(\mathcal S_t)\Big).$$
**Proof.** Descent lemma (A1) on $\mathbf w_{t+1}=\mathbf w_t-\eta\hat{\mathbf g}_t$; take expectation; use
unbiasedness (A2,A3) and the variance decomposition $\mathbb E\|\hat{\mathbf g}_t\|^2=\|\nabla F\|^2+\sigma^2/|\mathcal S_t|+\varepsilon_{\mathrm{agg}}$. ∎
**Corollary (non-convex rate).** $\frac1T\sum_t\mathbb E\|\nabla F\|^2\le\frac{2(F_0-F^\star)}{\eta T}+L\eta\,\overline{(\sigma^2/|\mathcal S_t|+\varepsilon_{\mathrm{agg}})}$.
**Corollary (PL floor).** $\mathbb E[F(\mathbf w_T)]-F^\star\le(1-\mu\eta)^T(F_0-F^\star)+\frac{L\eta}{2\mu}\,\overline{(\sigma^2/|\mathcal S_t|+\varepsilon_{\mathrm{agg}})}$ — the error floor is set by the AirComp MSE.
**Consequence.** Selecting $\mathcal S_t$ to *raise* the gradient utility while *lowering* $\varepsilon_{\mathrm{agg}}$ (exactly JEDI's two coupled terms) maximizes expected descent.
**Validation hook.** `analysis/convergence.py` fits $\Delta_t=\beta_0+\beta_1\,\|\hat{\mathbf g}_t\|^2+\beta_2\,\varepsilon_{\mathrm{agg},t}$
over all logged rounds, where $\Delta_t=F(\mathbf w_t)-F(\mathbf w_{t+1})$ (test-loss decrease). The bound predicts
$\beta_1>0$ and $\beta_2<0$ (significant). Reported with CIs + $R^2$ + partial-regression plots.
**Citations.** Zhu–Wang–Huang (TWC 2020, broadband analog aggregation); Cao–Zhu–Xu–Huang (TWC 2020, power control,
arXiv 2106.09316); Amiri–Gündüz (TSP 2020); Yang–Fang–Liu partial participation (ICLR 2021, arXiv 2101.11203);
Li et al. FedAvg non-IID (ICLR 2020); Karimi–Nutini–Schmidt PL (ECML 2016).

## P3-dual — Primal-dual feasibility / bounded violation
**Statement.** With dual ascent $\mu^{(t+1)}=[\mu^{(t)}+\eta_d(c(\mathcal S_t)-\varepsilon)]_+$ on the AirComp-MSE
constraint (SCOUT-v2) and the participation virtual queue $q_k^{(t+1)}=[q_k^{(t)}+\eta_d(\text{target}-\mathbb 1[k\in\mathcal S_t])]_+$
(JEDI fairness), the time-averaged constraint violation is bounded and $\to0$:
$\frac1T\sum_t (c(\mathcal S_t)-\varepsilon)_+\le \mathcal O(1/\sqrt T)$ (standard online-convex / Lyapunov
drift-plus-penalty argument; Neely 2010). I.e. constraints are satisfied **on average** without fixed weights.
**Validation hook.** `analysis/feasibility.py` reads the logged dual trajectory `dual_mse` and `mse_violation`
(SCOUT-v2) and the participation deficit `jedi_deficit_*` (JEDI), and checks the running-average violation
decays and the dual stays bounded (plot + slope test).

## P7 — Online regret of bandit selection (unknown FIM / channel)
**Statement.** When per-client utilities (sensing FIM / channel) are unknown and learned online, CUCB-style
selection with the greedy $(1-1/e)$-oracle achieves sublinear **$\alpha$-regret**
$R_\alpha(T)=\sum_t[\alpha\,U(\mathcal S^\star_t)-U(\mathcal S_t)]=\tilde{\mathcal O}(\sqrt{NKT})$, $\alpha=1-1/e$
(Chen–Wang–Yuan CMAB, ICML 2013 / JMLR 2016; Wang–Chen NeurIPS 2017; Streeter–Golovin NeurIPS 2008 for the
full-bandit $\tilde{\mathcal O}(T^{2/3})$ variant).
**Validation hook.** `selection/online.py` runs the UCB selector; `analysis/regret.py` computes
$R(T)=\sum_t[\alpha U(\mathcal S^\star_t)-U(\mathcal S_t)]$ vs the offline greedy oracle on the true utilities,
and certifies sublinearity by (i) $R(T)/T\to0$ and (ii) a log-log fit $R(T)\sim cT^p$ with $\hat p<1$.

---

**Theory floor for TWC (per the plan §G):** P1, P4, P6 are mandatory and are all stated above with proofs and
numerical validation hooks; P5/P3-dual/P7 strengthen the wireless-optimization and online story.
