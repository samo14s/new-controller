# Verification of the SADE optimiser (src/sade.py)

This folder checks `src/sade.py`, the self-adaptive differential evolution used for the
per-speed DVF gain search in the reproduction of Ozsoy, Sims & Ozturk (JMP 171, 2026).
There are three parts:

1. a line-by-line comparison with SaDE as defined by Qin, Huang & Suganthan (IEEE TEVC 13(2), 2009);
2. standard benchmarks against `scipy.optimize.differential_evolution`, an independent
   textbook SaDE, and brute force;
3. the real objective `fitness(rpm, g, 'force')` at 1000, 1950 and 2500 rpm, compared with a
   brute-force optimum, and a settings study.

Nothing outside this folder was modified.

## How to run

```
cd /home/user/new-controller/ozsoy2026/verification/sade
python3 bruteforce.py                 # ~40 s, Pool(2): reference optimum, writes bruteforce*.{csv,json}
python3 sade_real.py --tag A_strat --init strat                  # ~2 min, task baseline
python3 sade_real.py --tag B_uniform --init uniform              # ~1.5 min
python3 sade_real.py --tag C_rec_w25 --init strat --window 25    # recommended setting
python3 sade_real.py --tag C_rec_w25_seeds6-10 --init strat --window 25 --seeds 6 7 8 9 10
python3 surrogate_study.py            # ~2.5 min, 60 seeds x 14 settings x 3 speeds (tabulated objective)
python3 surrogate_extra.py            # ~30 s, surrogate fidelity + best-of-3 statistics
python3 benchmarks.py                 # ~6 min (pure Python, cheap functions)
```

| file | content |
|---|---|
| `common.py` | Path setup and `CachedFitness`. `CachedFitness` memoises `objective.fitness` with g rounded to 0.01 V s/m, and stores the cache in `cache_force_<rpm>.pkl`. The rounding moves the feasibility edge by at most 0.005 V s/m, which is about 1e-4 % in a. |
| `ref_sade.py` | Independent SaDE written from the 2009 paper (SUS strategy assignment, unclipped F, CR regenerated into [0,1], LP selectable). It shares no code with `src/sade.py`. |
| `bruteforce.py` | Grid search at step 5 V s/m over [0, 3000], then step 1 up to the last feasible point, then step 0.1 around the best local maxima, then bisection on the feasibility edge to 0.01 V s/m. |
| `sade_real.py` | Runs `src/sade.py` on the real objective. |
| `surrogate_study.py`, `surrogate_extra.py` | Runs the settings study on a tabulated copy of the real objective (see sec. 3.3). |
| `benchmarks.py` | Benchmark suite. |

---

## 0. Which SaDE does the paper cite?

In the paper's reference list, **[56] is Worden & Manson, MSSP 29 (2012) 201-212**, a paper on
hysteretic system identification that uses SADE. It is not Qin et al. 2009. Storn & Price
(1997) is [58]. I could not obtain the full text of [56], so I cannot confirm which SaDE
variant it uses: Qin & Suganthan 2005 (CEC) has two strategies, while Qin et al. 2009 has four.
As the task asks, the comparison below is against Qin et al. 2009. On a 1-D problem the
differences between the two variants are immaterial for the reasons given in section 1:
crossover is inert in 1-D, and all strategies reduce to scaled difference moves.

## 1. Logic compared with Qin, Huang & Suganthan (2009)

| # | element | SaDE 2009 | `src/sade.py` | verdict / matters in 1-D? |
|---|---|---|---|---|
| 1 | Strategy pool | rand/1/bin, rand-to-best/2/bin, rand/2/bin, current-to-rand/1. For current-to-rand/1, K ~ U(0,1) and there is no crossover. | Same four formulas (l.51-67). The five indices are mutually distinct and different from i. current-to-rand skips crossover. | **match** |
| 2 | Assigning strategies to individuals | Stochastic universal sampling with probabilities p_k | Independent roulette draws (`rng.choice(K, NP, p)`) | Minor deviation. The per-strategy counts have higher variance. No effect seen (src and the SUS reference give identical success rates on every benchmark). |
| 3 | Learning period LP | LP = 50 generations in the 2009 experiments | Default `lp=5`. `gainmap.run_sade` also uses 5. | Deviation. With a 50-60 generation budget, LP = 50 would leave adaptation almost unused. **Does not matter in 1-D**: in the surrogate study, LP = 20 gives the same success rates as LP = 5, and on the benchmarks the reference SaDE with LP = 5 and LP = 50 behaves the same. |
| 4 | Strategy probability update | p_k = S_k / sum S_k, with S_k = ns_k/(ns_k+nf_k) + 0.01, summed over the last LP generations. Equal p_k = 1/4 before LP. | Same (l.86-94). An unused strategy (ns+nf = 0) gets S = 0.01 instead of 0/0. | **match**. The guard is a sensible completion of the formula. |
| 5 | CR | CR ~ N(CRm_k, 0.1), CRm_k starts at 0.5. CRm_k = median of the successful CR values of strategy k over LP generations. Out-of-range draws are regenerated. | Same, except out-of-range CR values are **clipped** to [0, 1]. CR values of current-to-rand/1 are recorded although that strategy does not use them. | Deviation. **Irrelevant in 1-D**: with D = 1, the j_rand rule (`mask[rng.integers(d)] = True`) always copies the mutant, so the trial equals the mutant whatever CR is. CR adaptation and binomial crossover are both inert for the gain search. |
| 6 | F | F_i ~ N(0.5, 0.3) per individual, not adapted, not truncated | Same, but **clipped to [0.05, 2]**. About 6.7 % of draws (F < 0.05) are set to 0.05. | Minor deviation. For rand/1 and rand/2, a negative F is equivalent to swapping the difference pair, so clipping only shifts a little probability to small steps. No measurable effect. The paper's Step 4 says F is "updated". In SaDE-2009 only CR is learned and F is resampled, and src follows SaDE-2009. |
| 7 | Selection and success counting | Greedy, ties go to the trial (f(u) <= f(x) when minimising). A trial that enters the next generation counts as a success. | `fu >= fit[i]` when maximising (same rule) | **match** with DE/SaDE. The paper's wording ("higher fitness") would be strict. On Eq. (2), a zero-fitness parent accepts any zero-fitness trial and counts it as a success. Infeasible members therefore random-walk, and the strategy statistics carry almost no information on this objective. This is harmless (an all-infeasible start still finds the feasible set), but it means the "self-adaptive" part contributes little here. |
| 8 | Bound handling | Violated components are re-initialised at random inside the range (common SaDE practice) | Same (l.68-69) | **match**. At 1950 rpm the feasible set [0, 67.5] touches the lower bound, so a mutant that crosses 0 is thrown uniformly over [0, 3000] and is about 98 % likely to be infeasible. This is a small inefficiency, and no failure was traced to it. |
| 9 | Generation scheme | Generational: trials are compared with parents, and the best is fixed during a generation | Same (`new_pop`, `ib` set at the start of each generation) | **match** |
| 10 | Initialisation | Uniform random | Uniform random, or a user `init`. `gainmap.run_sade` passes a stratified uniform sample (one point in each NP-th of G). | Allowed variant. Both are "uniform over G". Stratification is more reliable here (sec. 3). |
| 11 | Termination | Fixed budget of function evaluations | `max_gen`, or a stagnation window: stop if the best fitness improved by <= tol over the last `window` generations. This follows the paper's Step 5. Default `window=8`; `gainmap` uses 12. | Deviation required by the paper. **This is the only element that matters in practice** (sec. 2 and 3). It is also the only route to an f = 0 result: if no member is feasible for `window` generations, the zero plateau counts as converged. |
| 12 | NP | 50 in the 2009 experiments | Default 20 | The paper does not state NP. 20 is adequate (sec. 3). |

The code has no defect that changes results. The deviations are (a) clipping of F and CR,
(b) roulette instead of SUS, (c) LP = 5 instead of 50, and (d) the stagnation stop. Only (d)
affects the 1-D gain search.

## 2. Benchmarks (`benchmarks.py`, `benchmarks.csv`)

All methods get the same budget, NP x (G+1) evaluations, and 20 seeds each.

| setting | value |
|---|---|
| 1-D cases | NP = 20, G = 60 (1220 evaluations) |
| 2-D cases | NP = 20, G = 300 |
| 5-D cases | NP = 50, G = 600 |
| Success, smooth functions | f - f* <= 1e-4 |
| Success, cliff functions | >= 99 % of the maximum, as in the real task |

The methods compared:

| column | method |
|---|---|
| src | `src/sade.py` with the window disabled |
| src W=15 | `src/sade.py`, window = 15, tol = 1e-8 |
| ref LP5, ref LP50 | the independent SaDE |
| scipy | `differential_evolution` (best1bin, popsize = NP/d, `tol = atol = 0`, `polish = False`) |
| grid | a cell-centred grid with the same budget |
| grid+loc | the grid followed by a bounded scalar search in the best cell |

The three cliff functions mimic Eq. (2):

| function | feasible set | location of the maximum |
|---|---|---|
| `cliff_edge` | [0, 205.55] | at the edge |
| `narrow_low` | 2.3 % of the box, touching the lower bound | at the edge |
| `islands` | 16 islands | on the edge of the last island |

| case | src | src W=15 | ref LP5 | ref LP50 | scipy | grid | grid+loc |
|---|---|---|---|---|---|---|---|
| Rastrigin 1-D | 20/20 | 19/20 | 20/20 | 20/20 | 19/20 | fail | pass |
| Schwefel 1-D | 20/20 | 19/20 | 20/20 | 20/20 | 20/20 | fail | pass |
| Gramacy-Lee | 20/20 | 19/20 | 20/20 | 20/20 | 20/20 | pass | pass |
| Shubert 1-D | 20/20 | 13/20 | 20/20 | 20/20 | 20/20 | fail | pass |
| cliff_edge | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | pass | pass |
| narrow_low | 20/20 | 20/20 | 20/20 | 20/20 | 15/20 | pass | pass |
| islands | 20/20 | 20/20 | 20/20 | 20/20 | 20/20 | pass | pass |
| Rastrigin 2-D | 20/20 | 14/20 | 20/20 | 20/20 | 16/20 | pass* | pass* |
| Rosenbrock 2-D | 20/20 | 9/20 | 20/20 | 20/20 | 20/20 | fail | fail |
| Rastrigin 5-D | 20/20 | 7/20 | 20/20 | 20/20 | 14/20 | pass* | pass* |
| Rosenbrock 5-D | 20/20 | 14/20 | 20/20 | 20/20 | 20/20 | fail | fail |

\* The cell-centred grid has an odd count per axis, so it contains the origin, which is the
Rastrigin optimum. That result is trivial.

What the benchmarks show:

* With the full budget, `src/sade.py` solves every case in every seed. It behaves exactly like
  the independent textbook SaDE, whether that uses LP = 5 or LP = 50. On Rastrigin it is more
  reliable than scipy's best1bin.
* The scipy failures on `narrow_low` come from its stopping rule, not from DE itself. When
  every member of the population has fitness 0, std(energies) = 0 <= atol, and scipy stops at once.
* The stagnation window is what loses runs. With window = 15, `src/sade.py` stops early in up
  to 13/20 seeds on smooth multimodal functions (it used 900 of 1220 evaluations on average in 1-D).

## 3. Real objective `fitness(rpm, g, 'force')`, g in [0, 3000] V s/m, damping 1.34 %

### 3.1 Brute-force reference (`bruteforce.json`, `bruteforce_<rpm>.csv`)

| rpm | a at g = 0 (mm) | feasible set (1 V s/m grid) | share of G | g_opt (V s/m) | a_opt (mm) | best on the step-5 grid |
|---|---|---|---|---|---|---|
| 1000 | 3.488 | one interval [0, 205.55] | 6.9 % | 205.55 | 9.1488 | 205 / 9.1324 |
| 1950 | 29.157 | one interval [0, 67.49] | 2.3 % | 67.49 | 29.4966 | 65 / 29.4839 |
| 2500 | 1.238 | one interval [0, 533.24] | 17.8 % | 533.24 | 6.1960 | 530 / 6.1636 |

At all three speeds, fitness rises monotonically with g on a single feasible interval. The
optimum is the cliff where peak |F_d| reaches 27 N. Near the cliff, F_d is not strictly
monotone in g at the 0.01 N level: at 2500 rpm it is 26.989 N at 532.6 and 26.988 N at 533.1.
The edge is therefore defined to about ±0.3 V s/m.

A 1 % tolerance in a corresponds to the following tolerance in g:

| rpm | Δg for 1 % in a (V s/m) |
|---|---|
| 1000 | 3.1 |
| 1950 | 58 |
| 2500 | 6.2 |

**At 1950 rpm the 1 % test in a is almost vacuous**: even g = 0 is only 1.15 % below the
optimum. A second criterion is therefore reported, |g - g_opt| <= 2 % of g_opt, which is what
matters for the gain map.

### 3.2 SADE on the real objective (`sade_real_*.csv/json`, 5 seeds per speed)

| setting (NP = 20, LP = 5, tol = 1e-4 mm) | 1000: a within 1 % | 1950: a within 1 % | 1950: g within 2 % | 2500: a within 1 % |
|---|---|---|---|---|
| A: G = 60, W = 15, stratified init (task baseline, `gainmap` init) | 5/5 | 5/5 | 3/5 (g errors -9.0, -7.9) | 5/5 |
| B: G = 60, W = 15, uniform init (`sade` default) | 5/5 | 5/5 | 4/5 (-4.9) | 5/5 |
| C: G = 60, W = 25, stratified, seeds 1-5 | 5/5 | 5/5 | 5/5 | 5/5 |
| C: same, fresh seeds 6-10 | 5/5 | 5/5 | 5/5 (max error 0.2) | 5/5 |

* At 1000 and 2500 rpm, every run ends within 0.01 V s/m of the brute-force edge, and a matches to 5 digits.
* In B, none of the five 1950 rpm runs had a feasible member at start; the first feasible
  point appeared at generations 1-11. Two of the 1000 rpm runs also started infeasible.
* The two 1950 rpm failures in A both stopped 15 generations after their last improvement
  (at generations 24 and 30). Both still had a within 0.16 %.

### 3.3 Settings study on a tabulated objective (`surrogate_study.*`, 60 seeds per cell)

The surrogate is a_lim interpolated from the brute-force table for g <= g_opt, and 0 above it.
It is faithful to the real objective. With the seeds and initial populations of run A, it
reproduces the real runs: at 1950 rpm, seeds 1, 3 and 4 end at the same g (58.486, 67.223,
59.573) and the same generation, and at 1000 and 2500 rpm the final g agrees to 0.01 V s/m
(`surrogate_extra.json`).

| variant (others as baseline: NP 20, G 60, W 15, LP 5, stratified) | 1000 a<=1 % | 1950 a<=1 % | 1950 g<=2 % | 1950 f=0 | 2500 a<=1 % | evaluations (1000/1950/2500) |
|---|---|---|---|---|---|---|
| baseline | 59 | 60 | 42 | 0 | 60 | 983 / 784 / 823 |
| uniform init | 57 | 58 | 35 | 2 | 60 | 896 / 759 / 781 |
| stratified + g = 0 member | 58 | 60 | 44 | 0 | 60 | 924 / 804 / 837 |
| window 8 | 38 | 59 | 15 | 1 | 59 | 515 / 341 / 571 |
| **window 25** | **60** | **60** | **60** | 0 | **60** | 1180 / 1164 / 1048 |
| no window (fixed 60 gens) | 60 | 60 | 60 | 0 | 60 | 1220 each |
| uniform init, no window | 60 | 60 | 60 | 0 | 60 | 1220 each |
| NP 10 | 58 | 52 | 25 | 8 | 60 | 499 / 326 / 443 |
| NP 30 | 60 | 60 | 35 | 0 | 60 | 1412 / 1102 / 1230 |
| G 100 | 59 | 60 | 42 | 0 | 60 | same as baseline (the window ends runs first) |
| LP 20 | 60 | 60 | 37 | 0 | 60 | 991 / 735 / 859 |
| tol 1e-6 | 59 | 60 | 42 | 0 | 60 | 1117 / 819 / 859 |
| stratified + g = 0, G 100, W 25 | 60 | 60 | 59 | 0 | 60 | 1226 / 1273 / 1057 |

Best of 3 seeds, which is what `run_gain_map.py` keeps (20 groups of 3 seeds, `surrogate_extra.json`):

| setting | single run, 1000 rpm a<=1 % | single run, 1950 rpm g<=2 % | best of 3: a<=1 % (all speeds) | best of 3: 1950 rpm g<=2 % |
|---|---|---|---|---|
| `gainmap.run_sade` as coded (G 50, W 12) | 52/60 | 30/60 | 20/20 | 19/20 |
| task baseline (G 60, W 15) | 59/60 | 42/60 | 20/20 | 20/20 |
| recommended (G 60, W 25) | 60/60 | 60/60 | 20/20 | 20/20 |

### 3.4 Answer and recommendation

With NP = 20, G = 60, W = 15:

* A single run reaches within 1 % of the brute-force a_lim in 15/15 real runs and in
  179/180 surrogate runs. The one miss is at 1000 rpm, where the optimum is on the cliff.
* At 1950 rpm, a single run gets the **gain** within 2 % in only 60-70 % of runs (3/5 real,
  42/60 surrogate). The fitness is flat in g there, so the 15-generation window ends the run
  before the population reaches the edge.
* With uniform initialisation, 3 % of runs at 1950 rpm end with f = 0 (no feasible member found
  before the window expired).

Recommended settings. These stay within the paper's description: uniform sampling over G,
DE variation from the strategy pool, SaDE adaptation, and a sliding-window or max-generation stop.

* **NP = 20.** NP = 10 fails (8/60 runs with f = 0 at 1950 rpm). NP = 30 costs 40 % more and
  gains nothing.
* **max_gen = 60, stagnation window >= 25 generations, tol = 1e-4 mm.** Alternatively, run all
  60 generations. Either way, 60/60 runs per speed hit both criteria, at about 1050-1220
  evaluations, roughly 25-30 s per run on this machine. This is consistent with the paper's
  remark that the generation count was chosen so that the best fitness had stabilised before
  termination. The short windows in the code are too aggressive: 8 is the `sade()` default and
  12 is used in `gainmap`.
* **Stratified (Latin-hypercube) uniform initial population**, as `gainmap.run_sade` already
  does. Optionally make one member g = 0. The uncontrolled cut has zero demand, so this member is
  always feasible and rules out an f = 0 exit.
* **Do not start the stagnation counter while the best fitness is still 0.** This needs a change
  in `src/sade.py`, so here it is only recommended.
* **Keep LP = 5.** LP has no measurable effect in 1-D.
* **Keep 3 seeds per speed and take the best**, as `run_gain_map.py` does. With W >= 25 this is
  belt-and-braces. With W = 12 it is what currently rescues the 1000 rpm cliff and the 1950 rpm gain.
* Every optimum found lies on the feasibility edge. `gainmap.reference` (grid plus bisection on
  the edge) is therefore an exact and cheaper cross-check for every speed, and should be kept
  alongside SADE.

## 4. Compute used

| item | wall time |
|---|---|
| Real objective (about 25 ms per evaluation, Pool(2)) | about 6 min in total: brute force 41 s, A 118 s, B 95 s, C 16 s + 84 s |
| Benchmarks (cheap functions, Python overhead only) | 6.3 min |
| Surrogate studies | 3 min |
