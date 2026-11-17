# Figures of the manuscript (Figs. 2-10)

All nine data figures are drawn by `make_figures.py` from the stored results of
the repository; nothing outside `paper/figures/` is written.

```bash
cd paper/figures
OMP_NUM_THREADS=1 python make_figures.py            # all nine, about 30 s
OMP_NUM_THREADS=1 python make_figures.py fig03 fig07   # selected figures
```

Outputs, for each figure: `figNN_<name>.pdf` (vector, fonts embedded as TrueType)
and `figNN_<name>.png` (300 dpi preview); plus `figure_numbers.json`, the numbers
quoted in `captions.tex`, computed from the same arrays that are plotted.
`captions.tex` holds a `figure` environment for each figure (label from
`paper/BRIEF.md`, caption, `% src:` line). Fig. 1 (`fig:model`, block diagram) is
drawn in TikZ by the integrator and is not here.

## Style

* Every figure is 190 mm wide (double column, final print size); heights 66-140 mm.
  With `width=\textwidth` the elsarticle preprint layout scales them by about 0.7.
* 8 pt Liberation Sans (Helvetica/Arial metrics), mathtext in the same font; lines
  0.8-1.2 pt; no titles; panel letters (a), (b), ... in the upper-left corner of each
  panel (just above the axes in Figs. 5, 6 and 9, where a row of "no chatter up to
  3 m/s" markers runs along the top of the axes); units in parentheses, (–) for ratios.
* Okabe-Ito colours, used the same way in every figure: DVF black; reformulated
  condition and delayed term blue (g2 = 1000 light, 2000 medium, 4000 dark);
  Eq. (2) of the published method grey; uncontrolled dashed grey; certificates C2
  orange and C3 green with open markers; simulations with filled markers; simulated
  step entry and the 9 V limit vermillion; simulated face entry reddish purple (the
  describing-function estimate of Fig. 5 uses the same colour).
* Legends sit outside the data area or in an empty corner.

## Data of each figure

| Figure (label) | File | Data read | Computed here |
|---|---|---|---|
| 2 (`fig:feas`) | `fig02_feasibility` | `results/gainmap_main.npz` (Eq. (2) as written: `g_best`, `a_best`), `results/feas_g1500.npz` (reformulated gains `g_best`), `results/feas_cert_g1500.npz` (`a_lim`, `a_op_linear` = 0.9 a_lim, `a_td_clipped`, `a_cert` = a_chk), `results/c1_map.npz` (`a_lim0`) | nothing |
| 3 (`fig:two`) | `fig03_two_outcomes` | `results/c1_basin.npz` (stored critical kick at the plotted phase, for the caption) | two runs of `src/tdsim.simulate` (each about 1.5 s): 2500 rpm, g = 1500 V s/m, a = 0.9 a_lim = 15.51 mm, \|V\| ≤ 9 V, face entry, full surface memory (`nreg = 0`), 400 revolutions, kick of 7.0 and 7.6 mm/s at step 100 x 512 (phase 0 of 16); outcome classified by `src/tdboundary.classify_run` over the 300 revolutions after the kick, as in `src/c1_unsafe_zone.py` |
| 4 (`fig:map`) | `fig04_entry_map` | `results/c1_map.npz` (`a_lim`, `a_lim0`, `a_step_clip`, `a_edge_clip`), `results/c1_band.npz` (`a_band_low`) | nothing |
| 5 (`fig:basin`) | `fig05_basin` | `results/c1_basin.npz` (`dv` per phase, clipped and unclipped; `p_mode`), `results/c1_df.npz` (`A_dual`) | the describing-function kick A_dual / (g p_mode), as in `src/figures_c1.py` |
| 6 (`fig:cert`) | `fig06_certificates` | `results/c2_validate.npz` (`cert`, `td_min`), `results/c3_basin.npz` (`cert`, `c2_cert`, `td_kick`) | C2 certificates rescaled to the closed-form command bound, factor (9 V - b_C3)/(9 V - b_C2) from `c2_certificate.orbit` and `c3_periodic.orbit` (as `src/figures_c2.py`, README 14.9); factors 0.9972-1 |
| 7 (`fig:surf`) | `fig07_certified_surface` | `results/c2_surface.npz` (`cert`, `u0`), `results/c2_step.npz` (`beta`), `results/c1_map.npz` (step-entry limit), `results/feas_g1500.npz` (gains) | the same rescaling (surface: with the stored `u0`, as `src/c3_summary.margin_errata`; factors 0.86-1) |
| 8 (`fig:dlim`) | `fig08_delayed_limits` | `results/c4_limits.npz` (`feas_a_lim`, `feas_g1`, `feas_names`, `feas_g2`), `results/c4_entry.npz` (`a_entry`, designs `feas_dvf` and `feas_d2000`) | nothing |
| 9 (`fig:dbasin`) | `fig09_delayed_basin` | `results/c4_gainmap.npz` (`kick`, `kick_dvf`, `bmax`, `bmax_dvf`), `results/c4_validate.npz` and `results/c4_validate2.npz` (`td_min`, `cert`, kinds `dvf`, `dvf_fine`, `comb`, `same_g1`) | gain-map certificates rescaled as in `src/c3_summary.margin_errata` (factors 0.59-1, below 0.99 only where the margin is almost nil); the validation statistics and the same-depth pairs of `src/c4_summary.py` |
| 10 (`fig:robust`) | `fig10_robustness` | `results/c4_robust.npz` (`eps_a_lim`, `eps_cmd`, `runout_cmd`, `tau_a_lim`, `a_lim`, `mode2`) | the runout at which the worst command reaches 9 V (linear interpolation, as `src/figures_c4.py`); the worst-case ratio a_lim(τ)/a_lim(0) over the four speeds and the three assumed modes |

Kick search, depth ladders, classifier and the models behind each file are those
of the campaign scripts (`src/c1_unsafe_zone.py`, `src/c2_certificate.py`,
`src/c3_periodic.py`, `src/c4_comb.py`, `src/c4_robust.py`) and README sections
8 and 11-14. Every stored campaign uses the damping ratio 1.34 %.

## Notes

* Fig. 9 shows the certificates of the gain map (best design at each depth); the
  validation ratios quoted in its caption (89 of 89, 1.60-7.32) use the
  strict-solver certificates of the simulated designs, rescaled in the same way.
* Fig. 10c is drawn at the computed delays only (0, 50, 100, 200, 300, 400 µs),
  with the markers offset horizontally by a few µs so that they do not cover each
  other; values between the computed delays are not known.
* The 300-revolution classifier biases simulated limits near a_lim up to 1.5 %
  high (4.4 % near the forced resonance; `verification/tdsim/check8`, `check10`);
  this affects the face-entry curves of Figs. 4 and 8, not the kick results.
