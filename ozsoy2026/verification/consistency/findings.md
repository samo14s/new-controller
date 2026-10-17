# Internal-consistency audit of Ozsoy, Sims & Ozturk (2026)

Written by the orchestrator from the structured report of the consistency
agent (the agent's harness refused `.md` files). Every number comes from the
scripts and outputs in this folder; data are the PDF's own vector paths
(`../../digitized`).

| item | finding | evidence |
|---|---|---|
| (a) damping | Fig. 5 "identified" curve is exactly Table 1 (free fit 129.307 Hz, 1.4499 %, 1.3401e7 N/m). Every *computed* curve fits 1.34 %: Fig. 6a uncontrolled FRF 1.3398 % (peak -111.104 dB), Fig. 6b open-loop pole 1.3405 %, Fig. 9 uncontrolled SLD 1.3394-1.3396 % (rms log-residual 0.00045; 1.45 % gives 0.063). | `a_damping.*` |
| (b) Fig. 6 labels | Fig. 6a and 6b agree with each other, but match the printed gain only at 1300 rpm (1162). The plotted curves/poles correspond to 1543 (label 1487), 685 (label 585) and 830 (label 797) V s/m. | `b_fig6.*` |
| (c) Fig. 9 vs Fig. 8 | The controlled curve is not the zero-order a_lim at the Fig. 8 gain: 2/250 speeds within 2 %, median deviation 23 %. It lies below the uncontrolled curve at 957.9-972.6 rpm and 1491.0-2048.7 rpm (by up to 25.9 mm at 1950 rpm: 3.24 vs 29.17 mm), i.e. below the model a_lim for every gain 0-2500. Two explanations tested as hypotheses, neither reproduces the curve: closed-loop lobe floor (median 7.9 %), force-limited depth (median 20-33 %). | `c_fig9_vs_fig8.*`, `c2_hypothesis_force_limited.*` |
| (d) white overlays | Six white fill+stroke rectangles are painted after the SLD curves and the legend. Five cover the controlled curve over 958.5-975.4 and 1498.5-2050.8 rpm (3.22-12.29 mm; 58 of 250 vertices, including its minimum 3.22 mm at 1940 rpm); the sixth sits in the legend. The covered intervals coincide within 5 rpm with the bands of item (c). No intent is inferred. | `d_white_shapes.*`, `d_hidden_controlled_points.csv` |
| (e) Fig. 8 straight runs | 37 of the 251 grid speeds carry no vertex; the curve is exactly linear over 1410-1860 rpm (46 speeds) and 620-740 rpm (13 speeds), plus shorter runs at 500-550 and 560-610 rpm. Hypothesis: interpolation between optimised speeds. The text's range 237-1487 V s/m differs from the plotted minimum 256.9 (1940 rpm); 237 is the single-gain value. | `e_fig8_linear_runs.*` |
| (f) Table 3 and worked example | Table 3 "proposed" exceeds the Fig. 9 curve by 0.24-0.51 mm at 1300, 2300, 2800, 3000 rpm. At 1300 rpm the paper gives four depths (10, 11.49, 11.8, 12.0 mm); the rejected gain 1240 (11.8 mm) and the accepted smaller gain (12.0 mm) conflict with a_lim increasing in g, which holds in the model at that speed. | `f_table3*` |
| (g) text figures | "about 13 times near 730 rpm": plotted ratio 12.15 at 730 rpm (max 12.65 at 710 rpm), consistent after rounding. "worst case 1.2 to 3.6 mm": uncontrolled minimum 1.212 mm; no plotted controlled minimum equals 3.6 mm (covered minimum 3.22, visible minimum 3.80). | `c_fig9_vs_fig8.*` |
