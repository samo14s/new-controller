# verification/targets — numerical targets from the paper

Ozsoy, Sims & Ozturk (2026), *J. Manufacturing Processes* 171:247–257. This folder lists every number
the reproduction has to be compared against. Each entry has its page, where it appears in the paper and a short
paraphrase of its context. No prose from the paper is copied (CC BY-NC-ND), and the raster images taken from the PDF
are written only to the session scratchpad.

Page convention: journal page J is PDF page J−245, i.e. pymupdf index J−246. PDF page 1 is the repository cover.

## Main outputs

| file | content |
|---|---|
| `targets.json` | 99 target entries (tables, equations, numbers in the text, Fig. 6 legend, values read from figures), 21 experimental test points, 3 chatter frequencies and 18 ambiguities |
| `targets.md` | the same content as readable tables |
| `experimental_points.csv` | test points from Figs. 9 and 11: control state, stable/chatter, rpm, depth, read position, uncertainty, caption check, published boundary at that speed, and whether the outcome is on the predicted side |
| `fig9_markers.csv`, `fig9_curves.csv`, `fig9_masks.csv`, `fig9_features.json` | Fig. 9 from the vector paths: markers, both SLD curves (with a `visible_in_print` flag), the white masking rectangles, and the curve minima, peaks, ratios and values at chosen speeds |
| `fig11_markers.csv`, `fig11_curves.csv`, `fig11_features.json`, `fig11_calibration_check.json` | Fig. 11 from the raster image: markers with uncertainty, column read-out of both curves, minima and peaks, and a calibration check against Fig. 9 |
| `fig8_gain_map.csv`, `fig8_features.json`, `fig8_segments.csv` | Fig. 8 gain map from the vector paths: values at chosen speeds, min and max, and stretches that are straight lines |
| `fig6b_poles.csv`, `frf_features.json` | Fig. 6(b) pole positions; FRF peaks of Fig. 5 and Fig. 6(a) |
| `fft_peaks.csv`, `fft_panels.json` | spectral lines of the FFT panels in Figs. 10 and 12, each labelled as a spindle harmonic or a non-synchronous line |
| `fig6a_peak_check.json`, `fig6b_pole_check.json`, `context_minima.json` | consistency checks used to judge the ambiguities. They use a 1-DOF model with Eq. (3) and zeta = 1.34 %, and the shared ZOA code in `../../src` |
| `targets_overview.png` | my own plot of the digitised data, not a copy of the paper's figures |

To regenerate the files, run the scripts in this order:
`extract_vector_figs.py`, `extract_fig11_raster.py`, `extract_fft_peaks.py`, `check_fig11_vs_fig9.py`,
`check_fig6b_poles.py`, `check_fig6a_peaks.py`, `context_minima.py`, `build_targets.py`, `plot_overview.py`.
Each takes a few seconds.

## How the figures were read and how accurate it is

- **Vector figures (Figs. 5, 6a, 6b, 8, 9).** The axis maps are linear least-squares fits to the tick-mark positions.
  The tick marks were detected again in the drawing list, and for Fig. 9 they match the nominal positions to 0.01 pt.
  Largest fit residuals: Fig. 9 0.045 rpm and 0.0009 mm; Fig. 8 0.044 rpm and 0.065 V s/m; Fig. 6(b) 0.07 rad/s.
  - Fig. 9 marker centres come out at 700.0, 760.0, 1200.0, 1300.0 and 2800.0 rpm (±0.05) and at depths of 0.5, 1.5,
    2.0, 4.0, 4.5, 5.5, 8.0 and 10.0 mm (±0.002).
  - Labels A and B are one glyph path, separated by a vertical gap. B is next to the red x at 1300/10 and A is next to
    the green circle at 1300/8. This matches the Fig. 10 caption.
- **Fig. 11 (JPEG, 1035×653 px).**
  - The axes are calibrated from the light-grey grid lines, located to sub-pixel accuracy: 2.894 rpm/px and
    0.0596 mm/px. Fit residuals are 1.2 rpm and 0.01 mm.
  - Markers are found by colour thresholding with connected components. The '+' and the diamond touch each other
    and are split by their row extent. Quoted uncertainty is ±4 rpm and ±0.07 mm.
  - Calibration check: Figs. 9 and 11 show the same uncontrolled SLD. The raster dashed curve differs from the
    Fig. 9 vector curve by a median of 0.023 mm (95th percentile 0.049 mm), with no rpm shift.
  - The cusp peaks of the dashed curve are lost in the dash gaps, so use the Fig. 9 vector peaks for it.
- **FFT panels (Figs. 10, 12).** Each panel is mapped from 0 to 400 Hz between its frame lines, about 1.24–1.29 Hz/px.
  Spectral lines are about 3 px wide. Synchronous lines fall within 0.3–2 Hz of k·f_spindle.

## Main targets

- **Tables 1–3:** every row. Table 3 has 9 speeds with clipped-DVF and proposed a_lim values.
- **Worked example (p.249):** 1300 rpm, g = 1240 V s/m → ZOA a_lim = 11.8 mm and time-domain peak demand 29 N.
  The candidate is infeasible because 29 N > 27 N.
- **Gain map:** the text gives 237–1487 V s/m. Read from Fig. 8: max 1487.0 at 740 rpm and min 256.9 at 1940 rpm.
  Fig. 8 values: 1479 at 500 rpm, 1158 at 1300, 543 at 2500, 794.5 at 3000 (all V s/m).
  Fig. 6 legend: 1487 at 500 rpm, 1162 at 1300, 585 at 2500, 797 at 3000.
- **Single gain:** 237 V s/m. The Fig. 11 controlled minima are 3.52 mm in every lobe. Its peaks are 8.65 mm at
  492 rpm, 10.91 at 657, 15.63 at 987 and 31.79 at 1971.
- **Uncontrolled SLD (Fig. 9, vector):** every lobe minimum is 1.212 mm. Peaks are 6.78 mm at 488 rpm, 8.99 at 651,
  13.64 at 977 and 29.27 at 1952.
- **Mapped-gain SLD (Fig. 9):** values at 500/560/600/800/1300/2300/2500/2800/3000 rpm are
  14.87/7.98/11.70/10.77/11.49/4.66/5.27/6.34/6.86 mm. The highest point is 15.44 mm at 740 rpm. The largest
  mapped/uncontrolled ratio is 12.65 at 710 rpm, and the ratio is 12.15 at 730 rpm. The text's "13x" is this ratio,
  rounded.
- **Headline numbers from the text:** 13x near 730 rpm; worst case 1.2 → 3.6 mm; 3x to 13x; at 1300 rpm 10 mm
  (mapped) vs about 3 mm (single gain); Table 3 discussion numbers 15.2/14.9, 12.1/12.0, 4.7→5.3, 6.2→7.1 mm and
  10–15 %.
- **Fig. 6(b) poles** (upper half plane, rad/s):

  | case | pole | zeta |
  |---|---|---|
  | open loop | −10.89 + 812.3j | 1.341 % |
  | g = 1487 | −125.35 + 800.3j | |
  | g = 1162 | −97.03 + 805.4j | |
  | g = 585 | −61.68 + 808.8j | |
  | g = 797 | −72.48 + 808.8j | |

  The actuator-mode poles are at about −7.4 to −7.7 + 53.1j.
- **Fig. 6(a) peaks:**

  | case | peak |
  |---|---|
  | uncontrolled | −111.10 dB at 129.0 Hz |
  | g = 1487 | −132.20 dB |
  | g = 1162 | −130.03 dB |
  | g = 585 | −126.14 dB |
  | g = 797 | −127.52 dB |

  Fig. 5 identified peak: 2.574e−6 m/N.
- **Experimental points.** Fig. 9 has 13 and Fig. 11 has 8. All captions check out:
  - Fig. 10: A = 1300 rpm / 8 mm stable and B = 1300 / 10 mm chatter, both mapped gain.
  - Fig. 12: A = 2800 / 0.5 mm stable and B = 2800 / 1.5 mm chatter, both uncontrolled; C = 2700 / 3 mm stable and
    D = 2700 / 4 mm chatter, both single gain.
- **Chatter frequencies:** about 149.5 Hz for Fig. 10 B, 133.3 Hz for Fig. 12 B and 138.1 Hz for Fig. 12 D (±2 Hz).

## Ambiguities (details in `targets.md`)

1. **Damping:** Table 1 gives 1.45 %, and the Fig. 5 identified curve agrees. Figs. 6(a) and 6(b) and the
   uncontrolled SLD all give 1.34 %.
2. **Gain-map minimum:** the text says 237, but Fig. 8 gives 256.9. 237 is the single-gain value.
3. **Fig. 6 legend gains:** for 500, 2500 and 3000 rpm they match neither Fig. 8 nor the plotted curves. The plotted
   curves correspond to about 1543, 685 and 830 V s/m. Only the 1300 rpm case is consistent.
4. **Hidden part of Fig. 9:** opaque white rectangles hide the mapped-gain curve from 1500 to 2050 rpm and from 960 to
   970 rpm. These are exactly the speeds where it lies below the uncontrolled SLD; in the hidden band it drops to
   3.23 mm at 1940 rpm.
5. **Interpolated gain map:** Fig. 8 is a straight line over 1410–1860, 620–740, 560–620 and 500–550 rpm, which
   looks like linear interpolation.
6. **"Worst case 3.6 mm":** this minimum appears nowhere on the Fig. 9 curve. The visible minimum is 3.82 mm and the
   hidden one is 3.23 mm.
7. **"10 mm stable at 1300 rpm":** Figs. 9 and 10 mark 1300 rpm / 10 mm as chatter. The "3 mm" single-gain value is
   the 1200 rpm test.
8. **Table 3 vs Fig. 9:** at 1300, 2300, 2800 and 3000 rpm, Table 3 is 0.24–0.5 mm higher than Fig. 9.
9. **Predicted boundary missed:** two mapped-gain chatter points fall below the published boundary, 1300 / 10 mm
   and 2800 / 5.5 mm.
10. **Unstated or unclear settings:**
    - The units of the 27 N check are unclear (g·ẋ is in V).
    - The SADE settings and the bounds of the search set are not given.
    - The actuator "stiffness" in Table 1 is ωp².
    - The text says the uncontrolled tests were at 1200 and 2800 rpm, but Fig. 9 also has a point at 700 rpm.
