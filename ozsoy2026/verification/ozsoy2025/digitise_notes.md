# Digitisation of Ozsoy, Sims & Ozturk 2025 (Mech. Syst. Signal Process. 224, 111942)

The article is open access under CC BY 4.0. Two of its figures were digitised; the PDF itself is not
stored in this repository.

## Method

Two digitisations of each figure were made independently (separate agents, separate working
directories, no shared intermediate results), each calibrating the axes from the pixel positions of at
least two labelled ticks or gridlines, measuring every marker or label from the pixels, and checking the
result on an overlay drawn on the figure. Their readings were then compared item by item; an adjudication
step was planned for any disagreement and was not needed.

- **Fig. 6** (stability lobe diagram, DVF, g = 253 V s/m, structure of the article's Table 2): an embedded
  JPEG raster of 1428 x 892 px at 300 ppi (axes 300-3100 rpm, 0-33 mm; 1 px = 2.09 rpm = 0.043 mm).
  Gridline fits had residuals below 0.21 px. 49 markers: 28 controlled stable, 9 controlled chatter,
  2 controlled marginal, 1 controlled loss of contact, 2 uncontrolled stable, 3 uncontrolled chatter and
  4 'onset of actuator saturation in time-domain simulation' (the article's simulation, not a test).
  Every unoccluded marker lies within 1 rpm and 0.03 mm of a round value (multiples of 100 rpm and
  0.5 mm; the simulation squares at multiples of 0.1 mm); the lettered points A-H reproduce the values
  given in the article's text. One controlled-stable circle at 2100 rpm / 2.5 mm is almost hidden under
  an uncontrolled-chatter diamond and was found by both digitisations (low confidence). Both
  digitisations agree on all 49 markers; the stability curves were sampled every 25 rpm.
- **Figs. 10 and 12** (saturation tests at 1700-2200 rpm, re-clamped structure, DVF): vector graphics,
  rendered at 600 dpi. 56 annotated test points: the measured amplitude of the command in N (36) or
  'Sat' (20), with the outcome '(chatter)' (5) or '(marginal)' (3) where given. The labels are
  MATLAB text, left-aligned at the test speed and vertically centred on the test depth (two-line 'Sat /
  (outcome)' labels centred as a block); every label centre lies within 0.06 mm of a round depth
  (0.5, 1.5, 2, 3, 5, 10, 15, 20 and 22 mm). Every value read in Fig. 10 equals the value at the same
  place in Fig. 12 where Fig. 12 shows it (52 of 56; Fig. 12 masks the region above its controlled
  limit). The two digitisations agree on all 56 values and outcomes, on the stability curves to
  0.08 mm and on the three ellipses of predicted saturation.

## Files

- `points_fig6.json`: markers (type, nominal speed and depth, measured position, label, confidence,
  gain used here: 0 for uncontrolled, 253 V s/m otherwise) and the two curves.
- `points_fig10.json`: test points (speed, nominal and measured depth, value, outcome), the two curves
  and the ellipses.
