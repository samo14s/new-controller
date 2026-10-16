# مخطّط الورقة 1 — جاهز للتحويل إلى مسودّة

> هذه الوثيقة هيكلٌ للكتابة لا نتيجة جديدة. كل رقم فيها منقول من `results/`، ومرجعه في `docs/03`.
> الملخّص والعنوان بالإنجليزية لأنّ المجلّات المستهدفة إنجليزية. الصياغة تحترم تموضع الجِدّة في
> `docs/04` §1: الأدوات الكلاسيكية تُنسب إلى أصحابها، والجديد هو ميزانية التخميد وما يتفرّع عنها
> وإعادة تحليل بيانات منشورة.

## العنوان المقترح

*The damping budget of tonal FxLMS: a one-dimensional, closed-form convergence map for active
vibration control of lightly damped thin-walled structures, verified on published experiments*

## الملخّص (مسودّة أولى، ≈ 220 كلمة)

> Recent data-driven frameworks for active vibration control of smart composite plates tune the
> step size μ and filter length L of the FxLMS algorithm by mapping hundreds of (μ, L) combinations
> in simulation and experiment. We show that, for tonal disturbances, this map is one-dimensional
> and largely predictable before any adaptive run. Building on the classical linear time-invariant
> equivalence of tonal FxLMS, we observe that the averaged adaptation loop and the structural mode
> nearest to the disturbance frequency obey a damping budget: their decay rates sum to the modal
> decay rate σ_n = ζ_n ω_n whatever the adaptation gain and whatever the secondary-path model error.
> Three closed-form consequences follow: divergence occurs when the adaptation rate reaches σ_n; the
> fastest convergence scales as t_min ≈ 4.4/σ_n; and the ratio of the two edges of the effective
> region depends on σ_n and the detuning only. On a calibrated virtual testbed (Rayleigh–Ritz plate,
> five piezoelectric patches, measurement chain, data-driven sd-DMDc model), the budget holds within
> 8 %, the edge ratio within 5 %, and the LTI predictor reproduces full-FxLMS convergence times within
> 2–8 %. Re-analysing the published experimental maps of Ouyang et al. (2026), we find that their μ-
> and L-sweeps collapse onto μL (median deviation 3–11 %) and that inverting their region edges
> yields modal damping ratios of 0.19–0.30 %, consistent with three independent estimates. Finally,
> a positive-position-feedback inner loop that multiplies σ_n by 3.6 shortens FxLMS convergence by
> the same factor within 15 %, giving a quantitative design law for hybrid feedback/FxLMS control.

## الأشكال (كلّها موجودة أو قابلة للتوليد من السكربتات)

| # | الشكل | المصدر |
|---|---|---|
| 1 | المختبر الافتراضي: اللوحة والرقع وسلسلة القياس، وتواترات الجدول 4 | `docs/02`، `results/calibration.txt` |
| 2 | الاختزال بوزنين: $\Gamma(L,\omega)$ وتطابقه مع FxLMS الكاملة (مثال) | `scripts/probe_theory.py` (G1) |
| 3 | **ميزانية التخميد:** معدّلا القطبين مقابل $\beta/\beta_{\rm div}$ للحالات الثلاث | `results/budget.txt` (يُرسم) |
| 4 | الحلقة LTI مقابل FxLMS الكاملة: $t_5(\mu)$ للحالات الثلاث | `results/theory_probe.txt` |
| 5 | **انهيار بيانات المقال على $\mu L$** | `figures/fig_collapse_pixel.png` |
| 6 | الحدود المغلقة ونسبة الحدّين؛ وعكس الخرائط لاستنتاج التخميد | `results/edges.txt`، `results/map_probe.txt` |
| 7 | **مخطّط التصميم الموحّد** $t_5\sigma_n$ مقابل $g/\sigma_n$ | `figures/fig_design_chart.png` |
| 8 | قانون الهجين (PPF + FxLMS) | `results/hybrid_probe.txt` (يُرسم) |

## الأقسام

1. **Introduction.** AVC للمركّبات الرقيقة؛ FxLMS ومشكلة الضبط؛ Ouyang 2026 وLi et al. 2024 كأقرب
   عملين؛ الفجوة: لا قانون يفسّر أين تقع المنطقة الفعّالة ولماذا.
2. **Tonal FxLMS as an LTI loop (background).** Glover، وElliott 1987، وMorgan & Sanford، وBayard،
   وVicente & Masgrau، وHaarnoja. والصيغة المدمجة $\Gamma(L,\omega)$ كملاحظة.
3. **The damping budget.** النموذج القاعدي ذو القطبين؛ قاعدة Grant؛ الشروط (نمط مجاور غالب،
   $|\Delta|\gg\sigma_n$)؛ الصمود أمام خطأ النموذج؛ كسر الميزانية (Messner & Bodson، التغذية الراجعة،
   نموذج داخلي بتخميد سالب).
4. **Closed-form consequences.** $\alpha_{\rm div}=\sigma_n$؛ $t_{\min}$؛ حدّا المنطقة ونسبتهما؛
   انعدام المنطقة.
5. **Virtual testbed and data-driven model.** Ritz، والمعايرة على أرقام منشورة بمجموعتين منفصلتين،
   وsd-DMDc وهشاشة قاعدة الرتبة (R1، في الملحق).
6. **Verification.** G1–G4، والتحقّق خارج العيّنة، وخرائط المختبر.
7. **Re-analysis of the published experiments.** الانهيار، وعكس الحدود، والمخطّط الموحّد، والتناقضات
   (H6، والحدّ السريع المنحني عند 120 Hz، والإشارة).
8. **Breaking the budget, and design rules.** PPF + FxLMS (الهجين كنتيجة لازمة لا كمبدأ جديد،
   مع نسبة Clark 1995 وVipperman & Clark 1996 وLi & Wang 2008)؛ قاعدة التصميم في أسوأ حالة
   $(\mu L)_{\rm div}\simeq\sigma_n/(A^2|G||\hat G|\cos\Delta\varphi)$ (وFraanje et al. 2007 للحالة العامّة)؛
   والردّ على «PSO رباعي الأبعاد»: الاختزال إلى $(g,r)$ وهامش ≤ 16 %.
9. **Discussion and limitations.** نمط واحد غالب؛ مرجع نغمي مستقرّ؛ لا لاخطّيات؛ المختبر عيّنة
   معقولة لا نسخة؛ الحاجة إلى بيانات خام (طلب مُرسل؟).
10. **Conclusions.**

## قبل الإرسال

- [ ] قراءة النصوص الكاملة في `docs/04` §4 (Li & Wang 2008، Darlington & Elliott 1985، Morgan &
      Sanford 1992، Zilletti 2016، Elliott 2001 ف3، Li et al. 2024، Siswanto & Chang 2025).
- [ ] المهامّ M1–M4 في `docs/05` §2 (تحسين صيغة $t_{\min}$، تصحيح $\Gamma$ الدقيق بفلوكيه،
      القناتان، التنبّؤ الأعمى بالخرائط).
- [ ] إرسال طلب البيانات (`docs/email_data_request.md`)؛ إن وصلت، يُعاد §7 على البيانات الخام.
- [ ] فحص FxLMS كاملة (لا LTI فقط) لقانون الهجين، بمسار معاد تعريفه على الحلقة المغلقة.
- [ ] بنود المراجِع 1 و3 و5 و6 و7 في `docs/02` §6 (هامش جهد MFC، والتأخير الحسابي، و«حقيقة» FE ثانية).
