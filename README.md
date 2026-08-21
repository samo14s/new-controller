# التحكّم النشط في اهتزاز فرز القِطع الرقيقة — المرحلتان 0 و1

مشروع بحثي مبنيّ على:

> J. Du, X. Liu, H. Dai, X. Long, **"Robust combined time delay control for
> milling chatter suppression of flexible workpieces"**,
> *International Journal of Mechanical Sciences* **274** (2024) 109257.

## الحالة

| المرحلة | الحالة |
|---|---|
| **0** — تثبيت النموذج المرجعي | ✅ [`docs/00_BASELINE_MODEL.md`](docs/00_BASELINE_MODEL.md) |
| **1** — إعادة إنتاج نتائج المرجع | ✅ [`docs/01_REPRODUCTION.md`](docs/01_REPRODUCTION.md) |
| **2** — بناء نموذج متقدّم | ⛔ يسقط — النموذج المرجعي **هو** Du 2024 |
| **3** — تعريف عدم اليقين | ✅ منجَز مسبقًا في `baseline/control_old/` |
| **4** — ربط عدم اليقين بإزالة المادة | ✅ [`docs/03_CONTRIBUTION_KERNEL.md`](docs/03_CONTRIBUTION_KERNEL.md) |
| **5** — النموذج المغلق | ✅ [`docs/04_METHOD.md`](docs/04_METHOD.md) |
| **6** — المتحكّمات المرجعية | ✅ PID / LQR / SMC / μ-synthesis |
| **7** — تحديد المشكلة الرياضية | ✅ [`docs/03`](docs/03_CONTRIBUTION_KERNEL.md) |
| **8** — المتحكّم المقترح PB-RAC | ✅ [`phase2/controllers.py`](phase2/controllers.py) |
| **9** — إثبات الاستقرار | ✅ اختبار مضبوط + شهادة LK |
| **10** — هامش عدم اليقين $\delta_{\max}$ | ✅ [`docs/05_COMPARISON.md`](docs/05_COMPARISON.md) |
| **11** — هامش التأخير | ✅ مسح فلوكيه على سرعة المغزل |
| **12–14** — المقارنة والسيناريوهات والمقاييس | ✅ [`docs/05_COMPARISON.md`](docs/05_COMPARISON.md) |
| **15–16** — النتيجة والمساهمات | ✅ [`docs/06_CONTRIBUTIONS.md`](docs/06_CONTRIBUTIONS.md) |

### مخطّط البحث المرفوع (المراحل 0→9)

| المرحلة | الحالة |
|---|---|
| 0–2 نموذج مُصادَق + دخل بيزو + فضاء حالة موحّد | ✅ `phase2/plant_ss.py` |
| 3 FOPID / LQG / μ-TDC | ✅ `phase2/ctrl2.py` |
| 4 المتحكّم المقترح المعتمِد على الموضع | ✅ `PS-AC` |
| 5–6 شرط الاستقرار، $\Delta_{\max}$، $\tau_{\max}$ | ✅ `phase2/certify2.py` |
| 7–8 المقارنة وأربعة سيناريوهات | ✅ `phase2/run_stage78.py` |
| 9 صياغة المساهمة | ✅ [`docs/07_STAGE_EXECUTION.md`](docs/07_STAGE_EXECUTION.md) |

> ⚠️ **تصحيح نَسَب:** لا يوجد نموذج Zhang 2019 في هذه الحزمة. ما هو مُحقَّق —
> وبإتقان — هو Du 2024 نفسه. الأدلة في
> [`docs/02_ASSUMPTIONS_VS_DU2024.md §0`](docs/02_ASSUMPTIONS_VS_DU2024.md).

## الوثائق

| الملف | المحتوى |
|---|---|
| [`docs/00_BASELINE_MODEL.md`](docs/00_BASELINE_MODEL.md) | النموذج المرجعي كاملًا: المعادلات (1)–(31)، كل المعاملات، تعريف كل متغيّر، فضاء الحالة، نموذج القوة، نموذج التأخير، نموذج البيزو، حساب الاضطرابات، الاصطلاحات |
| [`docs/01_REPRODUCTION.md`](docs/01_REPRODUCTION.md) | حكم إعادة الإنتاج على أربعة مستويات + الخلل الوحيد المكتشَف (اصطلاح الإشارة) |
| [`docs/02_ASSUMPTIONS_VS_DU2024.md`](docs/02_ASSUMPTIONS_VS_DU2024.md) | 30 فرضية، فرضيةً بفرضية، مع تصنيف: مطابق / إضافة / انحراف / خلل |
| [`docs/03_CONTRIBUTION_KERNEL.md`](docs/03_CONTRIBUTION_KERNEL.md) | **المعادلة الواحدة التي ستتغيّر**: ثلاثة إخفاقات مبرهَنة في وصف عدم اليقين، والبديل المشتقّ من الفيزياء |
| [`docs/04_METHOD.md`](docs/04_METHOD.md) | المراحل 4–11: المجموعة الفيزيائية، الحلقة المغلقة، المتحكّمات الخمسة، الشهادتان |
| [`docs/05_COMPARISON.md`](docs/05_COMPARISON.md) | المراحل 12–14: أربعة سيناريوهات، ستّة مقاييس، كل الأرقام |
| [`docs/06_CONTRIBUTIONS.md`](docs/06_CONTRIBUTIONS.md) | المراحل 15–16: النتيجة المستهدَفة والمساهمات الثلاث، وما **ليس** مساهمة |
| [`docs/07_STAGE_EXECUTION.md`](docs/07_STAGE_EXECUTION.md) | **تنفيذ المخطّط المرفوع**: المراحل 0→9، المتحكّم المعتمِد على الموضع، وكل الجداول |

## البنية

```
phase2/         العمل الجديد كلّه (المراحل 4-16)، اصطلاح الإشارة +1
baseline/       حزمة العمل السابقة كما هي، بلا أي تعديل
  plant/          نموذج الحلقة المفتوحة (القسم 2 من المرجع)
  control_old/    متحكّم المرجع: mu-synthesis + تأخير فعّال (القسم 3)
  control/        طبقة جديدة: FOPID مقابل ADRC-FOPID + PSO (ليست من المرجع)
  results/        مخرجات سابقة
analysis/       نصوص تحليل جديدة، كلّها قابلة لإعادة التشغيل
docs/           الوثائق أعلاه
results/        مخرجات analysis/
```

## التشغيل

```bash
pip install numpy scipy matplotlib cvxpy control slycot

# المراحل 0-4 : التحقّق والتدقيق
python analysis/reproduce_baseline.py    # المرحلة 1: مطابقة المرجع
python analysis/uncertainty_audit.py     # F1 + F2: تدقيق المعادلات (22)-(25)
python analysis/material_removal.py      # F3 + المبرهنات T1-T4

# المراحل 5-16 : التصميم والشهادة والمقارنة
cd phase2
python run_design.py     # المراحل 6 و8  : PID / LQR / SMC / PB-RAC بـ PSO
python run_musyn.py      # المتحكّم 4    : mu-synthesis بتكرار D-K
python run_certify.py    # المراحل 9-11 : a_p^DI، delta_max، مجال التأخير
python run_compare.py    # المراحل 12-14: أربعة سيناريوهات
python figures.py        # الأشكال
python report.py         # جداول docs/05
```

المخرجات النصّية الكاملة تُكتب في `results/`.

## النتائج الأساسية بسطر واحد لكل منها

* **إعادة الإنتاج:** التواترات ضمن 3 % بإزاحة منتظمة، اللارنينات تصادق أشكال
  الأنماط بمعزل عن المعايرة، والشكل 7 مُعاد إنتاجه ضمن 0.7 %.
* **خلل:** `baseline/control/config.py` يستعمل اصطلاح إشارة يجعل الحلقة المفتوحة
  **مستقرّةً** عند نقطة التشغيل المرجعية للمرجع — يجب تصحيحه إلى `+1`.
* **F1:** المعادلة (25) تُسقط الحدود المتقاطعة لجداء مجالين ⇒ تحت-تغطية
  **38.9 ×** على صلابة تجدّد النمط الأول.
* **F2:** $D^TD$ من الرتبة 1 دائمًا ($\max|\det|=3\times10^{-15}$)؛ صندوق المرجع
  رباعي الأبعاد حول مجموعة ثنائية، ويحتوي مصفوفات غير معرَّفة الإشارة.
* **F3:** صندوق الـ 10 % يسمح بارتفاع تواتري أقصاه 10.55 %، والمرجع نفسه يقيس
  **+17 %** ⇒ التجربة تخرج من مجموعة التصميم.
* **T1:** $\Delta M(\eta)\preceq0$ و$\Delta K(\eta)\preceq0$ **بالضبط** لكل مجال
  إزالة — نصف الصندوق المتناظر غير قابل للبلوغ فيزيائيًّا.
* **الآليّة:** إزالة المادة عند حافّة حرّة = فقدان كتلة شبه خالص
  (−27.3 % كتلة مقابل −0.038 % صلابة للنمط الأول)، مؤكَّدة بطريقين مستقلّين
  يتّفقان إلى 0.3 نقطة.
