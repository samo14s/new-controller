# مسودّة طلب البيانات من المؤلّفين

المقال ينصّ على: «Data will be made available on request». هذا أرخص مصدر للبيانات
الحقيقية ويحوّل المشروع من «تحقّق على مختبر افتراضي» إلى «تحقّق على قياسات حقيقية
بلا مختبر». المسودّة أدناه بالإنجليزية، جاهزة للإرسال من بريدك المؤسّسي (لم تُرسَل).

**إلى:** dongll@xjtu.edu.cn (Longlei Dong، المؤلّف المراسل) — نسخة إلى Jiaming Zhou
(Ningxia University، المؤلّف المراسل الثاني).

---

**Subject:** Data request — Ouyang et al., Thin-Walled Structures 228 (2026) 115090

Dear Prof. Dong,

I am a researcher at the École Nationale d'Ingénieurs de Sfax (ENIS, Tunisia) working on
data-driven modelling and adaptive vibration control of smart composite structures. Your
paper "From data-driven DMDc modeling to control parameter optimization: A framework for
adaptive vibration control of thin-walled composites" (Thin-Walled Structures 228, 2026,
115090) is the foundation of my current project, and the Data Availability statement
indicates that the data can be shared on request.

I would be very grateful if you could share, in any convenient format:

1. the identification record of Section 3.1 (excitation and actuation voltages, the 35
   displacement signals and the two MFC strain-sensor signals, sampled at 10 kHz);
2. the sine-validation records of Section 3.4;
3. the controlled and uncontrolled sensing/actuating voltages of Figs. 10, 13 and 14, and,
   if possible, the convergence times behind Figs. 16–23;
4. the processing details needed to reproduce the identified model: units used for the
   displacement and strain channels, the low-pass filter before differentiation, the
   resulting ranks p and r, and the actuator used in the two single-channel cases.

The data will be used for academic research only, your work will be cited in every
resulting publication, and I will be happy to share my analysis with you.

With my thanks and best regards,

[Name], [Position]
École Nationale d'Ingénieurs de Sfax (ENIS), University of Sfax, Tunisia
[institutional e-mail]

---

### لماذا هذه البنود بالذات

| البند | ما يحسمه في المشروع |
|---|---|
| 1 | تشغيل sd-DMDc على قياسات حقيقية؛ اختبار حساسية قاعدة الرتبة 99.9 % للوحدات (`docs/03`) |
| 2 | تحقّق خارج العيّنة حقيقي للنموذج المعرَّف |
| 3 | معايرة $\zeta$ و`mu_scale` على بيانات خام بدل القراءة من الصور |
| 4 | أربعة غموضات من `docs/01` لا تُحسم من المقال: الوحدات، المرشّح، $p,r$، المشغّل |
