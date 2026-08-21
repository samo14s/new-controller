# المرحلة 0 — النموذج المرجعي (Baseline Model)

> **تنبيه أساسي قبل كل شيء:** النموذج المُحقَّق في هذا المشروع ليس Zhang 2019.
> هو **Du, Liu, Dai & Long, *Int. J. Mech. Sci.* **274** (2024) 109257** —
> «Robust combined time delay control for milling chatter suppression of flexible
> workpieces». راجع [`02_ASSUMPTIONS_VS_DU2024.md`](02_ASSUMPTIONS_VS_DU2024.md)
> للأدلة. هذا يغيّر خطة العمل: المرحلتان 1 و2 من خطتك مدموجتان في مرحلة واحدة،
> ولا يوجد نموذج وسيط يُبنى.

هذا الملف هو **المرجع الوحيد** لكل ما يأتي بعده: كل معادلة، كل معامل، كل اصطلاح.
أي نتيجة لاحقة تُنسب إلى رقم معادلة من هنا.

---

## 1. الهندسة والإحداثيات

ثلاثة أنظمة إحداثيات (الشكل 1 في المرجع):

| النظام | الأصل | الاستعمال |
|---|---|---|
| $OXYZ$ | عام | مرجع المكنة |
| $O_P X_P Y_P Z_P$ | الركن السفلي الأيسر للمستوي المتوسط للّوحة | إزاحات اللوحة |
| $O_T X_T Y_T Z_T$ | مركز قاعدة الأداة | إزاحات الأداة |

اللوحة **كابولية** (مثبَّتة عند $z_P = 0$، حرّة على الأضلاع الثلاثة الأخرى).
الفرازة تمرّ على **الحافّة العليا** $z_P = h_P$، والأداة تتقدّم في اتجاه $x_P$.

| رمز | معنى | قيمة (الجدول 1) |
|---|---|---|
| $l_P$ | طول اللوحة | 100 mm |
| $h_P$ | ارتفاع اللوحة | 80 mm |
| $b_P$ | سماكة اللوحة | 4 mm |
| $\rho_P$ | الكثافة (AL6061) | 2830 kg/m³ |
| $E_P$ | معامل يونغ | 69 GPa |
| $\mu_P$ | معامل بواسون | 0.33 |
| $D_P = \dfrac{E_P b_P^3}{12(1-\mu_P^2)}$ | صلابة الانحناء | 413.4 N·m |

---

## 2. المعادلة الأمّ — المعادلة (1)

النظام الكامل قبل أي تبسيط:

$$
\begin{aligned}
m_{Tx}\ddot x_T + c_{Tx}\dot x_T + k_{Tx}x_T &= F_x\\
m_{Ty}\ddot y_T + c_{Ty}\dot y_T + k_{Ty}y_T &= F_y\\
m_{Px}\ddot x_P + c_{Px}\dot x_P + k_{Px}x_P &= -F_x\\
D_P\!\left(\frac{\partial^4 y_P}{\partial x_P^4}
 + 2\frac{\partial^4 y_P}{\partial x_P^2\partial z_P^2}
 + \frac{\partial^4 y_P}{\partial z_P^4}\right)
 + \rho_P b_P\frac{\partial^2 y_P}{\partial t^2} &= f_y(x_P,z_P,t)
\end{aligned}
$$

مع $f_y = -F_y$. السطر الرابع هو **صفيحة كيرشوف**، وهو معادلة تفاضلية جزئية.

---

## 3. قوى القطع — المعادلات (2)–(4)

$$
\begin{bmatrix}F_x\\F_y\end{bmatrix}
=\begin{bmatrix}\alpha_1(t)&\alpha_2(t)\\ \alpha_3(t)&\alpha_4(t)\end{bmatrix}
\begin{bmatrix}x_r(t)\\ y_r(x_P,z_P,t)\end{bmatrix}
\tag{2}
$$

معاملات القوة، المعادلة (3):

$$
\begin{aligned}
\alpha_1(t)&=\sum_{j=1}^{N_T}\big(-k_1k_t\,ss(t,j)-k_2k_t\,sc(t,j)\big)\\
\alpha_2(t)&=\sum_{j=1}^{N_T}\big(-k_1k_t\,sc(t,j)-k_2k_t\,cc(t,j)\big)\\
\alpha_3(t)&=\sum_{j=1}^{N_T}\big(k_2k_t\,ss(t,j)-k_1k_t\,sc(t,j)\big)\\
\alpha_4(t)&=\sum_{j=1}^{N_T}\big(k_2k_t\,sc(t,j)-k_1k_t\,cc(t,j)\big)
\end{aligned}
$$

$$
k_1=\frac{k_n}{\cos\eta},\qquad
k_2=1+\mu_c\tan\eta\,(\cos\gamma_n-k_n\sin\gamma_n)
$$

التكاملات على السنّ (Balachandran & Zhao، المعادلة 4):

$$
\begin{aligned}
ss(t,j)&=\frac{z_2-z_1}{2}+\frac{R_T}{2\tan\eta}\sin[\theta_2-\theta_1]\cos[\theta_2+\theta_1]\\
sc(t,j)&=\frac{R_T}{2\tan\eta}\sin[\theta_1-\theta_2]\sin[\theta_1+\theta_2]\\
cc(t,j)&=\frac{z_2-z_1}{2}-\frac{R_T}{2\tan\eta}\sin[\theta_2-\theta_1]\cos[\theta_2+\theta_1]
\end{aligned}
$$

مع $\theta(t,j,z)=\Omega t+\dfrac{2\pi j}{N_T}-\dfrac{z\tan\eta}{R_T}$، و
$z_1,z_2$ حدود الجزء المنخرط من السنّ.

| رمز | معنى | قيمة (الجدول 3) |
|---|---|---|
| $N_T$ | عدد الأسنان | 3 |
| $D_T=2R_T$ | قطر الأداة | 10 mm |
| $\eta$ | زاوية اللولب | 35° |
| $\gamma_n$ | زاوية المشط الناظمية | 15° |
| $k_t$ | معامل قوة القطع | 925 MPa |
| $k_n$ | ثابت التناسب | 0.26 |
| $\mu_c$ | معامل الاحتكاك | 0.20 |

**التنفيذ:** `baseline/plant/milling_dynamics.py::alpha34` — تكامل **تحليلي**
سنًّا بسنّ على المجال المنخرط، لا تكامل عددي. الانخراط في الفرز الموافق
(down milling): $\varphi_{st}=\pi-\arccos(1-a_e/R_T)$، $\varphi_{ex}=\pi$.

---

## 4. نموذج التأخير — المعادلة (5)

$$
\begin{aligned}
x_r(t)&=x_T(t)-x_T(t-\tau)-x_P(t)+x_P(t-\tau)+f_\tau\\
y_r(t)&=y_T(t)-y_T(t-\tau)-y_P(t)+y_P(t-\tau)
\end{aligned}
$$

$$
\boxed{\ \tau=\frac{60}{N_T\cdot \mathrm{rpm}}\ }
$$

**تأخير وحيد** — والمرجع يصرّح بذلك: «For the convenience of controller design,
only single time delay effect is considered». وهو يعود في القسم 4.2 ليقرّ بأن
هذا غير صحيح فيزيائيًّا: «The milling process is a multiple-time delay system
rather than a single-time delay control system actually.» احتفظ بهذه الجملة:
هي أحد أبواب المساهمة المستقبلية.

عند 4900 rpm: $\tau=4.0816$ ms، تواتر مرور الأسنان $1/\tau=245.0$ Hz
(المرجع: 245 Hz ✓).

---

## 5. التمييز المكاني — المعادلات (6)–(10)

قاعدة **تشيبيشيف من النوع الأول** بجداء كرونيكر:

$$
y_P(x_P,z_P,t)=Y_P(\tilde x_P,\tilde z_P)\,q_y(t),\qquad
\tilde x_P=\frac{2x_P}{l_P}-1,\quad \tilde z_P=\frac{2z_P}{h_P}-1
\tag{7}
$$

$$
Y_P=[\Phi_0(\tilde x),\dots,\Phi_{PX-1}(\tilde x)]\otimes
    [\Phi_0(\tilde z),\dots,\Phi_{PZ-1}(\tilde z)]
\tag{8}
$$

المبدأ التغايري المعدَّل (المعادلة 6) يعطي:

$$
M_{PD}\ddot q_y+(K_{PD}-K_\lambda+K_\kappa)q_y=F_y
\tag{9}
$$

بالتفصيل (الملحق A):

$$
M_{PD}=\iint_S\Big(I_2\frac{\partial Y_P^{T}}{\partial\tilde x}\frac{\partial Y_P}{\partial\tilde x}
+I_2\frac{\partial Y_P^{T}}{\partial\tilde z}\frac{\partial Y_P}{\partial\tilde z}
+I_0\,Y_P^{T}Y_P\Big)\,d\tilde x\,d\tilde z
\tag{A.1}
$$

$$
K_{PD}=D_P\iint_S\Big(
Y_{,\tilde x\tilde x}^{T}Y_{,\tilde x\tilde x}
+\mu_P\big(Y_{,\tilde x\tilde x}^{T}Y_{,\tilde z\tilde z}+Y_{,\tilde z\tilde z}^{T}Y_{,\tilde x\tilde x}\big)
+Y_{,\tilde z\tilde z}^{T}Y_{,\tilde z\tilde z}
+2(1-\mu_P)Y_{,\tilde x\tilde z}^{T}Y_{,\tilde x\tilde z}\Big)
\tag{A.2}
$$

مع $I_0=\rho_P b_P$ و $I_2=\rho_P b_P^3/12$ (قصور الدوران).

ثم إلى الفضاء النمطي:

$$
M_P\ddot q_P+C_P\dot q_P+K_Pq_P=U_P^{T}F_y
\tag{10}
$$

$$
M_P=U_P^{T}M_{PD}U_P,\quad
K_P=U_P^{T}(K_{PD}-K_\lambda+K_\kappa)U_P,\quad
q_y=U_Pq_P
$$

### خيارات التنفيذ (وهي انحرافات مصرَّح بها)

| بند | المرجع | التنفيذ (`chebyshev_plate.py`) |
|---|---|---|
| $PX\times PZ$ | غير مذكور | $14\times14=196$ درجة حرّية |
| التثبيت | $K_\lambda$ (مضاعفات لاغرانج) $+\,K_\kappa$ | **عقوبة فقط**: نوابض $k_w=10^{12}$، $k_r=10^{8}$ عند $z=0$ |
| $C_{PD}$ | «introducing plate damping» — بلا صيغة | تخميد نمطي: $C_P=\mathrm{diag}(2\zeta_i\omega_i)$ |
| التطبيع | غير محدَّد | تطبيع بالكتلة: $U_P^{T}M_{PD}U_P=I$ |

تطبيع الكتلة يجعل $M_P=I$، فتُقرأ **كل** النسب المئوية اللاحقة على المصفوفات
النمطية مباشرةً — وهذا ما يجعل مقارنة «10 %» في المرجع ذات معنى.

**النتيجة:** $f_n = [521.0,\ 1069.0,\ 2723.8,\ 3324.1,\ 4129.0]$ Hz
مقابل «theoretical» في الجدول 4: $[537,1101,2805,3423,4254]$ — خطأ
$-2.99\%,-2.91\%,-2.89\%,-2.89\%,-2.94\%$. **الخطأ منتظم الإشارة والمقدار على
الأنماط الخمسة**، أي إزاحة صلابة عامّة (غالبًا $K_\lambda$ المحذوف)، لا خطأ في
أشكال الأنماط.

ثم **معايرة** على التواترات المقاسة (الجدول 4)
$[540,1068,2787,3351,4122]$ Hz دون مسّ أشكال الأنماط — وهي الخطوة نفسها التي
يقوم بها المرجع قبل التصميم.

---

## 6. النموذج المختزل — المعادلة (12)

بإهمال اهتزاز الأداة واهتزاز اللوحة في $x$:

$$
\boxed{
M_P\ddot q_P(t)+C_P\dot q_P(t)+\big(K_P+\alpha_4(t)D_P^{T}D_P\big)q_P(t)
-\alpha_4(t)D_P^{T}D_P\,q_P(t-\tau)=f_\tau\alpha_3(t)D_P^{T}
}
\tag{12}
$$

مع $D_P(\tilde x_P,\tilde z_P)=Y_P(\tilde x_P,\tilde z_P)U_P$ — شعاع شكل النمط
عند نقطة التماسّ.

هذه المعادلة تجمع الخصائص الأربع التي يدّعي المرجع تناولها مجتمعةً لأول مرّة:
$\alpha_4(t)$ **غير ملساء** ودوريّة، $q_P(t-\tau)$ **تأخير**، $q_P\in\mathbb R^n$
**تعدّد الأنماط**، $D_P(x)$ **ديناميكا متغيّرة مع الموضع**.

---

## 7. المشغّل البيزوكهربائي — المعادلات (13)–(15)

$$
M_P\ddot q_P+C_P\dot q_P+(K_P+\alpha_4 D_P^{T}D_P)q_P
-\alpha_4 D_P^{T}D_Pq_P(t-\tau)=f_\tau\alpha_3 D_P^{T}+H_{Pe}u_P(t)
\tag{13}
$$

$$
\mathrm{piezo}_j=-C_{P0}d_{31}h_{Pa}\Big\{
\Big[\textstyle\int_{\tilde z_{P1}}^{\tilde z_{P2}}\!\!D_{Px}(\tilde x_{P2},\tilde z)d\tilde z
-\int_{\tilde z_{P1}}^{\tilde z_{P2}}\!\!D_{Px}(\tilde x_{P1},\tilde z)d\tilde z\Big]
+\big[\text{المِثل في }\tilde z\big]\Big\}U_{Pj}
\tag{14}
$$

$$
C_{P0}=-\frac16\frac{1+\mu_{Pe}}{1-\mu_P}
\frac{E_Pb_P^2P_M}{1+\mu_P-(1+\mu_{Pe})P_M},\quad
P_M=-\frac{E_{Pe}}{E_P}\frac{1-\mu_P^2}{1-\mu_{Pe}^2}
\frac{3h_{Pa}b_P(b_P+h_{Pa})}{0.5b_P^3+4h_{Pa}^3+3b_Ph_{Pa}^2}
\tag{15}
$$

| رمز | معنى | قيمة (الجدول 2) |
|---|---|---|
| $d_{31}$ | ثابت الانفعال | $175\times10^{-12}$ m/V |
| $h_{Pa}$ | سماكة الرقعة | 0.7 mm |
| $E_{Pe}$ | معامل يونغ | 63 GPa |
| $\mu_{Pe}$ | بواسون | 0.35 |
| الرقعة | QDA60-20-0.7 | 60 mm × 20 mm |

**التنفيذ** يكتب المعادلة (14) بمبرهنة التباعد في صورة مكافئة
$H_{Pe}=m_{pz}\int_{\text{patch}}\nabla^2Y\,dA$ مع
$m_{pz}=-\eta E_{Pe}d_{31}(b_P+h_{Pa})/[2(1-\mu_{Pe})]$، ويضيف **زيادتين لا
توجدان في المرجع**:

1. **تصليب وكتلة الرقعة** (مقطع محوَّل): $r_K=0.517$، $r_M=0.461$ — فالرقعة
   جسم مادّي لا مجرَّد مصدر عزم.
2. **مردود اللصق** (Crawley & de Luis، shear lag): $\eta=0.886$.

نتيجة عددية بعد المعايرة:

```
H_Pe  (N/V)      = [-0.0613, +0.0643, -0.0112, -0.1421, +0.2093]
D_obs (100,80mm) = [ 6.688,   9.893,   7.487,  10.754,   8.077 ]
D_obs · H_Pe     = [-0.410,  +0.636,  -0.084,  -1.528,  +1.690 ]
C_P0             = 1.3355e5
```

---

## 8. صيغة فضاء الحالة — المعادلة (16)

$$
\begin{aligned}
\dot x_P(t)&=A_P(\tilde x_P,\tilde z_P,t)\,x_P(t)+B_{Ps}F_{PI}(\tilde x_P,\tilde z_P,t,t-\tau)+B_Pu_P(t)\\
y_P(t)&=C_{PT}(\tilde x_P,\tilde z_P)\,x_P(t)
\end{aligned}
$$

$$
x_P=\begin{bmatrix}q_P\\\dot q_P\end{bmatrix},\quad
A_P=\begin{bmatrix}0&I\\-M_P^{-1}(D_P^{T}D_P\alpha_4(t)+K_P)&-M_P^{-1}C_P\end{bmatrix},
$$
$$
B_{Ps}=\begin{bmatrix}0\\M_P^{-1}\end{bmatrix},\quad
B_P=\begin{bmatrix}0\\M_P^{-1}H_{Pe}\end{bmatrix},\quad
C_{PT}=\begin{bmatrix}D_P&0\end{bmatrix}
$$

$$
F_{PI}=\alpha_4(t)D_P^{T}D_P\,q_{Pr}(t-\tau)+f_\tau D_P^{T}\alpha_3(t)
$$

> **انتبه إلى بنية هذه الصيغة.** المرجع **لا** يكتب النظام في الشكل
> $\dot x=Ax+A_dx(t-\tau)+Bu$ الذي في خطّتك. هو يضع الحدّ المتأخّر **داخل
> الدخل** $F_{PI}$ ويعامله كاضطراب خارجي. هذا الاختيار هو الذي يسمح له لاحقًا
> بتحويل النظام إلى «real-time differential equation with constant
> coefficients» — وهو أيضًا مصدر أهمّ ضعف في المرجع، لأن التأخير لم يعد يظهر في
> شرط الاستقرار إطلاقًا. راجع
> [`03_CONTRIBUTION_KERNEL.md`](03_CONTRIBUTION_KERNEL.md).

---

## 9. اختزال الأنماط وعدم اليقين الجَمعي — المعادلات (17)–(20)

$$
G_P(s)=G_{Pr}(s)+\Delta_{Pa}(s)W_{Pa}(s),\qquad \|\Delta_{Pa}\|_\infty<1
\tag{17}
$$

$G_{Pr}$ = **أول نمطين فقط**. الأنماط 3–5 تُبتلع في $\Delta_{Pa}W_{Pa}$.

$$
W_{Paf}(s)=\frac{r_{Paf}(s^2+2\zeta_{Paf1}\omega_{Paf1}s+\omega_{Paf1}^2)}
{s^2+2\zeta_{Paf2}\omega_{Paf2}s+\omega_{Paf2}^2},\qquad
W_{Pau}(s)=\text{المِثل}
\tag{18,19}
$$

| | $r$ | $\zeta_1$ | $\zeta_2$ | $\omega_1/2\pi$ | $\omega_2/2\pi$ |
|---|---|---|---|---|---|
| $W_{Paf}$ (مدخل القوة) | $14\times10^{-6}$ | 0.56 | 0.12 | 1400 Hz | 2800 Hz |
| $W_{Pau}$ (مدخل الجهد) | $4\times10^{-7}$ | 0.58 | 0.22 | 1100 Hz | 3500 Hz |

---

## 10. تمثيل عدم اليقين — المعادلات (21)–(25)

**هذه هي المعادلات التي ستتغيّر.** أُثبتها هنا كما هي.

$$
(M_{Pr0}+\Delta_{PrM})\ddot q_{Pr}+(C_{Pr0}+\Delta_{PrC})\dot q_{Pr}
+\big(K_{Pr0}+\Delta_{PrK}+\alpha_{40}D_{Pr0}^{T}D_{Pr0}+\Delta_{PrD}\big)q_{Pr}
=F_{PIr}+H_{Pe}u_P
\tag{21}
$$

$$
\Delta_{PrM}=L_{Pm}\delta_{Pm},\quad
\Delta_{PrC}=L_{Pc}\delta_{Pc},\quad
\Delta_{PrKD}=\Delta_{PrK}+\Delta_{PrD}=L_{Pkd}\delta_{Pkd}
\tag{22}
$$

مع $|\delta_{Pm i}|\le1$، $|\delta_{Pc i}|\le1$، $|\delta_{Pk i}|\le1$،
$|\delta_{PD i}|\le1$ — **عشرة معاملات حقيقية مستقلّة**.

$$
\alpha_{40}=1.6\,\bar\alpha_4,\qquad L_{P\alpha}=1.3\,\bar\alpha_4
\tag{23}
$$
(أي $\alpha_4(t)\in[0.3,\,2.9]\bar\alpha_4$، مبرَّرًا بالشكل 6.)

$$
D_{Pr0}^{T}D_{Pr0}=\begin{bmatrix}DD_{10}&DD_{20}\\DD_{30}&DD_{40}\end{bmatrix},\quad
\Delta_{DD}=\begin{bmatrix}L_{DD1}\delta_{DD1}&L_{DD2}\delta_{DD2}\\
L_{DD3}\delta_{DD3}&L_{DD4}\delta_{DD4}\end{bmatrix}
\tag{24}
$$
حيث $DD_{i0}$ = متوسط (أقصى + أدنى)/2 على طول الحافّة، و$L_{DDi}$ = نصف المدى.

$$
\alpha_{40}D_{Pr0}^{T}D_{Pr0}=1.6\bar\alpha_4\,[DD_{i0}],\qquad
\Delta_{PrD}=\big[L_{P\alpha}L_{DDi}\,\delta_{PDi}\big]
\tag{25}
$$

وأخيرًا، جملة نصّية في نهاية القسم 3.2:

> «the mode mass and mode stiffness use calculated results by the proposed model
> as nominal values, and are **set as 10 % perturbations**. The nominal values of
> damping ratio are measured by mode experiments, and the perturbation is **set
> as 20 %**.»

### القيم العددية عند نقطة التشغيل المرجعية S

نقطة S (القسم 4.2): **4900 rpm، $a_e=0.1$ mm، $a_p=0.3$ mm، $f_z=0.02$ mm/سنّ،
فرز موافق**.

```
abar4      = -11 263.3 N/m          (ذروة alpha4(t) = -139 792.6، النسبة 12.41)
alpha40    = -18 021.3 N/m          (المعادلة 23)
L_Palpha   = -14 642.3 N/m          (المعادلة 23)
نسبة الانخراط في الدور = 11.8 %
```

بمعايرة الشكل 7 (وهو محسوب على اللوحة **المجرَّدة** المتناظرة):

```
DD0  = [[3.7049, 0.0000],          الشكل 7:  DD11 من 3.60 إلى 3.81
        [0.0000, 1.6750]]                    DD12 سعة 3.45
L_DD = [[0.1012, 3.4746],                    DD22 من 0 إلى 3.35
        [3.4746, 1.6750]]
```

المطابقة مع الشكل 7 ممتازة: $DD_{11}$ متوسط 3.7049 (الهدف 3.705) وسعة 0.1012
(الهدف 0.105)، $DD_{12}$ سعة 3.4746 (الهدف 3.45)، $DD_{22}$ متوسط وسعة 1.675
(الهدف 1.675).

> **لاحظ الآن — وهذه أول علامة إنذار:** القيمة الاسمية للعنصر خارج القطر هي
> **صفر بالضبط**، وسعة اضطرابه 3.47، أي أكبر من القيمة الاسمية للعنصر القطري
> الثاني (1.675). «الاسمي» هنا لا يحمل أيّ معلومة، و«الاضطراب» هو المجموعة
> كلّها. هذا ليس صندوقًا صغيرًا حول نقطة تشغيل؛ هذا صندوق يبتلع النظام.

---

## 11. المتحكّم المرجعي — المعادلات (26)–(31)

* **(26)–(27)**: بناء الـ generalized plant مع الأوزان $W_{Pf},W_{Pu},W_{Pn}$.
  المرجع **لا يعطي أي قيمة عددية** لهذه الأوزان الثلاثة («designed as low-pass
  filters» / «designed as a constant»). التنفيذ يعرّفها صراحةً في
  `baseline/control_old/weights.py` ويصرّح بذلك.
* **(28)–(29)**: شرط $\sup_\omega\mu_\Delta(T_{P\mu}(j\omega))<\gamma$ وحلّه
  بتكرار D–K.
* **(30)**: متحكّم التأخير الفعّال
  $$u_{Pd}(t)=K_{Pp}\,y_{P\mu}(t-\tau)+K_{Pd}\,\dot y_{P\mu}(t-\tau)$$
  المرجع **لا يعطي $K_{Pp}$ ولا $K_{Pd}$** («after designing the coefficients
  suitably»).
* **(31)**: أثر ذلك على دخل القوة $F_{PId}$.

---

## 12. الاصطلاحات التي يجب احترامها

### 12.1 إشارة اقتران التجدّد — **مسألة مفتوحة محسومة الآن**

المعادلة (13) كما نُشرت تعطي `sign = +1`. اشتقاق الإشارة من
(1)(2)(5)(10) يعطي `-1`. الفرق ليس تجميليًّا:

| | حدّ الاستقرار عند 4900 rpm | $\rho$ عند $a_p=0.3$ mm | زمن التباعد |
|---|---|---|---|
| `sign = +1` | **0.039–0.045 mm** | 1.23–1.40 (**غير مستقرّ**) | 0.107 s |
| `sign = -1` | 0.33–1.35 mm | 0.85–0.95 (**مستقرّ**) | لا يتباعد |

المرجع يقول: الحدّ التجريبي بلا تحكّم $\approx0.1$ mm (الشكل 18)، والقطع عند S
**يتباعد** (الشكل 14a). إذن **`sign = +1` هو الاصطلاح الصحيح**، و`sign = -1`
يجعل نقطة التشغيل المرجعية للمرجع مستقرّةً — وهو ما لا يمكن قبوله.

الحجّة المضادّة الوحيدة لـ `-1` هي تواتر الاهتزاز: المرجع يذكر
$f_{c2}=1135$ Hz. عائلة فلوكيه (مطويّة على $1/\tau=245$ Hz) تعطي
$\{935.5,\,1024.5,\,1180.5,\,1269.5\}$ لـ `+1` و$\{1095.5,\,1109.5,\dots\}$
لـ `-1`. المسافة إلى 1135 Hz هي 45.5 Hz مقابل 25.5 Hz — أي **أصغر من نصف تباعد
العائلة (122 Hz) وأصغر من خطأ التواتر النمطي نفسه (3 % = 32 Hz)**. الدليل
الترددي غير حاسم؛ دليل الاستقرار حاسم.

> ⚠️ **`baseline/control/config.py` يضع حاليًّا `SIGN_SIM = -1`.** كل نتائج
> FOPID/ADRC-FOPID في `baseline/results/` مبنية على الاصطلاح الذي يجعل الحلقة
> المفتوحة مستقرّة عند نقطة المرجع. يجب إعادة تشغيلها بـ `+1` قبل أي مقارنة
> نهائية.

### 12.2 خطوة الزمن

* حلقة مفتوحة: `n_sub = 82` كافية.
* أي متحكّم رنيني: **≥ 164**.
* مؤثّرات كسرية بمرشّح Oustaloup حتى 100 kHz: **`n_sub = 656`** وإلاّ انطوى
  القطب وتباعدت المحاكاة زائفًا.

### 12.3 الموضع والعمق

* امسح $[0,100]$ mm **كاملة**: التناظر $x\leftrightarrow l_P-x$ مكسور مرّتين
  (الرقعة لا مركزية، والحسّاس عند ركن واحد).
* منطقة الاستقرار في الحلقة المغلقة **غير متّصلة في $a_p$**؛ قيّم على مجال
  أعماق لا على نقطة.

---

## 13. خريطة الملفات

| الملف | يُحقّق |
|---|---|
| `baseline/plant/chebyshev_plate.py` | (6)–(10)، (14)–(15)، الملحق A |
| `baseline/plant/milling_dynamics.py` | (2)–(4)، (21)–(25)، (18)–(19) |
| `baseline/plant/stability_fdm.py` | فلوكيه بالتمييز الكامل على (12) |
| `baseline/plant/lti_floquet.py` | فلوكيه في الحلقة المغلقة (متحكّم LTI أي رتبة) |
| `baseline/plant/time_domain.py` | تكامل نيومارك لـ (13) بموضع أداة متحرّك |
| `baseline/control_old/uncertain_plant.py` | (16)، (17)، (21)–(26)، (27) |
| `baseline/control_old/weights.py` | (18)–(19) + أوزان التصميم |
| `baseline/control_old/dk_synthesis.py` | (28)–(29) |
| `baseline/control_old/delay_control.py` | (30)–(31) |
| `baseline/control/` | طبقة جديدة (FOPID مقابل ADRC-FOPID + PSO) — ليست من المرجع |

---

## 14. أرقام مرجعية للتحقّق السريع

| المقياس | القيمة |
|---|---|
| $f_n$ اللوحة المجرَّدة (Hz) | 521.0 / 1069.0 / 2723.8 / 3324.1 / 4129.0 |
| الخطأ مقابل «theoretical» | −2.99 % … −2.89 % (منتظم) |
| $D_{obs}$ عند (100, 80) mm | [6.688, 9.893, 7.487, 10.754, 8.077] |
| $H_{Pe}$ (N/V) | [−0.0613, +0.0643, −0.0112, −0.1421, +0.2093] |
| المرونة الساكنة عند الركن | 6.600 μm/N (−103.6 dB re 1 m/N) |
| اللارنينات (مدخل القوة) | 731.2 / 2123.7 / 3008.3 / 3937.2 Hz |
| $\bar\alpha_4$ عند S | −11 263.3 N/m (الذروة −139 792.6، النسبة 12.41) |
| نسبة الانخراط | 11.8 % |
| $\tau$ عند 4900 rpm | 4.0816 ms (245.0 Hz) |
| زمن الممرّ | 20.408 s بتغذية 4.90 mm/s |
| حدّ الحلقة المفتوحة (sign +1) | 0.0393–0.0451 mm |
| زمن التباعد عند S (sign +1) | 0.1067 s |

لإعادة توليدها كلّها:

```bash
python analysis/reproduce_baseline.py
```
