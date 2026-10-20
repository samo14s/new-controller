/* Time-domain milling simulation of the 1-DOF flexible workpiece under DVF.
 *
 * Regenerative chip thickness per axial layer and tooth (Sec. 2.1):
 *   h = fz sin(phi) - (x(t) - x(t-T)) cos(phi),  force = 0 when h <= 0
 * Force on the workpiece along y:  f = -sum (Kt sin(phi) - Kr cos(phi)) dz h
 * Closed loop:  m x'' + c x' + k x = f_cut + f_act
 *               q'' + a1 q' + a0 q = V,   f_act = ga (V - a1 q' - a0 q)
 *               V = sat(-g x', vmax)
 * The loop is integrated with RK4 over nsub sub-steps per step.  Two ways of
 * feeding the cutting force to the integrator:
 *   fmode 0  force from the state at the start of the step, held over the
 *            step (the usual textbook scheme; it lags the force by half a
 *            step, which at 512 steps/rev lowers the simulated stability
 *            limit by up to ~9 % at low speed);
 *   fmode 1  force re-evaluated at every RK4 stage: tooth angles advanced to
 *            the stage time and the delayed displacement x(t - T) taken from
 *            a cubic Hermite interpolant of the stored history.
 *
 * Entry and perturbation options (opt[], all off by default; with every
 * option off the run is bit-identical to the plain step entry):
 *   opt[0] ramp_a   axial depth ramped from 0 to a over this many revolutions;
 *                   layer j is engaged over the fraction of [j dz, (j+1) dz]
 *                   below the current depth (idealised: uniform depth around
 *                   the arc with the delay kept at T)
 *   opt[1] ramp_f   feed per tooth ramped over this many revolutions,
 *   opt[2] f0       starting from f0 * fz (and fed at f0 * fz before t = 0);
 *                   the chip is the advance over the last tooth period
 *   opt[3] edge     1: straight entry through a workpiece face normal to the
 *                   feed, touched by the tool periphery at t = 0; an edge
 *                   point at angle phi is in the material once the tool
 *                   centre has advanced s(t) >= R (1 - sin phi)
 *   opt[4] kick_step, opt[5] kick_dv
 *                   velocity kick added to x' at the start of that step
 *   pert            (separate argument, NULL = none) a full-state deviation
 *                   added at kick_step: pert[0..3] to x, x', q, q', and
 *                   pert[4 + j], pert[5 + nd + j] (j = 0..nd) to the stored
 *                   x and x' j steps back, i.e. over the last tooth period
 *   opt[6] clip_on  the command limit vmax applies from this step on
 *   opt[7] nreg     passes remembered by the chip (default 1).  With nreg > 1
 *                   the chip is the least over the last nreg passes,
 *                   h = min_k [k fz sin(phi) - (x - x(t - kT)) cos(phi)],
 *                   so a pass that left the cut does not count as having
 *                   removed material (multiple regeneration, needed once
 *                   the tool leaves the cut, i.e. for chatter amplitudes)
 *   opt[8] g2, opt[9] kd   (only in the build with COMB = 1, see below)
 *                   comb terms of the command (C4):
 *                   V = sat(-g x' - g2 (x' - x'(t-T)) - kd (x - x(t-T)), vmax),
 *                   with x(t-T), x'(t-T) from the cubic Hermite interpolant
 *                   of the stored history (the measured motion, zero before
 *                   t = 0).  Both terms vanish on a tooth-periodic motion.
 *                   kd < 0 opposes the regenerative force (KAPPA < 0 in
 *                   c2_certificate.py), kd > 0 adds to it.
 *   opt[10] rho     (COMB = 1) the comb's delay is rho T instead of T (0 or 1:
 *                   exact): a controller set for a speed off by the factor
 *                   rho, or rho = N for a spindle-period comb
 *   opt[11] runout, opt[12] psi
 *                   (COMB = 1, nreg > 1) eccentric radial runout: tooth i
 *                   sits r_i = runout cos(2 pi i / N + psi) further out, and
 *                   a regenerated chip cut by tooth i over the surface of the
 *                   tooth k passes earlier (i + k mod N in this numbering)
 *                   grows by r_i - r_(i+k); fresh chips are left unchanged
 *   opt[13] tau     (COMB = 1) loop delay (s, at least one step): the command
 *                   uses the motion at t - tau (its delayed terms at
 *                   t - tau - rho T), from the stored history
 * Any part of a chip whose surface the previous tooth did not cut (newly
 * engaged depth, or material just reached through the face) is fresh: its
 * thickness is fz sin(phi) - x cos(phi), with no regeneration.
 */
/* Built twice: COMB = 0 (default; preprocesses to the plain-DVF code, so the
 * default run stays bit-identical to the build before C4) and COMB = 1 (the
 * comb terms; used by tdsim.py only when g2 or kd is non-zero). */
#ifndef COMB
#define COMB 0
#endif
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    double m, c, k, ga, a1, a0, g, vmax;
    /* cutting geometry */
    int nel, spr, nd, fmode;
    const double *ph, *sn, *cs;     /* per (step-in-rev, element) */
    const char *in;
    double phist, phiex, Kt, Kr, fz, dz, dth, dt;
    const double *xh, *vh;          /* history at step boundaries */
    int n;                          /* current step index */
    double fhold;                   /* held force (fmode 0) */
    /* entry options (soft = 0: plain step entry) */
    int soft, nlay, edge, nreg;
    double a, Ta, Tf, f0, R, Tt, vmax_now;
#if COMB
    double g2, kd, cdel;            /* cdel: comb delay in steps (0: exactly nd) */
    double ldel;                    /* loop delay in steps (0: none) */
    int runout, ntooth;
    double ro[16];                  /* radial runout of each tooth */
#endif
} Ctx;

#define NREG_MAX 8

static inline double sat(double v, double lim) {
    if (lim <= 0.0) return v;
    return v > lim ? lim : (v < -lim ? -lim : v);
}

/* depth engaged at time t (soft entry) */
static inline double depth_at(const Ctx *C, double t) {
    if (t < 0.0) return 0.0;
    return C->Ta > 0.0 && t < C->Ta ? C->a * t / C->Ta : C->a;
}

/* feed per tooth at time t, and tool-centre advance s(t) */
static inline double feed_at(const Ctx *C, double t) {
    if (C->Tf <= 0.0 || t >= C->Tf) return C->fz;
    return C->fz * (C->f0 + (1.0 - C->f0) * (t > 0.0 ? t / C->Tf : 0.0));
}
static inline double advance_at(const Ctx *C, double t) {
    double u = C->fz / C->Tt;                 /* nominal feed speed */
    if (t <= 0.0) return 0.0;
    if (C->Tf <= 0.0) return u * t;
    if (t < C->Tf) return u * (C->f0 * t + (1.0 - C->f0) * t * t / (2.0 * C->Tf));
    return u * (C->f0 * C->Tf + (1.0 - C->f0) * C->Tf / 2.0 + (t - C->Tf));
}
/* advance extended to t < 0: feeding at f0 * fz before the ramp starts */
static inline double advance_ext(const Ctx *C, double t) {
    return t < 0.0 ? C->f0 * C->fz / C->Tt * t : advance_at(C, t);
}
static inline double layer_frac(double depth, int j, double dz) {
    double w = (depth - j * dz) / dz;
    return w < 0.0 ? 0.0 : (w > 1.0 ? 1.0 : w);
}

/* x at step j + r (0 <= r <= 1) from the cubic Hermite interpolant of the
 * stored history; zero before t = 0 */
static inline double xhist(const Ctx *C, int j, double r) {
    if (j + 1 <= 0) return 0.0;
    double x0 = j >= 0 ? C->xh[j] : 0.0, v0 = j >= 0 ? C->vh[j] : 0.0;
    double x1 = C->xh[j + 1], v1 = C->vh[j + 1], h = C->dt;
    double r2 = r * r, r3 = r2 * r;
    return (2 * r3 - 3 * r2 + 1) * x0 + (r3 - 2 * r2 + r) * h * v0
         + (-2 * r3 + 3 * r2) * x1 + (r3 - r2) * h * v1;
}

#if COMB
/* x' at step j + r from the same interpolant (its derivative) */
static inline double vhist(const Ctx *C, int j, double r) {
    if (j + 1 <= 0) return 0.0;
    double x0 = j >= 0 ? C->xh[j] : 0.0, v0 = j >= 0 ? C->vh[j] : 0.0;
    double x1 = C->xh[j + 1], v1 = C->vh[j + 1], h = C->dt;
    double r2 = r * r;
    return ((6 * r2 - 6 * r) * x0 + (3 * r2 - 4 * r + 1) * h * v0
          + (-6 * r2 + 6 * r) * x1 + (3 * r2 - 2 * r) * h * v1) / h;
}

/* command before the limit at t_n + tau */
static inline double command(const Ctx *C, const double *z, double tau) {
    int j = C->n - C->nd;
    double r = tau / C->dt;
    if (C->cdel > 0.0 || C->ldel > 0.0) {    /* t - tau - rho T in steps: interval j, fraction r */
        double sj = C->n + tau / C->dt - C->ldel - (C->cdel > 0.0 ? C->cdel : C->nd);
        j = (int)floor(sj);
        r = sj - j;
    }
    if (C->ldel > 0.0) {                     /* the motion itself at t - tau */
        double s0 = C->n + tau / C->dt - C->ldel;
        int j0 = (int)floor(s0);
        double r0 = s0 - j0, x0 = xhist(C, j0, r0), v0 = vhist(C, j0, r0);
        return -C->g * v0 - C->g2 * (v0 - vhist(C, j, r)) - C->kd * (x0 - xhist(C, j, r));
    }
    return -C->g * z[1] - C->g2 * (z[1] - vhist(C, j, r)) - C->kd * (z[0] - xhist(C, j, r));
}
#endif

/* cutting force with multiple regeneration (nreg > 1) */
static double fcut_multi(const Ctx *C, double x, double tau) {
    int n = C->n, s0 = n % C->spr, e, k, K = C->nreg;
    double r = tau / C->dt, f = 0.0, xd[NREG_MAX + 1], S[NREG_MAX + 1];
    double dep[NREG_MAX + 1], rs[NREG_MAX + 1];
    double T = C->nd * C->dt, t = (n + r) * C->dt;
    for (k = 0; k <= K; k++) {
        double tk = t - k * T;
        xd[k] = k ? xhist(C, n - k * C->nd, r) : x;
        S[k] = C->Tf > 0.0 ? advance_ext(C, t) - advance_ext(C, tk) : k * C->fz;
        dep[k] = C->soft ? depth_at(C, tk) : C->a;
        rs[k] = C->edge ? (tk > 0.0 ? 1.0 - advance_at(C, tk) / C->R : 1.0) : -1.0;
    }
    double del = C->dth * r, cd = cos(del), sd = sin(del);
    const double *ph = C->ph + s0 * C->nel, *sn = C->sn + s0 * C->nel, *cs = C->cs + s0 * C->nel;
    for (e = 0; e < C->nel; e++) {
        double phi = ph[e] + del;
        if (phi < C->phist || phi > C->phiex) continue;
        double s = sn[e] * cd + cs[e] * sd;
        double c = cs[e] * cd - sn[e] * sd;
        int j = e % C->nlay;
        double w = C->soft ? layer_frac(dep[0], j, C->dz) : 1.0;
        if (C->edge && !(s >= rs[0])) w = 0.0;
        if (w <= 0.0) continue;
        double wp = C->soft ? layer_frac(dep[1], j, C->dz) : 1.0;
        if (C->edge && !(s >= rs[1])) wp = 0.0;
        if (wp > w) wp = w;
        double hr = S[1] * s - (x - xd[1]) * c, q = 0.0;
#if COMB
        int it = e / C->nlay;                  /* tooth of this element */
        if (C->runout) hr += C->ro[it] - C->ro[(it + 1) % C->ntooth];
#endif
        for (k = 2; k <= K; k++) {             /* only passes that were in the material */
            if (C->soft && layer_frac(dep[k], j, C->dz) <= 0.0) break;
            if (C->edge && !(s >= rs[k])) break;
            double hk = S[k] * s - (x - xd[k]) * c;
#if COMB
            if (C->runout) hk += C->ro[it] - C->ro[(it + k) % C->ntooth];
#endif
            if (hk < hr) hr = hk;
        }
        if (hr > 0) q += wp * hr;
        if (w > wp) {
            double hf = S[1] * s - x * c;     /* fresh surface */
            if (hf > 0) q += (w - wp) * hf;
        }
        f -= (C->Kt * s - C->Kr * c) * C->dz * q;
    }
    return f;
}

/* cutting force at time t_n + tau, with current displacement x */
static double fcut(const Ctx *C, double x, double tau) {
    if (C->nreg > 1) return fcut_multi(C, x, tau);
    int n = C->n, s0 = n % C->spr, e;
    int j = n - C->nd;                       /* delayed interval [j, j+1] */
    double r = tau / C->dt, xd;
    if (j + 1 <= 0) {
        xd = 0.0;
    } else {
        double x0 = j >= 0 ? C->xh[j] : 0.0, v0 = j >= 0 ? C->vh[j] : 0.0;
        double x1 = C->xh[j + 1], v1 = C->vh[j + 1], h = C->dt;
        double r2 = r * r, r3 = r2 * r;
        xd = (2 * r3 - 3 * r2 + 1) * x0 + (r3 - 2 * r2 + r) * h * v0
           + (-2 * r3 + 3 * r2) * x1 + (r3 - r2) * h * v1;
    }
    double dx = x - xd, f = 0.0;
    double del = C->dth * r, cd = cos(del), sd = sin(del);
    const double *ph = C->ph + s0 * C->nel, *sn = C->sn + s0 * C->nel, *cs = C->cs + s0 * C->nel;
    if (!C->soft) {
        for (e = 0; e < C->nel; e++) {
            double phi = ph[e] + del;
            if (phi < C->phist || phi > C->phiex) continue;
            double s = sn[e] * cd + cs[e] * sd;
            double c = cs[e] * cd - sn[e] * sd;
            double h = C->fz * s - dx * c;
            if (h > 0) f -= (C->Kt * s - C->Kr * c) * C->dz * h;
        }
        return f;
    }
    {
        double t = (n + r) * C->dt, tp = t - C->nd * C->dt;
        double dep = depth_at(C, t), depp = depth_at(C, tp);
        /* under a feed ramp the chip is the advance over the last tooth period */
        double fz = C->Tf > 0.0 ? advance_ext(C, t) - advance_ext(C, tp) : C->fz;
        double rs = C->edge ? 1.0 - advance_at(C, t) / C->R : -1.0;
        double rsp = C->edge ? (tp > 0.0 ? 1.0 - advance_at(C, tp) / C->R : 1.0) : -1.0;
        for (e = 0; e < C->nel; e++) {
            double phi = ph[e] + del;
            if (phi < C->phist || phi > C->phiex) continue;
            double s = sn[e] * cd + cs[e] * sd;
            double c = cs[e] * cd - sn[e] * sd;
            int j = e % C->nlay;
            double w = layer_frac(dep, j, C->dz), wp = layer_frac(depp, j, C->dz);
            if (C->edge) {
                if (!(s >= rs)) w = 0.0;
                if (!(s >= rsp)) wp = 0.0;
            }
            if (wp > w) wp = w;
            if (w <= 0.0) continue;
            double hr = fz * s - dx * c, hf = fz * s - x * c, q = 0.0;
            if (hr > 0) q += wp * hr;
            if (hf > 0) q += (w - wp) * hf;
            f -= (C->Kt * s - C->Kr * c) * C->dz * q;
        }
    }
    return f;
}

static inline void deriv(const Ctx *C, const double *z, double tau, double *dz, double *fact) {
#if COMB
    double V = sat(command(C, z, tau), C->vmax_now);
#else
    double V = sat(-C->g * z[1], C->vmax_now);
#endif
    double fa = C->ga * (V - C->a1 * z[3] - C->a0 * z[2]);
    double fc = C->fmode ? fcut(C, z[0], tau) : C->fhold;
    dz[0] = z[1];
    dz[1] = (fc + fa - C->c * z[1] - C->k * z[0]) / C->m;
    dz[2] = z[3];
    dz[3] = V - C->a1 * z[3] - C->a0 * z[2];
    if (fact) *fact = fa;
}

static void rk4(const Ctx *C, double *z, double tau, double h) {
    double k1[4], k2[4], k3[4], k4[4], t[4];
    int i;
    deriv(C, z, tau, k1, NULL);
    for (i = 0; i < 4; i++) t[i] = z[i] + 0.5 * h * k1[i];
    deriv(C, t, tau + 0.5 * h, k2, NULL);
    for (i = 0; i < 4; i++) t[i] = z[i] + 0.5 * h * k2[i];
    deriv(C, t, tau + 0.5 * h, k3, NULL);
    for (i = 0; i < 4; i++) t[i] = z[i] + h * k3[i];
    deriv(C, t, tau + h, k4, NULL);
    for (i = 0; i < 4; i++) z[i] += h / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
}

/* summary[0] = max |g x'|  (demanded command before the limit; with COMB the
 *              comb terms are included)
 * summary[1] = max |f_act| (actuator force produced, N)
 * summary[2] = max |x|     (m)
 * summary[3] = max |V|     (applied command after the limit)
 * summary[4] = 1 if the run diverged (|x| > 1e-2 m), else 0
 * Maxima are taken at the step boundaries.  If store != 0 the per-step
 * histories are written to the out arrays (length nrev*spr + 1).
 */
int simulate(double rpm, double a, double g,
             double m, double c, double k,
             double ga, double a1, double a0,
             double Kt, double Kr, double fz, double D, int N, double helix,
             double phist, double phiex,
             int spr, int nrev, int nlay, int nsub, double vmax, int fmode,
             int store, double *x_out, double *v_out, double *fact_out,
             double *fcut_out, double *vcmd_out, double *summary,
             const double *opt, const double *pert)
{
    const double twopi = 2.0 * M_PI;
    int nsteps = spr * nrev;
    double dt = 60.0 / (rpm * spr);
    int nd = spr / N;
    double dz = a / nlay, R = 0.5 * D;
    int nel = N * nlay;
    double *ph = malloc(sizeof(double) * spr * nel);
    double *sn = malloc(sizeof(double) * spr * nel);
    double *cs = malloc(sizeof(double) * spr * nel);
    char *in = malloc(spr * nel);
    double *xh = calloc(nsteps + 1, sizeof(double));
    double *vh = calloc(nsteps + 1, sizeof(double));
    int s, i, j, e, n, kk;
    double z[4] = {0, 0, 0, 0};
    double maxd = 0, maxfa = 0, maxx = 0, maxv = 0;
    int diverged = 0;
    Ctx C = {m, c, k, ga, a1, a0, g, vmax, nel, spr, nd, fmode, ph, sn, cs, in,
             phist, phiex, Kt, Kr, fz, dz, twopi / spr, dt, xh, vh, 0, 0.0};
    double Trev = 60.0 / rpm;
    int kick_step = -1, clip_on = 0;
    double kick_dv = 0.0;
    C.nlay = nlay; C.a = a; C.R = R; C.Tt = Trev / N; C.f0 = 0.0;
    C.Ta = 0.0; C.Tf = 0.0; C.edge = 0; C.nreg = 1;
    if (opt) {
        C.Ta = opt[0] > 0.0 ? opt[0] * Trev : 0.0;
        C.Tf = opt[1] > 0.0 ? opt[1] * Trev : 0.0;
        C.f0 = opt[2];
        C.edge = opt[3] != 0.0;
        kick_step = (int)opt[4];
        kick_dv = opt[5];
        clip_on = (int)opt[6];
        C.nreg = (int)opt[7] < 1 ? 1 : ((int)opt[7] > NREG_MAX ? NREG_MAX : (int)opt[7]);
#if COMB
        C.g2 = opt[8];
        C.kd = opt[9];
        C.cdel = opt[10] > 0.0 && opt[10] != 1.0 ? opt[10] * nd : 0.0;
        C.ldel = opt[13] > 0.0 ? opt[13] / dt : 0.0;
        C.ntooth = N;
        C.runout = opt[11] != 0.0 && N <= 16;
        for (i = 0; i < N && i < 16; i++) C.ro[i] = opt[11] * cos(twopi * i / N + opt[12]);
#endif
    }
    C.soft = C.Ta > 0.0 || C.Tf > 0.0 || C.edge;

    for (s = 0; s < spr; s++) {
        double th = twopi * s / spr;
        for (i = 0; i < N; i++)
            for (j = 0; j < nlay; j++) {
                double zj = (j + 0.5) * dz;
                double phi = th + i * twopi / N - zj * tan(helix) / R;
                phi = fmod(phi, twopi);
                if (phi < 0) phi += twopi;
                e = s * nel + i * nlay + j;
                ph[e] = phi;
                sn[e] = sin(phi);
                cs[e] = cos(phi);
                in[e] = (phi >= phist && phi <= phiex);
            }
    }

    for (n = 0; n < nsteps; n++) {
        double f, fa, dzz[4];
        C.n = n;
        C.vmax_now = n >= clip_on ? vmax : 0.0;
        if (n == kick_step) {
            z[1] += kick_dv;
            if (pert) {
                int jj;
                for (jj = 0; jj < 4; jj++) z[jj] += pert[jj];
                for (jj = 1; jj <= nd && n - jj >= 0; jj++) {
                    xh[n - jj] += pert[4 + jj];
                    vh[n - jj] += pert[5 + nd + jj];
                }
            }
            xh[n] = z[0];
            vh[n] = z[1];
        }
        f = fcut(&C, z[0], 0.0);
        C.fhold = f;
        deriv(&C, z, 0.0, dzz, &fa);
        {
#if COMB
            double dem = fabs(command(&C, z, 0.0));
            double V = sat(command(&C, z, 0.0), C.vmax_now);
#else
            double dem = fabs(g * z[1]);
            double V = sat(-g * z[1], C.vmax_now);
#endif
            if (dem > maxd) maxd = dem;
            if (fabs(fa) > maxfa) maxfa = fabs(fa);
            if (fabs(z[0]) > maxx) maxx = fabs(z[0]);
            if (fabs(V) > maxv) maxv = fabs(V);
            if (store) {
                x_out[n] = z[0]; v_out[n] = z[1]; fact_out[n] = fa;
                fcut_out[n] = f; vcmd_out[n] = V;
            }
        }
        for (kk = 0; kk < nsub; kk++) rk4(&C, z, kk * dt / nsub, dt / nsub);
        xh[n + 1] = z[0];
        vh[n + 1] = z[1];
        if (!(fabs(z[0]) < 1e-2)) { diverged = 1; break; }
    }
    if (store && !diverged) {
        double dzz[4], fa;
        C.n = nsteps;
        C.fhold = 0.0;
        C.vmax_now = nsteps >= clip_on ? vmax : 0.0;
        deriv(&C, z, 0.0, dzz, &fa);
        x_out[nsteps] = z[0]; v_out[nsteps] = z[1]; fact_out[nsteps] = fa;
        fcut_out[nsteps] = nsteps >= nd ? fcut(&C, z[0], 0.0) : 0.0;
#if COMB
        vcmd_out[nsteps] = sat(command(&C, z, 0.0), C.vmax_now);
#else
        vcmd_out[nsteps] = sat(-g * z[1], C.vmax_now);
#endif
    }
    summary[0] = maxd; summary[1] = maxfa; summary[2] = maxx; summary[3] = maxv;
    summary[4] = diverged;
    free(ph); free(sn); free(cs); free(in); free(xh); free(vh);
    return diverged;
}
