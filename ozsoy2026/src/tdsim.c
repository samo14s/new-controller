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
 *                   below the current depth
 *   opt[1] ramp_f   feed per tooth ramped over this many revolutions,
 *   opt[2] f0       starting from f0 * fz
 *   opt[3] edge     1: straight entry through a workpiece face normal to the
 *                   feed, touched by the tool periphery at t = 0; an edge
 *                   point at angle phi is in the material once the tool
 *                   centre has advanced s(t) >= R (1 - sin phi)
 *   opt[4] kick_step, opt[5] kick_dv
 *                   velocity kick added to x' at the start of that step
 *   opt[6] clip_on  the command limit vmax applies from this step on
 * Any part of a chip whose surface the previous tooth did not cut (newly
 * engaged depth, or material just reached through the face) is fresh: its
 * thickness is fz sin(phi) - x cos(phi), with no regeneration.
 */
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
    int soft, nlay, edge;
    double a, Ta, Tf, f0, R, Tt, vmax_now;
} Ctx;

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
static inline double layer_frac(double depth, int j, double dz) {
    double w = (depth - j * dz) / dz;
    return w < 0.0 ? 0.0 : (w > 1.0 ? 1.0 : w);
}

/* cutting force at time t_n + tau, with current displacement x */
static double fcut(const Ctx *C, double x, double tau) {
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
        double dep = depth_at(C, t), depp = depth_at(C, tp), fz = feed_at(C, t);
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
    double V = sat(-C->g * z[1], C->vmax_now);
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

/* summary[0] = max |g x'|  (demanded command, units of g * m/s)
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
             const double *opt)
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
    C.Ta = 0.0; C.Tf = 0.0; C.edge = 0;
    if (opt) {
        C.Ta = opt[0] > 0.0 ? opt[0] * Trev : 0.0;
        C.Tf = opt[1] > 0.0 ? opt[1] * Trev : 0.0;
        C.f0 = opt[2];
        C.edge = opt[3] != 0.0;
        kick_step = (int)opt[4];
        kick_dv = opt[5];
        clip_on = (int)opt[6];
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
        if (n == kick_step) { z[1] += kick_dv; vh[n] = z[1]; }
        f = fcut(&C, z[0], 0.0);
        C.fhold = f;
        deriv(&C, z, 0.0, dzz, &fa);
        {
            double dem = fabs(g * z[1]);
            double V = sat(-g * z[1], C.vmax_now);
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
        vcmd_out[nsteps] = sat(-g * z[1], C.vmax_now);
    }
    summary[0] = maxd; summary[1] = maxfa; summary[2] = maxx; summary[3] = maxv;
    summary[4] = diverged;
    free(ph); free(sn); free(cs); free(in); free(xh); free(vh);
    return diverged;
}
