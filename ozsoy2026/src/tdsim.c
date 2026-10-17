/* Time-domain milling simulation of the 1-DOF flexible workpiece under DVF.
 *
 * Regenerative chip thickness per axial layer and tooth (Sec. 2.1):
 *   h = fz sin(phi) - (x(t) - x(t-T)) cos(phi),  force = 0 when h <= 0
 * Force on the workpiece along y:  f = -sum (Kt sin(phi) - Kr cos(phi)) dz h
 * Closed loop:  m x'' + c x' + k x = f_cut + f_act
 *               q'' + a1 q' + a0 q = V,   f_act = ga (V - a1 q' - a0 q)
 *               V = sat(-g x', vmax)
 * The cutting force is held over each step (zero-order hold); the loop is
 * integrated with RK4 over nsub sub-steps, so the controller stays
 * continuous and an optional voltage limit acts inside the loop.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    double m, c, k, ga, a1, a0, g, vmax, fcut;
} Loop;

static inline double sat(double v, double lim) {
    if (lim <= 0.0) return v;
    return v > lim ? lim : (v < -lim ? -lim : v);
}

static inline void deriv(const Loop *L, const double *z, double *dz, double *fact) {
    double V = sat(-L->g * z[1], L->vmax);
    double fa = L->ga * (V - L->a1 * z[3] - L->a0 * z[2]);
    dz[0] = z[1];
    dz[1] = (L->fcut + fa - L->c * z[1] - L->k * z[0]) / L->m;
    dz[2] = z[3];
    dz[3] = V - L->a1 * z[3] - L->a0 * z[2];
    if (fact) *fact = fa;
}

static void rk4(const Loop *L, double *z, double h) {
    double k1[4], k2[4], k3[4], k4[4], t[4];
    int i;
    deriv(L, z, k1, NULL);
    for (i = 0; i < 4; i++) t[i] = z[i] + 0.5 * h * k1[i];
    deriv(L, t, k2, NULL);
    for (i = 0; i < 4; i++) t[i] = z[i] + 0.5 * h * k2[i];
    deriv(L, t, k3, NULL);
    for (i = 0; i < 4; i++) t[i] = z[i] + h * k3[i];
    deriv(L, t, k4, NULL);
    for (i = 0; i < 4; i++) z[i] += h / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
}

/* summary[0] = max |g x'|  (demanded actuator command, units of g*m/s)
 * summary[1] = max |f_act| (actuator force actually produced, N)
 * summary[2] = max |x|     (m)
 * summary[3] = max |V|     (applied command after the limit)
 * summary[4] = 1 if the run diverged (|x| > 1e-2 m), else 0
 * If store != 0, the per-step histories are written to the out arrays
 * (length nrev*spr + 1).
 */
int simulate(double rpm, double a, double g,
             double m, double c, double k,
             double ga, double a1, double a0,
             double Kt, double Kr, double fz, double D, int N, double helix,
             double phist, double phiex,
             int spr, int nrev, int nlay, int nsub, double vmax,
             int store, double *x_out, double *v_out, double *fact_out,
             double *fcut_out, double *vcmd_out, double *summary)
{
    const double twopi = 2.0 * M_PI;
    int nsteps = spr * nrev;
    double dt = 60.0 / (rpm * spr);
    int nd = spr / N;                     /* delay in steps (tooth period) */
    double dz = a / nlay;
    double R = 0.5 * D;
    int nel = N * nlay;
    double *sn = malloc(sizeof(double) * spr * nel);
    double *cs = malloc(sizeof(double) * spr * nel);
    char *in = malloc(spr * nel);
    double *xh = calloc(nsteps + 1, sizeof(double));
    int s, i, j, e, n, kk;
    Loop L = {m, c, k, ga, a1, a0, g, vmax, 0.0};
    double z[4] = {0, 0, 0, 0};
    double maxd = 0, maxfa = 0, maxx = 0, maxv = 0;
    int diverged = 0;

    for (s = 0; s < spr; s++) {
        double th = twopi * s / spr;
        for (i = 0; i < N; i++)
            for (j = 0; j < nlay; j++) {
                double zj = (j + 0.5) * dz;
                double phi = th + i * twopi / N - zj * tan(helix) / R;
                phi = fmod(phi, twopi);
                if (phi < 0) phi += twopi;
                e = s * nel + i * nlay + j;
                sn[e] = sin(phi);
                cs[e] = cos(phi);
                in[e] = (phi >= phist && phi <= phiex);
            }
    }

    for (n = 0; n < nsteps; n++) {
        int s0 = n % spr;
        double xd = (n >= nd) ? xh[n - nd] : 0.0;
        double dx = z[0] - xd;
        double f = 0.0;
        for (e = 0; e < nel; e++) {
            int q = s0 * nel + e;
            if (!in[q]) continue;
            double h = fz * sn[q] - dx * cs[q];
            if (h > 0) f -= (Kt * sn[q] - Kr * cs[q]) * dz * h;
        }
        L.fcut = f;
        {
            double dzz[4], fa;
            deriv(&L, z, dzz, &fa);
            double dem = fabs(g * z[1]);
            double V = sat(-g * z[1], vmax);
            if (dem > maxd) maxd = dem;
            if (fabs(fa) > maxfa) maxfa = fabs(fa);
            if (fabs(z[0]) > maxx) maxx = fabs(z[0]);
            if (fabs(V) > maxv) maxv = fabs(V);
            if (store) {
                x_out[n] = z[0]; v_out[n] = z[1]; fact_out[n] = fa;
                fcut_out[n] = f; vcmd_out[n] = V;
            }
        }
        for (kk = 0; kk < nsub; kk++) rk4(&L, z, dt / nsub);
        xh[n + 1] = z[0];
        if (!(fabs(z[0]) < 1e-2)) { diverged = 1; break; }
    }
    if (store && !diverged) {
        double dzz[4], fa;
        deriv(&L, z, dzz, &fa);
        x_out[nsteps] = z[0]; v_out[nsteps] = z[1]; fact_out[nsteps] = fa;
        fcut_out[nsteps] = L.fcut; vcmd_out[nsteps] = sat(-g * z[1], vmax);
    }
    summary[0] = maxd; summary[1] = maxfa; summary[2] = maxx; summary[3] = maxv;
    summary[4] = diverged;
    free(sn); free(cs); free(in); free(xh);
    return diverged;
}
