"""
dk_synthesis.py — Valeur singuliere structuree et iteration D-K (Eqs. 28-29)
============================================================================
Le papier ecrit exactement :

  Eq. (28)   sup_w mu_D( T_Pmu(jw) ) < gamma     (stabilite robuste)
  Eq. (29)   inf_K sup_w inf_D  sigma_bar( D(w) T_Pmu(jw) D^-1(w) )
             avec D = { diag(D1,...,DS, d_{S+1} I, ..., d_{S+F} I) }
  "For a given D(w), the solution K of inf_K sigma_bar(D T D^-1) is a standard
   H-infinity problem. With the obtained K, find D(w) ... Repeat the process
   until the obtained D(w) in two loops has no obvious difference."

C'est l'iteration D-K classique de Packard & Doyle [80]. Ce module l'implante
integralement, sans MATLAB :

  * mu_upper_bound        : inf_D sigma_bar(D M D^-1) a une frequence donnee
                            (optimisation convexe sur log d, L-BFGS-B) ;
  * mu_curve              : la borne sur toute une grille de frequences ;
  * fit_dscale            : approximation rationnelle STABLE et A PHASE
                            MINIMALE de d_i(w) (parametrage par poles/zeros
                            positifs, moindres carres sur log|d|) ;
  * scale_plant           : plant generalise pondere D_hat P D_hat^-1 ;
  * dk_iteration          : la boucle complete, H-infini via slycot (SB10AD).

Les blocs sont decrits par (type, n_out_du_plant, n_in_du_plant) :
  'F' bloc plein complexe (ici Delta_Pa, 1x2), 'S' bloc scalaire (les dix
  parametres reels de l'Eq. 22 — traites en complexes comme dans l'Eq. 29),
  'P' bloc de performance (echelle figee a 1).
"""
import numpy as np
import control as ct
from scipy.optimize import minimize, least_squares


# ---------------------------------------------------------------------------
# Mise a l'echelle numerique du plant generalise
# ---------------------------------------------------------------------------
def prepare_plant(P, w0=2 * np.pi * 1000.0, u0=100.0, y0=1e-6, balance=True,
                  alpha_shift=0.0):
    """Rend le plant generalise numeriquement traitable par SB10AD.

    Trois transformations, toutes EXACTEMENT reversibles et sans effet sur le
    probleme (ni sur gamma, ni sur mu) :
      1. changement d'unite de temps  s = w0 * s~   (A, B divises par w0) ;
      2. changement d'unite d'entree/sortie du correcteur : la commande est
         exprimee en u0 volts et la mesure en y0 metres ;
      3. equilibrage diagonal des etats (SLICOT TB01ID).
    Retour (P_scaled_ss, info) ; `unscale_controller` defait 1 et 2.
    """
    A = np.array(P['A'], float); B = np.array(P['B'], float)
    C = np.array(P['C'], float); D = np.array(P['D'], float)
    i_up = B.shape[1] - P['ncon']
    o_y = C.shape[0] - P['nmeas']
    d0 = P.get('dscale_init')
    if d0 is not None:                      # D-scale constante initiale
        dy, du = _scale_vectors(P['blocks'], np.asarray(d0, float))
        C[:len(dy), :] *= dy[:, None]
        D[:len(dy), :] *= dy[:, None]
        B[:, :len(du)] /= du[None, :]
        D[:, :len(du)] /= du[None, :]
    A = (A + alpha_shift * np.eye(A.shape[0])) / w0
    B = B / w0
    B[:, i_up:] *= u0
    D[:, i_up:] *= u0
    C[o_y:, :] /= y0
    D[o_y:, :] /= y0
    scale = np.ones(A.shape[0])
    if balance:
        try:
            from slycot import tb01id
            _, A, B, C, scale = tb01id(A.shape[0], B.shape[1], C.shape[0],
                                       0.0, A, B, C, job='A')
        except Exception:
            pass
    return (ct.ss(A, B, C, D),
            dict(w0=w0, u0=u0, y0=y0, scale=scale, alpha_shift=alpha_shift))


def unscale_controller(K, info):
    """Correcteur reel (volts par metre, temps en secondes)."""
    w0, u0, y0 = info['w0'], info['u0'], info['y0']
    A = np.array(K.A, float) * w0 - info.get('alpha_shift', 0.0) * np.eye(len(K.A))
    B = np.array(K.B, float) * w0 / y0
    C = np.array(K.C, float) * u0
    D = np.array(K.D, float) * u0 / y0
    return ct.ss(A, B, C, D)


# ---------------------------------------------------------------------------
def frf(sys, w):
    """Reponse frequentielle d'un systeme ct.ss sur les pulsations w."""
    A, B, C, D = np.asarray(sys.A), np.asarray(sys.B), np.asarray(sys.C), \
        np.asarray(sys.D)
    n = A.shape[0]
    out = np.empty((len(w), C.shape[0], B.shape[1]), complex)
    I = np.eye(n)
    for k, wk in enumerate(w):
        out[k] = C @ np.linalg.solve(1j * wk * I - A, B) + D
    return out


def _scale_vectors(blocks, d):
    """Vecteurs diagonaux (D_y, D_u) a partir des echelles par bloc."""
    dy, du, i = [], [], 0
    for (kind, no, ni) in blocks:
        if kind == 'P':
            val = 1.0
        else:
            val = d[i]
            i += 1
        dy += [val] * no
        du += [val] * ni
    return np.array(dy), np.array(du)


def n_scales(blocks):
    return sum(1 for b in blocks if b[0] != 'P')


def mu_upper_bound(M, blocks, d0=None, tol=1e-8):
    """inf_D sigma_bar(D_y M D_u^-1) : borne superieure de mu (Eq. 29).

    M : matrice complexe (n_out x n_in) de la boucle fermee sur les canaux
        d'incertitude ET de performance.
    Retour (mu, d) avec d les echelles optimales (bloc de performance = 1).
    """
    ns = n_scales(blocks)
    if ns == 0:
        return float(np.linalg.norm(M, 2)), np.array([])
    x0 = np.zeros(ns) if d0 is None else np.log(np.maximum(d0, 1e-12))

    def cost(x):
        dy, du = _scale_vectors(blocks, np.exp(x))
        return float(np.linalg.norm((dy[:, None] * M) / du[None, :], 2))

    res = minimize(cost, x0, method='L-BFGS-B',
                   options=dict(maxiter=200, ftol=tol, gtol=1e-10))
    return float(res.fun), np.exp(res.x)


def mu_curve(T, blocks, w, warm=True):
    """Borne superieure de mu sur une grille (T : ct.ss ou tableau de FRF)."""
    H = frf(T, w) if hasattr(T, 'A') else np.asarray(T)
    mus = np.empty(len(w))
    D = np.empty((len(w), n_scales(blocks)))
    d0 = None
    for k in range(len(w)):
        mus[k], d = mu_upper_bound(H[k], blocks, d0 if warm else None)
        D[k] = d
        d0 = d
    return mus, D


# ---------------------------------------------------------------------------
def fit_dscale(w, mag, order=2, dyn_range=1e3):
    """Ajuste |d(jw)| par une fraction rationnelle stable a phase minimale.

        d(s) = k * prod (s + z_i) / prod (s + p_i),   z_i > 0, p_i > 0

    Moindres carres sur log|d|. Retour un ct.ss (SISO, bipropre).
    """
    mag = np.maximum(np.asarray(mag, float), 1e-12)
    # plage dynamique bornee : une echelle qui varie de plus de 60 dB sur la
    # bande ne peut pas etre approchee par une fraction bipropre d'ordre bas,
    # et une D-scale sous-optimale reste une BORNE SUPERIEURE valide de mu.
    mag = np.clip(mag, mag.max() / dyn_range, mag.max())
    y = np.log(mag)
    if order == 0:
        k = float(np.exp(np.mean(y)))
        return ct.ss([], [], [], [[k]])
    lw = np.log(np.maximum(w, 1e-6))
    z0 = np.exp(np.linspace(lw[0] + 0.5, lw[-1] - 0.5, order))
    x0 = np.concatenate([[np.mean(y)], np.log(z0), np.log(z0)])
    lo = np.concatenate([[np.log(1e-4)], np.full(2 * order, lw[0] - 3.0)])
    hi = np.concatenate([[np.log(1e4)], np.full(2 * order, lw[-1] + 3.0)])
    x0 = np.clip(x0, lo + 1e-6, hi - 1e-6)

    def model(x):
        k = x[0]
        zs = np.exp(x[1:1 + order])
        ps = np.exp(x[1 + order:])
        v = k * np.ones_like(w)
        for zi in zs:
            v = v + 0.5 * np.log(w**2 + zi**2)
        for pi in ps:
            v = v - 0.5 * np.log(w**2 + pi**2)
        return v

    res = least_squares(lambda x: model(x) - y, x0, bounds=(lo, hi),
                        max_nfev=4000)
    k = float(np.exp(res.x[0]))
    zs = np.exp(res.x[1:1 + order])
    ps = np.exp(res.x[1 + order:])
    num = np.poly(-zs) * k
    den = np.poly(-ps)
    return ct.ss(ct.tf(num, den))


def _blkdiag_ss(systems):
    """Concatenation diagonale de systemes ct.ss (au moins un)."""
    out = systems[0]
    for s in systems[1:]:
        out = ct.append(out, s)
    return out


def dscale_systems(blocks, d_fits):
    """(D_y(s), D_u(s)) en etat-espace a partir des ajustements par bloc."""
    ys, us, i = [], [], 0
    for (kind, no, ni) in blocks:
        if kind == 'P':
            ys += [ct.ss([], [], [], np.eye(no))]
            us += [ct.ss([], [], [], np.eye(ni))]
        else:
            di = d_fits[i]
            i += 1
            ys += [_rep(di, no)]
            us += [_rep(di, ni)]
    return _blkdiag_ss(ys), _blkdiag_ss(us)


def _rep(sys, n):
    """diag(sys, ..., sys) n fois."""
    out = sys
    for _ in range(n - 1):
        out = ct.append(out, sys)
    return out


def scale_plant(P, blocks, d_fits, nmeas=1, ncon=1):
    """Plant generalise pondere : P_hat = diag(D_y, I) P diag(D_u^-1, I)."""
    Dy, Du = dscale_systems(blocks, d_fits)
    ny_unc, nu_unc = Dy.noutputs, Du.ninputs
    Dy_a = ct.append(Dy, ct.ss([], [], [], np.eye(nmeas)))
    Du_i = ct.append(_inv_ss(Du), ct.ss([], [], [], np.eye(ncon)))
    return ct.ss(Dy_a * ct.ss(P) * Du_i)


def _inv_ss(sys):
    """Inverse d'un systeme carre bipropre."""
    A, B, C, D = (np.asarray(sys.A), np.asarray(sys.B),
                  np.asarray(sys.C), np.asarray(sys.D))
    Di = np.linalg.inv(D)
    return ct.ss(A - B @ Di @ C, B @ Di, -Di @ C, Di)


# ---------------------------------------------------------------------------
def dk_iteration(P, blocks, w, nmeas=1, ncon=1, n_iter=4, orders=(0, 2, 2, 2),
                 verbose=True, reduce_to=None):
    """Iteration D-K complete (Eq. 29). Retour dict(K, mu, history)."""
    Pss = ct.ss(P['A'], P['B'], P['C'], P['D']) if isinstance(P, dict) else P
    d_fits = None
    history = []
    best = None
    for it in range(n_iter):
        Phat = Pss if d_fits is None else scale_plant(Pss, blocks, d_fits,
                                                      nmeas, ncon)
        try:
            K, CL, gam, _ = ct.hinfsyn(Phat, nmeas, ncon)
        except Exception as exc:                       # pragma: no cover
            if verbose:
                print(f"   [D-K {it}] hinfsyn a echoue : {exc}")
            break
        # boucle fermee NON ponderee, pour l'analyse mu
        T = ct.ss(ct.ss(Pss).lft(ct.ss(K), ncon, nmeas))
        mus, D = mu_curve(T, blocks, w)
        mu_peak = float(np.max(mus))
        if verbose:
            print(f"   [D-K {it}] gamma = {gam:8.4f}   mu_peak = {mu_peak:8.4f}"
                  f"   ordre K = {K.nstates}")
        history.append(dict(iter=it, gamma=float(gam), mu=mu_peak,
                            order=int(K.nstates), mus=mus.copy()))
        if best is None or mu_peak < best['mu']:
            best = dict(K=K, mu=mu_peak, T=T, mus=mus, D=D, iter=it,
                        gamma=float(gam))
        if it == n_iter - 1:
            break
        order = orders[min(it + 1, len(orders) - 1)]
        # normalisation par UNE constante globale : les echelles relatives
        # entre blocs portent toute l'information, les normaliser colonne par
        # colonne detruirait la ponderation.
        Dn = D / np.median(D)
        d_fits = [fit_dscale(w, Dn[:, j], order) for j in range(D.shape[1])]
    if best is None:
        raise RuntimeError("aucune iteration D-K n'a abouti")
    if reduce_to is not None and best['K'].nstates > reduce_to:
        best['K_full'] = best['K']
        best['K'] = reduce_controller(best['K'], reduce_to)
    best['history'] = history
    return best


def reduce_controller(K, order):
    """Reduction equilibree du correcteur (troncature de Moore)."""
    try:
        Kr = ct.balred(ct.ss(K), order, method='truncate')
        return ct.ss(Kr)
    except Exception:
        return K
