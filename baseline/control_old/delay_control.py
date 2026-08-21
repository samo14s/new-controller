"""
delay_control.py — Commande a retard actif et correcteur combine (Section 3.4)
==============================================================================
Eq. (30)    u_Pd(t) = K_Pp y_Pmu(t - tau) + K_Pd y'_Pmu(t - tau)

Eq. (31)    F_PId = alpha4(t) D_P^T D_P y_P(t - tau)
                    + H_Pe K_Pp y_Pmu(t - tau) + H_Pe K_Pd y'_Pmu(t - tau)
                    + S_P(x, z, t)

Le papier ne donne AUCUNE valeur de K_Pp ni de K_Pd ("After designing the
coefficients K_Pp and K_Pd suitably"). Deux voies sont fournies ici :

  1. `pd_gains_cancellation` : la valeur ANALYTIQUE de l'Eq. (31), obtenue en
     annulant au sens des moindres carres la regeneration PROPRE des modes — c'est la traduction
     directe de la phrase du papier "If the active control force satisfy
     -alpha4(t) D_P^T(x,z) y_P(t-tau), F_PI will have no time delay part".
     L'annulation EXACTE est impossible avec une pastille unique (deux
     matrices de rang 1 de directions differentes), mais la partie qui compte
     — la regeneration propre de chaque mode, sur la diagonale modale — est
     annulee a 85 % par
              K_Pp = -alpha40 * sum_i D_i^2 g_i / sum_i g_i^2,  g_i = H_i D_obs,i.

  2. `optimize_pd_gains` : balayage (K_Pp, K_Pd) maximisant la limite de coupe
     en boucle fermee au sens de Floquet, avec le correcteur robuste dans la
     boucle — la lecture "suitably" du papier.

`CombinedController` realise u = u_robuste(y) + u_retard(y(t-tau), y'(t-tau))
pour la simulation temporelle (correcteur LTI discretise par bloqueur d'ordre
zero, memoire circulaire pour le retard).
"""
import numpy as np
from scipy.signal import cont2discrete


# ---------------------------------------------------------------------------
def pd_gains_cancellation(U, kd_ratio=0.0, target='diagonal'):
    """(K_Pp, K_Pd) de l'Eq. (31) : annulation du terme regeneratif retarde.

    Le papier : "If the active control force satisfy -alpha4(t) D_P^T(x,z)
    y_P(t-tau), F_PI will have no time delay part."  Ecrite sur les modes, la
    condition porte sur deux matrices de rang 1 :

        a annuler :  alpha40 * D_nom D_nom^T      (regeneratif)
        disponible :  K_Pp   * H_Pe  D_obs^T      (une seule pastille)

    `target` fixe le sens donne a "annuler" :

      'diagonal' (defaut) : moindres carres sur les elements DIAGONAUX
            (i, i), c'est-a-dire sur la regeneration PROPRE de chaque mode.
            C'est la seule lecture qui corresponde au mecanisme de broutement :
            l'instabilite regenerative de chaque mode est alimentee par son
            propre terme alpha4 D_i^2 q_i(t-tau) ; les elements hors diagonale
            ne font que coupler les modes entre eux.
      'mode1' / 'mode2' : annulation EXACTE du mode indique.
      'frobenius' : moindres carres sur la matrice ENTIERE. A EVITER — la
            norme est dominee par les elements hors diagonale, qu'une pastille
            unique ne peut pas produire, et la projection s'effondre alors
            vers ~0 (2.5e4 V/m ici, soit un effet nul). Conserve pour montrer
            l'ecart : c'est cette lecture qui rend l'Eq. (31) "inoperante".
    """
    H, D_obs, D_nom = U.H, U.D_obs, U.D_nom
    a40 = U.par['alpha40'] * U.sign
    if target == 'frobenius':
        num = float(H @ D_nom) * float(D_nom @ D_obs)
        den = float(H @ H) * float(D_obs @ D_obs)
        Kpp = -a40 * num / den
    elif target in ('mode1', 'mode2'):
        i = 0 if target == 'mode1' else 1
        Kpp = -a40 * D_nom[i] ** 2 / (H[i] * D_obs[i])
    else:                                            # 'diagonal'
        g = H * D_obs                                # element (i, i) par volt
        Kpp = -a40 * float(D_nom ** 2 @ g) / float(g @ g)
    w1 = U.par['omega'][0]
    return float(Kpp), float(kd_ratio * Kpp / w1)


def cancellation_ratio(U, Kpp, target='diagonal'):
    """Fraction du terme regeneratif reellement annulee.

    Mesuree sur la DIAGONALE modale (la regeneration propre de chaque mode),
    qui est ce que la commande peut et doit annuler ; `target='frobenius'`
    donne l'ancienne mesure sur la matrice entiere.
    """
    H, D_obs, D_nom = U.H, U.D_obs, U.D_nom
    a40 = U.par['alpha40'] * U.sign
    if target == 'frobenius':
        tgt = a40 * np.outer(D_nom, D_nom)
        res = tgt + Kpp * np.outer(H, D_obs)
    else:
        tgt = a40 * D_nom ** 2
        res = tgt + Kpp * (H * D_obs)
    return float(1.0 - np.linalg.norm(res) / np.linalg.norm(tgt))


# ---------------------------------------------------------------------------
def parasitic_coupling(U):
    """Couplage retarde HORS DIAGONALE injecte par la pastille, par volt.

    C'est la limite physique de l'Eq. (31). Le terme regeneratif a annuler est
    `alpha4 D D^T` : ses elements hors diagonale valent alpha4*D_1*D_2, faibles
    (et nuls a mi-passe, ou le mode 2 a un noeud). La commande, elle, ne peut
    produire que `K_Pp H D_obs^T`, dont les elements hors diagonale valent
    K_Pp*H_i*D_obs,j et ne correspondent A RIEN dans la dynamique reelle : ils
    sont un couplage retarde PARASITE entre modes. Au gain de l'Eq. (31) ils
    atteignent ici ~1e6, soit une centaine de fois la regeneration croisee
    (~9e3). C'est ce parasite, et non l'annulation elle-meme, qui plafonne
    K_Pp bien au-dessous de la valeur d'annulation complete.
    """
    off = np.array([U.H[0] * U.D_obs[1], U.H[1] * U.D_obs[0]])
    a40 = U.par['alpha40'] * U.sign
    reg_off = a40 * U.D_nom[0] * U.D_nom[1]
    return off, float(reg_off)


def suitable_delay_gain(plate, rpm, ap, sign=1.0, n_modes=2, m=200,
                        positions=(0.0, 0.25, 0.5, 0.75, 1.0),
                        grid=None, verbose=False):
    """K_Pp "designed suitably" (mot du papier) : le gain qui MINIMISE le pire
    taux de croissance de la boucle a retard SEULE le long de la passe.

    C'est le seul critere qui reproduise la Fig. 14(c) du papier — "the single
    active time delay control is still unstable at the start and end
    positions" — c'est-a-dire instable aux deux extremites et stable au
    milieu. Appliquer litteralement le gain d'annulation de l'Eq. (31) donne
    au contraire une divergence PARTOUT, parce que le couplage parasite
    (cf. `parasitic_coupling`) croit avec K_Pp aussi vite que l'annulation.
    """
    from closed_loop import period_maps, spectral_radius
    grid = np.linspace(1e5, 1.8e6, 18) if grid is None else np.asarray(grid)
    best, best_w, hist = 0.0, np.inf, []
    for k in grid:
        rates = []
        for f in positions:
            maps, tau = period_maps(plate, rpm, ap, f * plate.lp, None,
                                    (float(k), 0.0), n_modes, m,
                                    coeff_scale=sign)
            rho = spectral_radius(maps, m, maps[0][0].shape[0], 40)
            rates.append(np.log(max(rho, 1e-300)) / tau)
        w = max(rates)
        hist.append((float(k), w, list(rates)))
        if verbose:
            print(f"    K_Pp={k:9.3g} -> pire taux {w:+7.1f} s^-1")
        if w < best_w:
            best_w, best = w, float(k)
    return best, best_w, hist


# ---------------------------------------------------------------------------
def optimize_pd_gains(plate, rpm, ctrl, kp_grid, kd_grid, ap_probe=0.8e-3,
                      positions=(0.0, 0.5, 1.0), n_modes=2, m=32,
                      n_period=25, verbose=False):
    """Choisit (K_Pp, K_Pd) minimisant le rayon spectral maximal le long de la
    passe, a la profondeur ap_probe, correcteur robuste inclus."""
    from closed_loop import is_stable
    best, best_rho = (0.0, 0.0), np.inf
    for kp in kp_grid:
        for kd in kd_grid:
            rho = 0.0
            for f in positions:
                _, r = is_stable(plate, rpm, ap_probe, f * plate.lp, ctrl,
                                 (kp, kd), n_modes, m, n_period=n_period)
                rho = max(rho, r)
                if rho > best_rho:
                    break
            if verbose:
                print(f"    Kpp={kp:10.3g} Kpd={kd:10.3g} -> rho={rho:.4f}")
            if rho < best_rho:
                best_rho, best = rho, (float(kp), float(kd))
    return best, float(best_rho)


# ---------------------------------------------------------------------------
class LTIController:
    """Correcteur LTI discretise (bloqueur d'ordre zero) : y [m] -> u [V]."""

    def __init__(self, ss_ctrl, dt):
        A, B, C, D = [np.atleast_2d(np.asarray(m, float)) for m in ss_ctrl]
        n = A.shape[0]
        if n == 0:
            self.Ad = np.zeros((0, 0)); self.Bd = np.zeros((0, 1))
        else:
            self.Ad, self.Bd, _, _, _ = cont2discrete((A, B, C, D), dt,
                                                      method='zoh')
        self.C, self.D = C, D
        self.x = np.zeros(n)

    def reset(self):
        self.x = np.zeros(self.x.size)

    def __call__(self, y):
        u = (float(self.C[0] @ self.x) + float(self.D[0, 0]) * y) \
            if self.x.size else float(self.D[0, 0]) * y
        if self.x.size:
            self.x = self.Ad @ self.x + self.Bd[:, 0] * y
        return u


class CombinedController:
    """u(t) = u_robuste(y(t)) + K_Pp y(t-tau) + K_Pd y'(t-tau), Eqs. (30)-(31).

    `robust` : (A, B, C, D) du correcteur mu (ou None)
    `pd`     : (K_Pp, K_Pd) (ou None)
    `n_tau`  : nombre de pas d'integration dans une periode de dent
    `u_max`  : saturation de la tension d'actionneur [V] (None = aucune)
    """

    def __init__(self, dt, n_tau, robust=None, pd=None, u_max=None,
                 deriv='exact', f_deriv=3000.0):
        self.lti = LTIController(robust, dt) if robust is not None else None
        self.Kp, self.Kd = (0.0, 0.0) if pd is None else (float(pd[0]),
                                                          float(pd[1]))
        self.dt, self.n_tau = dt, int(n_tau)
        self.u_max = u_max
        self.deriv = deriv
        # derivateur a bande limitee s/(1 + s/wd), discretise (Tustin) :
        # utilise quand la vitesse n'est pas mesuree (deriv='filtered')
        wd = 2 * np.pi * f_deriv
        a = 2.0 / (dt * wd)
        self._dnum = (2.0 / dt) / (1.0 + a), -(2.0 / dt) / (1.0 + a)
        self._dden = (a - 1.0) / (a + 1.0)
        self._dstate = 0.0
        self._ylast = 0.0
        self.buf = np.zeros(self.n_tau + 4)
        self.bufd = np.zeros(self.n_tau + 4)
        self.k_last = -1
        self.u_rob_last = 0.0
        self.u_pd_last = 0.0

    def reset(self):
        if self.lti is not None:
            self.lti.reset()
        self.buf[:] = 0.0
        self.bufd[:] = 0.0
        self._dstate = 0.0
        self._ylast = 0.0
        self.k_last = -1

    def __call__(self, y=0.0, yd=None, t=0.0, k=0):
        nb = self.buf.size
        if self.deriv == 'exact' and yd is not None:
            ydot = float(yd)
        else:                                  # derivateur a bande limitee
            ydot = (self._dnum[0] * y + self._dnum[1] * self._ylast
                    + self._dden * self._dstate)
            self._dstate, self._ylast = ydot, y
        self.buf[k % nb] = y
        self.bufd[k % nb] = ydot
        u_r = self.lti(y) if self.lti is not None else 0.0
        u_d = 0.0
        if self.Kp or self.Kd:
            kd_i = k - self.n_tau
            if kd_i > 1:
                u_d = (self.Kp * self.buf[kd_i % nb]
                       + self.Kd * self.bufd[kd_i % nb])
        self.u_rob_last, self.u_pd_last = u_r, u_d
        u = u_r + u_d
        if self.u_max is not None:
            u = float(np.clip(u, -self.u_max, self.u_max))
        return u
