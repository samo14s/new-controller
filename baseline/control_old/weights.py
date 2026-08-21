"""
weights.py — Fonctions de ponderation de la Section 3 du papier
================================================================
Du, Liu, Dai & Long, IJMS 274 (2024) 109257.

1. Incertitude additive des modes tronques, Eqs. (18)-(19), AVEC LES VALEURS
   NUMERIQUES DU PAPIER (aucune liberte laissee) :

        W(s) = r (s^2 + 2 z1 w1 s + w1^2) / (s^2 + 2 z2 w2 s + w2^2)

   W_Paf : r = 14e-6, z1 = 0.56, z2 = 0.12, w1 = 2pi*1400, w2 = 2pi*2800
   W_Pau : r = 4e-7,  z1 = 0.58, z2 = 0.22, w1 = 2pi*1100, w2 = 2pi*3500

2. Ponderations de synthese W_Pf, W_Pu (passe-bas) et W_Pn (constante),
   Section 3.3. Le papier ne donne AUCUNE valeur numerique pour celles-ci :
   seules leurs natures sont precisees ("designed as low-pass filters",
   "designed as a constant"). Les valeurs par defaut ci-dessous sont
   identifiees pour reproduire les resultats rapportes (Figs. 14-15, 18) et
   sont declarees explicitement dans le README.

   Les passe-bas sont ecrits BIPROPRES

        W(s) = k (s/M + wc) / (s + wc)      -> gain k en DC, k/M en HF

   ce qui est necessaire pour que D12 soit de rang plein colonne (condition
   de regularite du probleme H-infini standard) tout en gardant le caractere
   passe-bas demande par le papier.
"""
import numpy as np

TWO_PI = 2.0 * np.pi

# --- Eqs. (18)-(19) : valeurs litterales du papier --------------------------
W_PAF = dict(r=14e-6, z1=0.56, z2=0.12, w1=TWO_PI * 1400.0, w2=TWO_PI * 2800.0)
W_PAU = dict(r=4e-7, z1=0.58, z2=0.22, w1=TWO_PI * 1100.0, w2=TWO_PI * 3500.0)


def second_order_weight(r, z1, w1, z2, w2):
    """Etat-espace (forme compagne observable) de
    W(s) = r (s^2 + 2 z1 w1 s + w1^2) / (s^2 + 2 z2 w2 s + w2^2).

    Retour (A, B, C, D) avec D = r (le poids est bipropre).
    """
    a1 = 2 * z2 * w2
    # W(s) = r + (b1 s + b0)/(s^2 + a1 s + w2^2)
    b1 = r * (2 * z1 * w1 - 2 * z2 * w2)
    b0 = r * (w1**2 - w2**2)
    # realisation EQUILIBREE (rotation) : etats d'amplitude comparable
    A = np.array([[0.0, w2], [-w2, -a1]])
    B = np.array([[0.0], [1.0]])
    C = np.array([[b0 / w2, b1]])
    D = np.array([[r]])
    return A, B, C, D


def W_Paf_ss():
    """Poids d'incertitude additive sur l'entree force, Eq. (18)."""
    p = W_PAF
    return second_order_weight(p['r'], p['z1'], p['w1'], p['z2'], p['w2'])


def W_Pau_ss():
    """Poids d'incertitude additive sur l'entree tension, Eq. (19)."""
    p = W_PAU
    return second_order_weight(p['r'], p['z1'], p['w1'], p['z2'], p['w2'])


def weight_mag(p, f_hz):
    """|W(j2pi f)| pour un dictionnaire de poids du second ordre."""
    s = 1j * TWO_PI * np.asarray(f_hz, float)
    num = s**2 + 2 * p['z1'] * p['w1'] * s + p['w1']**2
    den = s**2 + 2 * p['z2'] * p['w2'] * s + p['w2']**2
    return np.abs(p['r'] * num / den)


# --- Ponderations de synthese (Section 3.3) ---------------------------------
def lowpass_biproper(k, fc_hz, M=100.0):
    """Passe-bas bipropre W(s) = k (s/M + wc)/(s + wc) en etat-espace.

    gain DC = k, gain HF = k/M, coupure a fc_hz.
    """
    wc = TWO_PI * fc_hz
    # realisation a etat normalise (B = wc) : |x| ~ |entree| en basse frequence
    A = np.array([[-wc]])
    B = np.array([[wc]])
    C = np.array([[k * (1.0 / M - 1.0)]])
    D = np.array([[k / M]])
    return A, B, C, D


def constant_weight(k):
    """Poids constant (0 etat) — W_Pn du papier."""
    return (np.zeros((0, 0)), np.zeros((0, 1)),
            np.zeros((1, 0)), np.array([[float(k)]]))


# Valeurs par defaut des ponderations de synthese (identifiees, cf. README)
W_PF_DEF = dict(k=1.0e6, fc_hz=1500.0, M=250.0)   # deplacement -> performance
W_PU_DEF = dict(k=1.7e-2, fc_hz=2500.0, M=60.0)   # tension de commande
W_PN_DEF = 2.0e-8                                  # bruit de mesure (m)


def synthesis_weights(wpf=None, wpu=None, wpn=None):
    """(W_Pf, W_Pu, W_Pn) en etat-espace, avec les valeurs par defaut."""
    wpf = dict(W_PF_DEF, **(wpf or {}))
    wpu = dict(W_PU_DEF, **(wpu or {}))
    wpn = W_PN_DEF if wpn is None else wpn
    return (lowpass_biproper(**wpf), lowpass_biproper(**wpu),
            constant_weight(wpn))
