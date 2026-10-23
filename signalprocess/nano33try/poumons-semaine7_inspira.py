import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
from scipy.signal import butter, filtfilt, medfilt, find_peaks
from scipy.integrate import trapezoid

# --- 1. PARAMETRES ---
FS_CIBLE = 20.0          # Hz, frequence de reechantillonnage commune
BANDE = (0.15, 0.80)     # Hz, bande respiratoire (9 a 48 resp/min)
AGREGATION = "mediane"   # "mediane" (robuste aux cycles aberrants) ou "moyenne"

# --- 2. FONCTIONS DE TRAITEMENT ---
def reechantillonner(t, x, fs):
    """Interpole le signal sur une grille temporelle reguliere."""
    t, idx = np.unique(t, return_index=True)   # enleve les timestamps dupliques
    x = x[idx]
    t_reg = np.arange(0, t[-1], 1.0 / fs)
    return t_reg, np.interp(t_reg, t, x)

def filtre_passe_bande(data, lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs  # CORRECTION : Ajout de l'opérateur *
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, data)

def aire_inspiration(t, signal, fs):
    """
    Pour chaque cycle (creux -> creux suivant), integre l'aire sous la courbe
    uniquement pendant l'inspiration (creux -> pic), base = valeur au creux.
    Retourne la mediane ou la moyenne des aires selon AGREGATION.
    """
    # CORRECTION : Ajout de l'opérateur * et conversion en int() obligatoire
    creux, _ = find_peaks(-signal, distance=int(fs * 1.5))
    aires = []

    for i in range(len(creux) - 1):
        a, b = creux[i], creux[i + 1]
        seg_t = t[a:b + 1]
        seg_s = signal[a:b + 1]

        ipic = np.argmax(seg_s)
        seg_t = seg_t[:ipic + 1]
        seg_s = seg_s[:ipic + 1]

        if len(seg_t) < 2:
            continue
            
        # CORRECTION : Ajout de l'opérateur -
        aires.append(trapezoid(seg_s - seg_s[0], seg_t))

    if not aires:
        return 0.0
    return float(np.mean(aires) if AGREGATION == "moyenne" else np.median(aires))

# --- 3. CONFIGURATION DE LA MATRICE 3x3 ---
zones = {
    "Haut_Gauche":   {"fichier": "zone_haut_gauche.csv",   "coords": [(-2.9, 4.1), (-2.9, 5.9), (-1.1, 5.9), (-1.1, 4.1)]},
    "Haut_Centre":   {"fichier": "zone_haut_centre.csv",   "coords": [(-0.9, 4.1), (-0.9, 5.9), ( 0.9, 5.9), ( 0.9, 4.1)]},
    "Haut_Droit":    {"fichier": "zone_haut_droit.csv",    "coords": [( 1.1, 4.1), ( 1.1, 5.9), ( 2.9, 5.9), ( 2.9, 4.1)]},

    "Milieu_Gauche": {"fichier": "zone_milieu_gauche.csv", "coords": [(-2.9, 2.1), (-2.9, 3.9), (-1.1, 3.9), (-1.1, 2.1)]},
    "Milieu_Centre": {"fichier": "zone_milieu_centre.csv", "coords": [(-0.9, 2.1), (-0.9, 3.9), ( 0.9, 3.9), ( 0.9, 2.1)]},
    "Milieu_Droit":  {"fichier": "zone_milieu_droit.csv",  "coords": [( 1.1, 2.1), ( 1.1, 3.9), ( 2.9, 3.9), ( 2.9, 2.1)]},

    "Bas_Gauche":    {"fichier": "zone_bas_gauche.csv",    "coords": [(-2.9, 0.1), (-2.9, 1.9), (-1.1, 1.9), (-1.1, 0.1)]},
    "Bas_Centre":    {"fichier": "zone_bas_centre.csv",    "coords": [(-0.9, 0.1), (-0.9, 1.9), ( 0.9, 1.9), ( 0.9, 0.1)]},
    "Bas_Droit":     {"fichier": "zone_bas_droit.csv",     "coords": [( 1.1, 0.1), ( 1.1, 1.9), ( 2.9, 1.9), ( 2.9, 0.1)]},
}

# --- 4. TRAITEMENT DES 9 FICHIERS ---
aires_zones = {}
signaux_a_tracer = []

for nom_zone, config in zones.items():
    fichier = config["fichier"]

    if os.path.exists(fichier):
        df = pd.read_csv(fichier)
        t = df["Timestamp"].values.astype(float)
        # CORRECTION : Ajout de l'opérateur -
        t = t - t[0]
        rssi_brut = df["RSSI"].values.astype(float)
    else:
        # Simulation (amplitude aleatoire, independante de la zone)
        t = np.linspace(0, 30, 1200)
        amplitude = np.random.uniform(1.5, 4.0)
        # CORRECTION : Ajout des opérateurs * sur une seule ligne
        rssi_brut = (-50 + amplitude * np.sin(2 * np.pi * 0.25 * t) + np.random.normal(0, 0.5, len(t)))

    # Reechantillonnage a frequence fixe, puis filtrage
    t_reg, rssi_reg = reechantillonner(t, rssi_brut, FS_CIBLE)
    rssi_propre = medfilt(rssi_reg, kernel_size=5)
    sig_respiration = filtre_passe_bande(rssi_propre, BANDE[0], BANDE[1], FS_CIBLE)

    aires_zones[nom_zone] = aire_inspiration(t_reg, sig_respiration, FS_CIBLE)
    signaux_a_tracer.append((t_reg, sig_respiration))

# Valeurs normalisees entre 0 et 1 (1 = zone la plus forte)
aire_max = max(aires_zones.values()) if max(aires_zones.values()) > 0 else 1.0
normalisees = {z: v / aire_max for z, v in aires_zones.items()}

# --- 5. AFFICHAGE ---
plt.style.use("dark_background")
fig = plt.figure(figsize=(15, 7))

# Panneau gauche : les 9 signaux + moyenne
ax1 = plt.subplot(1, 2, 1)
for t_local, sig in signaux_a_tracer:
    ax1.plot(t_local, sig, color="#00ffcc", alpha=0.4, linewidth=1.5)

longueur_min = min(len(sig) for _, sig in signaux_a_tracer)
signaux_tronques = [sig[:longueur_min] for _, sig in signaux_a_tracer]
t_moyen = signaux_a_tracer[0][0][:longueur_min]
ax1.plot(t_moyen, np.mean(signaux_tronques, axis=0), color="white",
         linewidth=2.5, label="Moyenne")

ax1.set_title("1. Signaux respiratoires filtres (9 capteurs)", fontsize=12)
ax1.set_xlabel("Temps (secondes)")
ax1.set_ylabel("Amplitude filtree (dB)")
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.2)

# Panneau droit : matrice 3x3
ax2 = plt.subplot(1, 2, 2)
norm = mcolors.Normalize(vmin=0, vmax=1)
cmap = plt.get_cmap("coolwarm")

for nom_zone, config in zones.items():
    val = normalisees[nom_zone]
    polygone = Polygon(config["coords"], closed=True, facecolor=cmap(norm(val)),
                       edgecolor="black", linewidth=3, alpha=0.9)
    ax2.add_patch(polygone)

    xs = [p[0] for p in config["coords"]]
    ys = [p[1] for p in config["coords"]]
    cx, cy = sum(xs) / 4, sum(ys) / 4

    ax2.text(cx, cy + 0.3, nom_zone.replace("_", "\n"), color="white",
             fontsize=8, ha="center")
    # CORRECTION : Ajout de l'opérateur -
    ax2.text(cx, cy - 0.2, f"{val:.2f}", color="white", fontsize=11,
             fontweight="bold", ha="center",
             bbox=dict(facecolor="black", alpha=0.5, edgecolor="none",
                       boxstyle="round,pad=0.2"))

ax2.set_xlim(-3.5, 3.5)
ax2.set_ylim(-0.5, 6.5)
ax2.set_aspect("equal")
ax2.axis("off")

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax2, fraction=0.04, pad=0.04)
cbar.set_label("Aire d'inspiration normalisee (0 a 1)", color="white", fontsize=10)
cbar.ax.yaxis.set_tick_params(color="white")

ax2.set_title("2. Cartographie 3x3 (aire d'inspiration normalisee)", fontsize=12)

plt.tight_layout()
plt.show()