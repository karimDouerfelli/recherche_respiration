import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, medfilt, find_peaks
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.colors as mcolors
import os

# --- 1. FONCTIONS DE TRAITEMENT ---
def filtre_passe_bande(data, lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, data)

def calculer_volume_signal(t, signal, fs):
    """Détecte les cycles et intègre l'aire sous la courbe pour un signal donné."""
    creux, _ = find_peaks(-signal, distance=fs * 1.5)
    K_CALIBRATION = 0.12
    volumes = []
    
    for i in range(len(creux) - 1):
        idx_debut, idx_fin = creux[i], creux[i+1]
        temps_cycle = t[idx_debut:idx_fin]
        signal_cycle = signal[idx_debut:idx_fin]
        
        signal_positif = signal_cycle - np.min(signal_cycle)
        aire = trapezoid(y=signal_positif, x=temps_cycle)
        volumes.append(aire * K_CALIBRATION)
        
    if len(volumes) > 0:
        return np.mean(volumes[-10:])
    return 0.0

# --- 2. CONFIGURATION DE LA MATRICE 3x3 ---
zones = {
    "Haut_Gauche":  {"fichier": "zone_haut_gauche.csv",  "coords": [(-2.9, 4.1), (-2.9, 5.9), (-1.1, 5.9), (-1.1, 4.1)], "poids": 0.5},
    "Haut_Centre":  {"fichier": "zone_haut_centre.csv",  "coords": [(-0.9, 4.1), (-0.9, 5.9), ( 0.9, 5.9), ( 0.9, 4.1)], "poids": 0.6},
    "Haut_Droit":   {"fichier": "zone_haut_droit.csv",   "coords": [( 1.1, 4.1), ( 1.1, 5.9), ( 2.9, 5.9), ( 2.9, 4.1)], "poids": 0.5},
    
    "Milieu_Gauche":{"fichier": "zone_milieu_gauche.csv","coords": [(-2.9, 2.1), (-2.9, 3.9), (-1.1, 3.9), (-1.1, 2.1)], "poids": 0.8},
    "Milieu_Centre":{"fichier": "zone_milieu_centre.csv","coords": [(-0.9, 2.1), (-0.9, 3.9), ( 0.9, 3.9), ( 0.9, 2.1)], "poids": 0.9},
    "Milieu_Droit": {"fichier": "zone_milieu_droit.csv", "coords": [( 1.1, 2.1), ( 1.1, 3.9), ( 2.9, 3.9), ( 2.9, 2.1)], "poids": 0.8},
    
    "Bas_Gauche":   {"fichier": "zone_bas_gauche.csv",   "coords": [(-2.9, 0.1), (-2.9, 1.9), (-1.1, 1.9), (-1.1, 0.1)], "poids": 1.1},
    "Bas_Centre":   {"fichier": "zone_bas_centre.csv",   "coords": [(-0.9, 0.1), (-0.9, 1.9), ( 0.9, 1.9), ( 0.9, 0.1)], "poids": 1.2},
    "Bas_Droit":    {"fichier": "zone_bas_droit.csv",    "coords": [( 1.1, 0.1), ( 1.1, 1.9), ( 2.9, 1.9), ( 2.9, 0.1)], "poids": 1.1},
}

# --- 3. TRAITEMENT DES 9 FICHIERS ---
resultats_volumes = {}
signaux_a_tracer = []  # Va stocker des paires (temps, signal)

for nom_zone, config in zones.items():
    fichier = config["fichier"]
    
    if os.path.exists(fichier):
        df = pd.read_csv(fichier)
        t = df["Timestamp"].values
        t = t - t[0]
        rssi_brut = df["RSSI"].values
        fs = len(t) / t[-1] if t[-1] > 0 else 39.2
    else:
        fs, t = 40.0, np.linspace(0, 30, 1200)
        amplitude = config["poids"] * 4.0
        rssi_brut = -50 + amplitude * np.sin(2 * np.pi * 0.25 * t) + np.random.normal(0, 0.5, len(t))
        
    rssi_propre = medfilt(rssi_brut, kernel_size=5)
    sig_respiration = filtre_passe_bande(rssi_propre, 0.15, 0.80, fs, order=3)
    
    volume_zone = calculer_volume_signal(t, sig_respiration, fs)
    resultats_volumes[nom_zone] = volume_zone
    
    # CORRECTION : On stocke l'axe des temps spécifique à CE signal
    signaux_a_tracer.append((t, sig_respiration))

volume_total = sum(resultats_volumes.values())

# --- 4. AFFICHAGE DU TABLEAU DE BORD ---
plt.style.use("dark_background")
fig = plt.figure(figsize=(15, 7))

# Panneau Gauche : Faisceau des 9 signaux
ax1 = plt.subplot(1, 2, 1)

# On trace les 9 signaux avec leur propre axe temporel
for t_local, sig in signaux_a_tracer:
    ax1.plot(t_local, sig, color="#00ffcc", alpha=0.4, linewidth=1.5)

# CORRECTION DU CALCUL DE LA MOYENNE :
# On trouve la taille du fichier le plus court
longueur_min = min(len(sig) for _, sig in signaux_a_tracer)
# On coupe tous les signaux pour qu'ils aient la taille du plus court
signaux_tronques = [sig[:longueur_min] for _, sig in signaux_a_tracer]
t_moyen = signaux_a_tracer[0][0][:longueur_min] # On prend l'axe temporel du 1er signal, tronqué

# On trace la moyenne des 9 capteurs
signal_moyen = np.mean(signaux_tronques, axis=0)
ax1.plot(t_moyen, signal_moyen, color="white", linewidth=2.5, label="Moyenne synchronisée")

ax1.set_title("1. Faisceau Multipoint (9 Capteurs Synchronisés)", fontsize=12)
ax1.set_xlabel("Temps (secondes)")
ax1.set_ylabel("Amplitude Filtrée")
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.2)

# Panneau Droit : Matrice 3x3
ax2 = plt.subplot(1, 2, 2)
volume_max = max(resultats_volumes.values()) if volume_total > 0 else 1.0
norm = mcolors.Normalize(vmin=0, vmax=volume_max)
cmap = plt.get_cmap("coolwarm")

for nom_zone, config in zones.items():
    volume_mesure = resultats_volumes[nom_zone]
    couleur = cmap(norm(volume_mesure))
    
    polygone = Polygon(config["coords"], closed=True, facecolor=couleur, edgecolor="black", linewidth=3, alpha=0.9)
    ax2.add_patch(polygone)
    
    x_coords, y_coords = [p[0] for p in config["coords"]], [p[1] for p in config["coords"]]
    centre_x, centre_y = sum(x_coords) / 4, sum(y_coords) / 4
    
    label_texte = nom_zone.replace("_", "\n")
    ax2.text(centre_x, centre_y + 0.3, label_texte, color="white", fontsize=8, ha="center")
    ax2.text(centre_x, centre_y - 0.2, f"{volume_mesure:.2f} L", color="white", fontsize=11, fontweight="bold",
             bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'), ha="center")

ax2.set_xlim(-3.5, 3.5)
ax2.set_ylim(-0.5, 6.5)
ax2.set_aspect("equal")
ax2.axis("off")

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax2, fraction=0.04, pad=0.04)
cbar.set_label("Volume par capteur (Litres)", color="white", fontsize=10)
cbar.ax.yaxis.set_tick_params(color="white")

ax2.set_title(f"2. Cartographie 3x3 (Volume Total Mesuré : {volume_total:.2f} L)", fontsize=12)

plt.tight_layout()
plt.show()