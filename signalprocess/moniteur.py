import sys
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel
from PyQt5.QtCore import QTimer

# =====================================================================
# PARAMÈTRES RÉGLABLES
# =====================================================================
FICHIER = 'sig2.csv'
CANAL = 'ch1'
FS = 50.0  

# Nombre d'échantillons à supprimer au DÉBUT du signal brut.
# Les premières lignes du CSV contiennent des valeurs aberrantes
# (transitoire de démarrage du capteur / stabilisation du VNA).
# 10 échantillons = 0.2 s | 50 = 1 s | 250 = 5 s
SKIP_DEBUT = 50

# Facteur de zoom vertical : plus c'est PETIT, plus le signal est zoomé.
# 1.0 = le signal remplit exactement la fenêtre | 1.2 = 20% de marge
# < 1.0 = zoom agressif (les pics peuvent dépasser hors de la fenêtre)
ZOOM_MARGE_ECG = 0.7
ZOOM_MARGE_RESP = 0.8     # <-- respiration plus zoomée

# Percentile utilisé pour l'échelle Y : ignore les pics extrêmes
# isolés qui écraseraient visuellement le reste du signal.
# Plus la valeur est BASSE, plus le zoom est agressif.
PERCENTILE_ECG = 99.5
PERCENTILE_RESP = 85.0      # <-- respiration plus zoomée

FENETRE_SECONDES = 5      # largeur de la fenêtre glissante
VITESSE_MS = 40           # période du timer (ms)
PAS_DEFILEMENT = 1        # échantillons par rafraîchissement

# =====================================================================
# 1. Chargement, nettoyage et filtrage
# =====================================================================
df = pd.read_csv(FICHIER).dropna()
signal_complet = df[CANAL].values.astype(float)

# --- Suppression des premiers échantillons (AVANT filtrage) ---
if SKIP_DEBUT >= len(signal_complet):
    sys.exit(f"Erreur : SKIP_DEBUT ({SKIP_DEBUT}) >= taille du signal ({len(signal_complet)}).")

signal_brut = signal_complet[SKIP_DEBUT:]
print(f"[INFO] {SKIP_DEBUT} échantillons supprimés au début "
      f"({SKIP_DEBUT/FS:.2f} s) — reste {len(signal_brut)} points "
      f"({len(signal_brut)/FS:.1f} s)")

temps = np.arange(len(signal_brut)) / FS


def appliquer_filtre(donnees, coupure_basse, coupure_haute, fs, ordre=4):
    nyquist = 0.5 * fs
    b, a = butter(ordre, [coupure_basse / nyquist, coupure_haute / nyquist], btype='bandpass')
    return filtfilt(b, a, donnees)


respiration = appliquer_filtre(signal_brut, 0.05, 0.5, FS)
coeur = appliquer_filtre(signal_brut, 0.8, 2.5, FS)

# --- Marge de sécurité supplémentaire : on ignore aussi le tout début
#     des signaux FILTRÉS (effet de bord résiduel de filtfilt) ---
BORD_FILTRE = int(FS * 1.0)   # 1 seconde
if len(signal_brut) > 2 * BORD_FILTRE:
    temps = temps[BORD_FILTRE:-BORD_FILTRE]
    respiration = respiration[BORD_FILTRE:-BORD_FILTRE]
    coeur = coeur[BORD_FILTRE:-BORD_FILTRE]
    temps = temps - temps[0]   # on repart de t = 0

# =====================================================================
# 2. Calcul des fréquences
# =====================================================================
duree = len(coeur) / FS
pics_coeur, _ = find_peaks(coeur, distance=FS * 0.4)
bpm = int(len(pics_coeur) / duree * 60)
pics_resp, _ = find_peaks(respiration, distance=FS * 2.0)
rpm = int(len(pics_resp) / duree * 60)
print(f"[INFO] PR = {bpm} bpm | RR = {rpm} rpm")


def calculer_echelle(donnees, percentile, marge):
    """Calcule une échelle Y centrée et adaptée à l'amplitude réelle du signal,
    en ignorant les valeurs extrêmes isolées (via le percentile)."""
    centre = np.median(donnees)
    amplitude = np.percentile(np.abs(donnees - centre), percentile)
    if amplitude <= 0:
        amplitude = np.std(donnees) if np.std(donnees) > 0 else 1.0
    demi_hauteur = amplitude * marge
    return centre - demi_hauteur, centre + demi_hauteur


y_min_ecg, y_max_ecg = calculer_echelle(coeur, PERCENTILE_ECG, ZOOM_MARGE_ECG)
y_min_resp, y_max_resp = calculer_echelle(respiration, PERCENTILE_RESP, ZOOM_MARGE_RESP)
print(f"[INFO] Échelle ECG  : [{y_min_ecg:.4f}, {y_max_ecg:.4f}]")
print(f"[INFO] Échelle Resp : [{y_min_resp:.4f}, {y_max_resp:.4f}]")

# =====================================================================
# 3. Interface graphique
# =====================================================================
app = QApplication(sys.argv)
fenetre = QMainWindow()
fenetre.setWindowTitle("Moniteur Patient - Temps Réel")
fenetre.resize(1000, 600)
fenetre.setStyleSheet("background-color: #1a1a1a;")

widget_central = QWidget()
layout_principal = QHBoxLayout()
layout_courbes = QVBoxLayout()
layout_valeurs = QVBoxLayout()

pg.setConfigOption('background', '#1a1a1a')
pg.setConfigOption('foreground', '#ffffff')

# --- Courbe ECG (zoom automatique) ---
plot_ecg = pg.PlotWidget(title="<span style='color: #00ff00; font-size: 14pt'>ECG</span>")
courbe_ecg = plot_ecg.plot(pen=pg.mkPen(color='#00ff00', width=2.5))
plot_ecg.setYRange(y_min_ecg, y_max_ecg)
plot_ecg.showGrid(x=True, y=True, alpha=0.15)
layout_courbes.addWidget(plot_ecg)

# --- Courbe Respiration (zoom automatique) ---
plot_resp = pg.PlotWidget(title="<span style='color: #ffff00; font-size: 14pt'>Resp</span>")
courbe_resp = plot_resp.plot(pen=pg.mkPen(color='#ffff00', width=2.5))
plot_resp.setYRange(y_min_resp, y_max_resp)
plot_resp.showGrid(x=True, y=True, alpha=0.15)
layout_courbes.addWidget(plot_resp)

style_label = ("font-family: Arial; font-weight: bold; background-color: #2a2a2a; "
               "border-radius: 5px; padding: 15px;")
label_bpm = QLabel(f"<span style='font-size: 14pt; color: #00ff00;'>PR bpm &#9829;</span>"
                   f"<br><br><span style='font-size: 50pt; color: #00ff00;'>{bpm}</span>")
label_bpm.setStyleSheet(style_label)
label_rpm = QLabel(f"<span style='font-size: 14pt; color: #ffff00;'>RR rpm</span>"
                   f"<br><br><span style='font-size: 50pt; color: #ffff00;'>{rpm}</span>")
label_rpm.setStyleSheet(style_label)

layout_valeurs.addWidget(label_bpm)
layout_valeurs.addWidget(label_rpm)

layout_principal.addLayout(layout_courbes, stretch=4)
layout_principal.addLayout(layout_valeurs, stretch=1)
widget_central.setLayout(layout_principal)
fenetre.setCentralWidget(widget_central)

# =====================================================================
# 4. Moteur d'animation (QTimer)
# =====================================================================
fenetre_affichage = int(FS * FENETRE_SECONDES)
ptr = 0


def update():
    global ptr
    if ptr + fenetre_affichage < len(temps):
        x = temps[ptr: ptr + fenetre_affichage]
        y_ecg = coeur[ptr: ptr + fenetre_affichage]
        y_resp = respiration[ptr: ptr + fenetre_affichage]

        courbe_ecg.setData(x, y_ecg)
        courbe_resp.setData(x, y_resp)

        ptr += PAS_DEFILEMENT
    else:
        ptr = 0   # rebouclage au début


timer = QTimer()
timer.timeout.connect(update)
timer.start(VITESSE_MS)

fenetre.show()
sys.exit(app.exec_())