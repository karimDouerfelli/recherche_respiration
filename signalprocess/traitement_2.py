import sys
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel

# 1. Traitement des données
df = pd.read_csv('sig1.csv').dropna()
signal_brut = df['ch1'].values
fs = 50.0
temps = np.arange(len(signal_brut)) / fs

def appliquer_filtre(donnees, coupure_basse, coupure_haute, fs, ordre=4):
    nyquist = 0.5 * fs
    b, a = butter(ordre, [coupure_basse / nyquist, coupure_haute / nyquist], btype='bandpass')
    return filtfilt(b, a, donnees)

respiration = appliquer_filtre(signal_brut, 0.05, 0.5, fs)
coeur = appliquer_filtre(signal_brut, 0.8, 2.5, fs)

# 2. Détection des pics et calculs
# Coeur : distance min de 0.4s (équivaut à max 150 BPM)
pics_coeur, _ = find_peaks(coeur, distance=fs*0.4)
bpm = int(len(pics_coeur) / (len(signal_brut) / fs) * 60)

# Respiration : distance min de 2.0s (équivaut à max 30 RPM)
pics_resp, _ = find_peaks(respiration, distance=fs*2.0)
rpm = int(len(pics_resp) / (len(signal_brut) / fs) * 60)

# 3. Création de l'interface graphique
app = QApplication(sys.argv)
fenetre = QMainWindow()
fenetre.setWindowTitle("Moniteur Patient Hôpital")
fenetre.resize(1000, 600)
fenetre.setStyleSheet("background-color: #1a1a1a;") # Fond gris très foncé

widget_central = QWidget()
layout_principal = QHBoxLayout()
layout_courbes = QVBoxLayout()
layout_valeurs = QVBoxLayout()

# Configuration PyQtGraph
pg.setConfigOption('background', '#1a1a1a')
pg.setConfigOption('foreground', '#ffffff')

# Courbe ECG (Verte)
plot_ecg = pg.PlotWidget(title="<span style='color: #00ff00; font-size: 14pt'>ECG</span>")
plot_ecg.plot(temps, coeur, pen=pg.mkPen(color='#00ff00', width=2.5))
layout_courbes.addWidget(plot_ecg)

# Courbe Respiration (Jaune)
plot_resp = pg.PlotWidget(title="<span style='color: #ffff00; font-size: 14pt'>Resp</span>")
plot_resp.plot(temps, respiration, pen=pg.mkPen(color='#ffff00', width=2.5))
layout_courbes.addWidget(plot_resp)

# Panneau latéral des constantes
style_label = "font-family: Arial; font-weight: bold; background-color: #2a2a2a; border-radius: 5px; padding: 15px;"

label_bpm = QLabel(f"<span style='font-size: 14pt; color: #00ff00;'>PR bpm ♥</span><br><br><span style='font-size: 50pt; color: #00ff00;'>{bpm}</span>")
label_bpm.setStyleSheet(style_label)

label_rpm = QLabel(f"<span style='font-size: 14pt; color: #ffff00;'>RR rpm</span><br><br><span style='font-size: 50pt; color: #ffff00;'>{rpm}</span>")
label_rpm.setStyleSheet(style_label)

layout_valeurs.addWidget(label_bpm)
layout_valeurs.addWidget(label_rpm)

# Assemblage de la fenêtre
layout_principal.addLayout(layout_courbes, stretch=4) # 80% de l'écran pour les courbes
layout_principal.addLayout(layout_valeurs, stretch=1) # 20% pour les chiffres
widget_central.setLayout(layout_principal)
fenetre.setCentralWidget(widget_central)

fenetre.show()
sys.exit(app.exec_())