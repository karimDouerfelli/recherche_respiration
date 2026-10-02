import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, medfilt, welch

# 1. Chargement et remise à l'échelle du temps
df = pd.read_csv("respiration_brute.csv")
t = df["Timestamp"].values - df["Timestamp"].values[0]
rssi_brut = df["RSSI"].values

# Calcul de la fréquence d'échantillonnage réelle (env. 39.2 Hz)
fs = len(t) / t[-1]
print(f"Fréquence d'échantillonnage mesurée : {fs:.2f} Hz")

# 2. Suppression des artefacts radio (comme le -105 dBm à l'éch. 116)
rssi_propre = medfilt(rssi_brut, kernel_size=5)


# 3. Conception des filtres passe-bande (Butterworth ordre 3)
def filtre_passe_bande(data, lowcut, highcut, fs, order=3):
  nyq = 0.5 * fs
  b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
  return filtfilt(b, a, data)


# Extraction Respiration : 0.15 Hz (9 resp/min) à 0.5 Hz (30 resp/min)
sig_respiration = filtre_passe_bande(rssi_propre, 0.15, 0.50, fs, order=3)

# Extraction bande Cardiaque / Micro-mouvements : 0.9 Hz (54 BPM) à 2.0 Hz (120 BPM)
sig_cardiaque = filtre_passe_bande(rssi_propre, 0.90, 2.00, fs, order=3)

# 4. Affichage de type moniteur biomédical
fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

axs[0].plot(t, rssi_brut, color="lightgray", label="RSSI Brut (avec spike)")
axs[0].plot(
    t, rssi_propre, color="black", linewidth=1.2, label="RSSI Nettoyé (Médian)"
)
axs[0].set_ylabel("Puissance (dBm)")
axs[0].set_title("1. Signal Radio BLE Capturé et Dépollué")
axs[0].legend(loc="lower right")
axs[0].grid(True, alpha=0.3)

axs[1].plot(t, sig_respiration, color="#007acc", linewidth=2)
axs[1].set_ylabel("Amplitude (dB)")
axs[1].set_title(
    "2. Courbe Respiratoire Extraite (Filtre Passe-Bande 0.15 - 0.50 Hz)"
)
axs[1].grid(True, alpha=0.3)

axs[2].plot(t, sig_cardiaque, color="#d9534f", linewidth=1.2)
axs[2].set_ylabel("Amplitude (dB)")
axs[2].set_xlabel("Temps (secondes)")
axs[2].set_title(
    "3. Bande Cardiaque & Bruit de Quantification (0.90 - 2.00 Hz)"
)
axs[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()