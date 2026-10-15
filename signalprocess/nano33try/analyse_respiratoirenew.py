import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, medfilt, find_peaks

# --- 1. Chargement et remise à l'échelle du temps ---
df = pd.read_csv("respiration_brute.csv")
t = df["Timestamp"].values
t = t - t[0]  # Correction : Normalisation pour que le graphique commence à t=0
rssi_brut = df["RSSI"].values

# Calcul de la fréquence d'échantillonnage réelle (env. 39.2 Hz)
fs = len(t) / t[-1]
print(f"Fréquence d'échantillonnage mesurée : {fs:.2f} Hz")

# --- 2. Suppression des artefacts radio ---
rssi_propre = medfilt(rssi_brut, kernel_size=5)

# --- 3. Conception des filtres passe-bande (Butterworth ordre 3) ---
def filtre_passe_bande(data, lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs  # Correction : calcul de la fréquence de Nyquist
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, data)

# Extraction Respiration : 0.15 Hz à 0.80 Hz
sig_respiration = filtre_passe_bande(rssi_propre, 0.15, 0.80, fs, order=3)

# Extraction bande Cardiaque : 0.50 Hz à 3.00 Hz
sig_cardiaque = filtre_passe_bande(rssi_propre, 0.50, 3.00, fs, order=3)


# --- 4. Calcul des rythmes avec Sécurité Apnée ---

# Rythme Respiratoire (avec analyse d'énergie pour bloquer le filtre)
amplitude_resp = np.max(sig_respiration) - np.min(sig_respiration)
seuil_apnee = 0.3  # Seuil d'amplitude en dB (à ajuster selon ton capteur)

if amplitude_resp < seuil_apnee:
    rythme_resp = 0.0
    etat_patient = "⚠️ ALERTE APNÉE"
    couleur_alerte = "red"
    peaks_resp = [] # Aucun pic valide
else:
    # Correction : distance=fs*1.2
    peaks_resp, _ = find_peaks(sig_respiration, distance=fs*1.2)
    if len(peaks_resp) > 1:
        periodes_resp = np.diff(t[peaks_resp])
        rythme_resp = 60 / np.mean(periodes_resp)
        etat_patient = "NORMAL"
        couleur_alerte = "black"
    else:
        rythme_resp = 0.0
        etat_patient = "RECHERCHE..."
        couleur_alerte = "orange"

# Rythme Cardiaque (Correction : distance=fs*0.33)
peaks_card, _ = find_peaks(sig_cardiaque, distance=fs*0.33)
if len(peaks_card) > 1:
    periodes_card = np.diff(t[peaks_card])
    rythme_card = 60 / np.mean(periodes_card)
else:
    rythme_card = 0.0


# --- 5. Affichage de type moniteur biomédical ---
fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Graphe 1 : Signal Brut
axs[0].plot(t, rssi_brut, color="lightgray", label="RSSI Brut (avec spike)")
axs[0].plot(t, rssi_propre, color="black", linewidth=1.2, label="RSSI Nettoyé (Médian)")
axs[0].set_ylabel("Puissance (dBm)")
axs[0].set_title("1. Signal Radio BLE Capturé et Dépollué")
axs[0].legend(loc="lower right")
axs[0].grid(True, alpha=0.3)

# Graphe 2 : Respiration
axs[1].plot(t, sig_respiration, color="#007acc", linewidth=2)
if len(peaks_resp) > 0:
    axs[1].plot(t[peaks_resp], sig_respiration[peaks_resp], "x", color="black", markersize=8)
axs[1].set_ylabel("Amplitude (dB)")
titre_resp = axs[1].set_title(
    f"2. Courbe Respiratoire | Rythme estimé : {rythme_resp:.1f} resp/min | État : {etat_patient}"
)
titre_resp.set_color(couleur_alerte)
axs[1].grid(True, alpha=0.3)

# Graphe 3 : Cardiaque
axs[2].plot(t, sig_cardiaque, color="#d9534f", linewidth=1.2)
if len(peaks_card) > 0:
    axs[2].plot(t[peaks_card], sig_cardiaque[peaks_card], "x", color="black", markersize=6)
axs[2].set_ylabel("Amplitude (dB)")
axs[2].set_xlabel("Temps (secondes)")
axs[2].set_title(
    f"3. Bande Cardiaque (0.50 - 3.00 Hz) | Rythme estimé : {rythme_card:.1f} BPM"
)
axs[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()