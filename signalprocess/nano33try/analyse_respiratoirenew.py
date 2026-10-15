import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, medfilt, find_peaks

# --- 1. Chargement et normalisation ---
df = pd.read_csv("respiration_brute.csv")
t = df["Timestamp"].values
t = t - t[0]
rssi_brut = df["RSSI"].values
fs = len(t) / t[-1]
print(f"Fréquence d'échantillonnage : {fs:.2f} Hz")

# --- 2. Écrêtage agressif et lissage (Protection contre les pertes BLE) ---
mediane_globale = np.median(rssi_brut)
ecart_type_brut = np.std(rssi_brut)
rssi_clippe = np.clip(rssi_brut, mediane_globale - 1.5*ecart_type_brut, mediane_globale + 1.5*ecart_type_brut)
rssi_propre = medfilt(rssi_clippe, kernel_size=5)

# --- 3. Filtres IIR ---
def filtre_passe_bande(data, lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, data)

sig_resp_brut = filtre_passe_bande(rssi_propre, 0.15, 0.80, fs, order=3)
sig_cardiaque = filtre_passe_bande(rssi_propre, 0.50, 3.00, fs, order=3) # Libéré du masque !

# --- 4. NOISE GATE CALIBRÉ (Uniquement pour la Respiration) ---
# Fenêtre de 4.5 secondes pour donner de l'inertie
fenetre_echantillons = int(4.5 * fs)

# Calcul de l'énergie (écart-type) lissée
std_locale = pd.Series(sig_resp_brut).rolling(window=fenetre_echantillons, min_periods=1, center=True).std()
energie_lisse = std_locale.rolling(window=int(fs), min_periods=1, center=True).mean().values

# CRITIQUE : Le seuil doit être supérieur à l'énergie du bruit d'apnée.
# Si la courbe ne devient pas plate pendant l'apnée, monte cette valeur (ex: 1.8, 2.0).
# Si elle coupe tes vraies respirations, baisse-la légèrement (ex: 1.2).
seuil_activation = 1.5 
masque_respiration = energie_lisse > seuil_activation

# Lissage du masque étendu à 1.5 seconde pour un "fondu" très doux (fade in/out)
masque_adouci = pd.Series(masque_respiration.astype(float)).rolling(window=int(fs*1.5), min_periods=1, center=True).mean().values

# Application du masque uniquement au thorax
sig_respiration = sig_resp_brut * masque_adouci

# --- 5. Détection des pics ---
# Respiration
peaks_resp, _ = find_peaks(sig_respiration, distance=fs*1.2, prominence=0.5)
if len(peaks_resp) > 1:
    rythme_resp = 60 / np.mean(np.diff(t[peaks_resp]))
    etat_resp = "NORMAL"
    couleur_resp = "white"
else:
    rythme_resp = 0.0
    etat_resp = "⚠️ APNÉE DÉTECTÉE"
    couleur_resp = "red"

# Cardiaque (Détection continue avec prominence ajustée pour le bruit de fond)
peaks_card, _ = find_peaks(sig_cardiaque, distance=fs*0.4, prominence=1.5)
if len(peaks_card) > 1:
    rythme_card = 60 / np.mean(np.diff(t[peaks_card]))
    etat_card = f"{rythme_card:.1f} BPM"
    couleur_card = "white"
else:
    rythme_card = 0.0
    etat_card = "⚠️ SIGNAL PERDU"
    couleur_card = "orange"

# --- 6. Affichage biomédical ---
plt.style.use("dark_background")
fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Graphe 1
axs[0].plot(t, rssi_brut, color="dimgray", alpha=0.5, label="RSSI Brut")
axs[0].plot(t, rssi_propre, color="white", linewidth=1.2, label="RSSI Filtré")
axs[0].set_ylabel("Puissance (dBm)")
axs[0].set_title("1. Signal Radio (Temps Réel)")
axs[0].legend(loc="lower right")

# Graphe 2
axs[1].plot(t, sig_respiration, color="#007acc", linewidth=2)
if len(peaks_resp) > 0:
    axs[1].plot(t[peaks_resp], sig_respiration[peaks_resp], "x", color="red", markersize=8)
axs[1].set_ylabel("Amplitude (dB)")
titre_resp = axs[1].set_title(f"2. Respiration | Rythme : {rythme_resp:.1f} resp/min | {etat_resp}")
titre_resp.set_color(couleur_resp)

# Graphe 3
axs[2].plot(t, sig_cardiaque, color="#d9534f", linewidth=1.2)
if len(peaks_card) > 0:
    axs[2].plot(t[peaks_card], sig_cardiaque[peaks_card], "x", color="yellow", markersize=6)
axs[2].set_ylabel("Amplitude (dB)")
axs[2].set_xlabel("Temps (secondes)")
titre_card = axs[2].set_title(f"3. Cardiaque | {etat_card}")
titre_card.set_color(couleur_card)

plt.tight_layout()
plt.show()