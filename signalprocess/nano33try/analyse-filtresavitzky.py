import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, savgol_filter, find_peaks

# --- 1. Chargement et normalisation ---
df = pd.read_csv("respiration_brute.csv")
t = df["Timestamp"].values
t = t - t[0]
rssi_brut = df["RSSI"].values
fs = len(t) / t[-1]
print(f"Fréquence d'échantillonnage : {fs:.2f} Hz")

# --- 2. Écrêtage agressif et Lissage Polynomial (Savitzky-Golay) ---
# L'écrêtage détruit les pertes de paquets massives (-100 dBm)
mediane_globale = np.median(rssi_brut)
ecart_type_brut = np.std(rssi_brut)
rssi_clippe = np.clip(rssi_brut, mediane_globale - 1.5*ecart_type_brut, mediane_globale + 1.5*ecart_type_brut)

# Savitzky-Golay "invente" les décimales manquantes et lisse l'effet d'escalier de la puce BLE
# window_length=15 (environ 0.4s), polyorder=3 (lissage cubique)
rssi_propre = savgol_filter(rssi_clippe, window_length=15, polyorder=3)


# --- 3. Filtres IIR ---
def filtre_passe_bande(data, lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, data)

sig_resp_brut = filtre_passe_bande(rssi_propre, 0.15, 0.80, fs, order=3)
sig_cardiaque = filtre_passe_bande(rssi_propre, 0.50, 3.00, fs, order=3)


# --- 4. NOISE GATE CALIBRÉ (Uniquement pour la Respiration) ---
fenetre_echantillons = int(4.5 * fs)
std_locale = pd.Series(sig_resp_brut).rolling(window=fenetre_echantillons, min_periods=1, center=True).std()
energie_lisse = std_locale.rolling(window=int(fs), min_periods=1, center=True).mean().values

# Seuil d'activation du Noise Gate (à ajuster si l'apnée ne s'aplatit pas)
seuil_activation = 1.3
masque_respiration = energie_lisse > seuil_activation
masque_adouci = pd.Series(masque_respiration.astype(float)).rolling(window=int(fs*1.5), min_periods=1, center=True).mean().values

# Application du masque uniquement au thorax
sig_respiration = sig_resp_brut * masque_adouci


# --- 5. Calcul des Rythmes ---

# A. Respiration (Détection de pics classique car le signal est fort)
peaks_resp, _ = find_peaks(sig_respiration, distance=fs*1.2, prominence=0.5)
if len(peaks_resp) > 1:
    rythme_resp = 60 / np.mean(np.diff(t[peaks_resp]))
    etat_resp = "NORMAL"
    couleur_resp = "white"
else:
    rythme_resp = 0.0
    etat_resp = "⚠️ APNÉE DÉTECTÉE"
    couleur_resp = "red"

# B. Cardiaque (Détection par Autocorrélation Protégée)

if rythme_resp == 0.0:
    rythme_card = 0.0
    etat_card = "⚠️ SIGNAL PERDU"
    couleur_card = "orange"
else:
    # 1. Écrêtage (Clipping) des harmoniques respiratoires
    # On calcule l'énergie normale du cœur dans les zones calmes (environ 1.5 dB sur ton graphe)
    limite_card = np.std(sig_cardiaque) * 0.8
    # On coupe tout ce qui dépasse cette limite pour détruire les ondes de choc de 6 dB
    sig_card_propre = np.clip(sig_cardiaque, -limite_card, limite_card)

    # 2. Autocorrélation sur le signal raboté
    correlation = np.correlate(sig_card_propre, sig_card_propre, mode='full')
    correlation = correlation[len(correlation)//2:] 

    min_lag = int(fs * 0.4)  # 150 BPM
    max_lag = int(fs * 1.5)  # 40 BPM
    zone_recherche = correlation[min_lag:max_lag]

    if len(zone_recherche) > 0 and np.max(zone_recherche) > 0:
        lag_max = np.argmax(zone_recherche) + min_lag
        rythme_card = 60.0 / (lag_max / fs)
        etat_card = f"{rythme_card:.1f} BPM (AutoCorr)"
        couleur_card = "white"
    else:
        rythme_card = 0.0
        etat_card = "⚠️ RECHERCHE..."
        couleur_card = "orange"


# --- 6. Affichage biomédical ---
plt.style.use("dark_background")
fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Graphe 1
axs[0].plot(t, rssi_brut, color="dimgray", alpha=0.5, label="RSSI Brut (Escalier)")
axs[0].plot(t, rssi_propre, color="white", linewidth=1.2, label="RSSI Lissé (Savitzky-Golay)")
axs[0].set_ylabel("Puissance (dBm)")
axs[0].set_title("1. Signal Radio (Correction de Quantification)")
axs[0].legend(loc="lower right")

# Graphe 2
axs[1].plot(t, sig_respiration, color="#007acc", linewidth=2)
if len(peaks_resp) > 0:
    axs[1].plot(t[peaks_resp], sig_respiration[peaks_resp], "x", color="red", markersize=8)
axs[1].set_ylabel("Amplitude (dB)")
titre_resp = axs[1].set_title(f"2. Respiration | Rythme : {rythme_resp:.1f} resp/min | {etat_resp}")
titre_resp.set_color(couleur_resp)

# Graphe 3 (Pas de croix 'x' dessinées car on utilise l'autocorrélation globale)
axs[2].plot(t, sig_cardiaque, color="#d9534f", linewidth=1.2)
axs[2].set_ylabel("Amplitude (dB)")
axs[2].set_xlabel("Temps (secondes)")
titre_card = axs[2].set_title(f"3. Cardiaque | {etat_card}")
titre_card.set_color(couleur_card)

plt.tight_layout()
plt.show()