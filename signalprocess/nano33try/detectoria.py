import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy.signal import butter, filtfilt, medfilt, find_peaks
import matplotlib.pyplot as plt

# 1. Création du modèle IA
TAILLE_FENETRE = 600

def creer_filtre_ia():
    modele = models.Sequential([
        layers.Input(shape=(TAILLE_FENETRE, 1)),
        layers.Conv1D(filters=32, kernel_size=15, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(filters=64, kernel_size=10, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.GlobalAveragePooling1D(),
        layers.Dense(32, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    modele.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return modele

print("Initialisation de l'IA...")
filtre_ia = creer_filtre_ia()

# 2. Fonctions de filtrage classique
def filtre_passe_bande(data, lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs  # Opérateur de multiplication restauré
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, data)

# 3. Chargement et remise à l'échelle du temps
df = pd.read_csv("respiration_brute.csv")
t = df["Timestamp"].values
t = t - t[0]  # Opérateur de soustraction restauré
rssi_brut = df["RSSI"].values

fs = len(t) / t[-1] if t[-1] > 0 else 39.2
print(f"Fréquence d'échantillonnage mesurée : {fs:.2f} Hz")

# 4. Analyse IA sur la première fenêtre
classe_detectee = 0
etat_ia = "Données insuffisantes pour l'IA"
if len(rssi_brut) >= TAILLE_FENETRE:
    fenetre_brute = rssi_brut[:TAILLE_FENETRE]
    fenetre_ia = fenetre_brute.reshape(1, TAILLE_FENETRE, 1)
    
    probabilites = filtre_ia.predict(fenetre_ia, verbose=0)
    classe_detectee = np.argmax(probabilites)
    
    etats = ["RESPIRATION NORMALE (Filtres fiables)", "APNÉE DÉTECTÉE (Rythmes forcés à 0)", "MOUVEMENT PARASITE (Rythmes ignorés)"]
    etat_ia = etats[classe_detectee]

# 5. Suppression des artefacts radio et filtrage
rssi_propre = medfilt(rssi_brut, kernel_size=5)
sig_respiration = filtre_passe_bande(rssi_propre, 0.15, 0.80, fs, order=3)
sig_cardiaque = filtre_passe_bande(rssi_propre, 0.50, 3.00, fs, order=3)

# 6. Calcul des rythmes par détection de pics (avec opérateurs restaurés)
peaks_resp, _ = find_peaks(sig_respiration, distance=fs * 1.2)
if len(peaks_resp) > 1 and classe_detectee == 0:
    periodes_resp = np.diff(t[peaks_resp])
    rythme_resp = 60 / np.mean(periodes_resp)
else:
    rythme_resp = 0.0

peaks_card, _ = find_peaks(sig_cardiaque, distance=fs * 0.33)
if len(peaks_card) > 1 and classe_detectee == 0:
    periodes_card = np.diff(t[peaks_card])
    rythme_card = 60 / np.mean(periodes_card)
else:
    rythme_card = 0.0

# 7. Affichage de type moniteur biomédical avec diagnostic IA
fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
fig.canvas.manager.set_window_title("Analyse Combinée - Traitement du Signal & IA")

# Couleur du titre global selon le diagnostic IA
couleur_titre = "green" if classe_detectee == 0 else ("orange" if classe_detectee == 1 else "red")
fig.suptitle(f"Diagnostic IA : {etat_ia}", fontsize=14, fontweight='bold', color=couleur_titre)

# Graphe 1 : Signal Brut
axs[0].plot(t, rssi_brut, color="lightgray", label="RSSI Brut (avec spike)")
axs[0].plot(t, rssi_propre, color="black", linewidth=1.2, label="RSSI Nettoyé (Médian)")
axs[0].set_ylabel("Puissance (dBm)")
axs[0].set_title("1. Signal Radio BLE Capturé et Dépollué")
axs[0].legend(loc="lower right")
axs[0].grid(True, alpha=0.3)

# Graphe 2 : Respiration
axs[1].plot(t, sig_respiration, color="#007acc", linewidth=2)
axs[1].plot(t[peaks_resp], sig_respiration[peaks_resp], "x", color="black", markersize=8)
axs[1].set_ylabel("Amplitude (dB)")
axs[1].set_title(f"2. Courbe Respiratoire (0.15 - 0.50 Hz) | Rythme estimé : {rythme_resp:.1f} resp/min")
axs[1].grid(True, alpha=0.3)

# Graphe 3 : Cardiaque
axs[2].plot(t, sig_cardiaque, color="#d9534f", linewidth=1.2)
axs[2].plot(t[peaks_card], sig_cardiaque[peaks_card], "x", color="black", markersize=6)
axs[2].set_ylabel("Amplitude (dB)")
axs[2].set_xlabel("Temps (secondes)")
axs[2].set_title(f"3. Bande Cardiaque (0.50 - 3.00 Hz) | Rythme estimé : {rythme_card:.1f} BPM")
axs[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.subplots_adjust(top=0.90)  # Espace pour le titre principal
plt.show()