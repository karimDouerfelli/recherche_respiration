import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# 1. Chargement du fichier
df = pd.read_csv('sig1.csv').dropna()

# 2. Ciblage direct du premier canal biomédical par son en-tête
signal_brut = df['ch1'].values 

# 3. Paramètres temporels (pas = 0.02s -> fs = 50 Hz)
fs = 50.0
temps = np.arange(len(signal_brut)) / fs

def appliquer_filtre(donnees, coupure_basse, coupure_haute, fs, ordre=4):
    nyquist = 0.5 * fs
    bas = coupure_basse / nyquist
    haut = coupure_haute / nyquist
    b, a = butter(ordre, [bas, haut], btype='bandpass')
    return filtfilt(b, a, donnees)

# 4. Séparation stricte par bandes
# Respiration : 0.05 Hz à 0.5 Hz
respiration = appliquer_filtre(signal_brut, 0.05, 0.5, fs)

# Coeur : 0.8 Hz à 2.5 Hz
coeur = appliquer_filtre(signal_brut, 0.8, 2.5, fs)

# 5. Affichage net avec l'axe du temps en secondes
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(temps, signal_brut, color='gray')
plt.title("Signal Brut (Canal ch1)")
plt.xlabel("Temps (s)")

plt.subplot(3, 1, 2)
plt.plot(temps, respiration, color='blue')
plt.title("Respiration Isolée (0.05 - 0.5 Hz)")
plt.xlabel("Temps (s)")

plt.subplot(3, 1, 3)
plt.plot(temps, coeur, color='red')
plt.title("Rythme Cardiaque Isolé (0.8 - 2.5 Hz)")
plt.xlabel("Temps (s)")

plt.tight_layout()
plt.show()