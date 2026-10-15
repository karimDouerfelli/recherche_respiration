import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import asyncio
from collections import deque
import csv
import struct
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, medfilt, savgol_filter
import tensorflow as tf
from bleak import BleakClient, BleakScanner

# --- Paramètres du moniteur ---
NOM_BALISE = "Nano_Respiration"
UUID_CHAR = "00002a37-0000-1000-8000-00805f9b34fb"
FS_REEL = 39.2  
DT = 1.0 / FS_REEL
TAILLE_FENETRE_IA = 600

temps_buffer = deque(maxlen=TAILLE_FENETRE_IA)
rssi_buffer = deque(maxlen=TAILLE_FENETRE_IA)
compteur_echantillons = 0

# --- 1. CHARGEMENT ET CHAUFFE DE L'IA (WARM-UP) ---
print("Chargement du cerveau IA...")
modele_ia = tf.keras.models.load_model('modele_respiratoire_v1.keras')

print("Chauffe du réseau de neurones (Warm-up)...")
fausse_donnee = np.zeros((1, TAILLE_FENETRE_IA, 1))
modele_ia.predict(fausse_donnee, verbose=0) # TensorFlow alloue la mémoire ici
print("IA prête et ultra-rapide !")

def filtre_passe_bande(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, data)

async def main():
    global compteur_echantillons
    print(f"Recherche du capteur '{NOM_BALISE}'...")
    device = await BleakScanner.find_device_by_name(NOM_BALISE, timeout=5.0)

    if not device:
        print("Échec : Capteur introuvable.")
        return

    fichier_csv = open("respiration_live_ia.csv", mode="w", newline="")
    writer = csv.writer(fichier_csv)
    writer.writerow(["Index", "Temps_s", "RSSI_dBm"])

    def reception_temps_reel(sender, data):
        global compteur_echantillons
        rssi = struct.unpack("<i", data)[0]

        if rssi < -120 or rssi >= 0:
            if len(rssi_buffer) > 0:
                rssi = rssi_buffer[-1]
            else:
                return

        t_exact = compteur_echantillons * DT
        compteur_echantillons += 1

        temps_buffer.append(t_exact)
        rssi_buffer.append(rssi)
        writer.writerow([compteur_echantillons, f"{t_exact:.4f}", rssi])

    # --- Configuration de l'Interface ---
    plt.style.use("dark_background")
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.canvas.manager.set_window_title("Moniteur IA - Temps Réel")

    (ligne_brute,) = ax1.plot([], [], color="#00ffcc", linewidth=1.5)
    ax1.set_ylabel("RSSI (dBm)")
    ax1.set_title("1. Signal Radio", color="#00ffcc")
    ax1.grid(True, alpha=0.2)

    (ligne_resp,) = ax2.plot([], [], color="#00aaff", linewidth=2.5)
    ax2.set_ylabel("Amplitude (dB)")
    ax2.set_xlabel("Temps écoulé (secondes)")
    titre_ax2 = ax2.set_title("2. Onde Respiratoire & Décision IA", color="#00aaff")
    ax2.set_ylim(-4.0, 4.0)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show(block=False)

    print("Connexion au capteur...")
    try:
        async with BleakClient(device) as client:
            await client.start_notify(UUID_CHAR, reception_temps_reel)
            print("--> MONITEUR ACTIF ! Respire amplement...")

            compteur_ia = 0
            classe_actuelle = 0

            while plt.fignum_exists(fig.number):
                if len(rssi_buffer) > 10:
                    t_arr = np.array(temps_buffer)
                    y_arr = np.array(rssi_buffer, dtype=float)

                    y_med = medfilt(y_arr, kernel_size=5)
                    y_lisse = savgol_filter(y_med, window_length=15, polyorder=2)
                    
                    # --- LE SUPER-ARBITRE IA ---
                    if len(y_arr) == TAILLE_FENETRE_IA:
                        if compteur_ia % 10 == 0:
                            fenetre_ia = y_arr.reshape(1, TAILLE_FENETRE_IA, 1)
                            probabilites = modele_ia.predict(fenetre_ia, verbose=0)
                            classe_actuelle = np.argmax(probabilites)
                        compteur_ia += 1

                    # --- TRAITEMENT CONDITIONNEL ---
                    if classe_actuelle == 0:
                        y_resp = filtre_passe_bande(y_lisse, 0.15, 0.50, FS_REEL, order=2)
                        titre_ax2.set_text("ÉTAT : RESPIRATION NORMALE")
                        titre_ax2.set_color("#00ff00")
                    elif classe_actuelle == 1:
                        y_resp = np.zeros_like(y_lisse)
                        titre_ax2.set_text("ALERTE : APNÉE DÉTECTÉE")
                        titre_ax2.set_color("#ff0000")
                    else:
                        y_resp = np.zeros_like(y_lisse)
                        titre_ax2.set_text("ÉTAT : MOUVEMENT PARASITE")
                        titre_ax2.set_color("#ffa500")

                    # --- MISE À JOUR VISUELLE ---
                    ligne_brute.set_data(t_arr, y_lisse)
                    ligne_resp.set_data(t_arr, y_resp)

                    ax1.set_xlim(t_arr[0], t_arr[-1])
                    ax1.set_ylim(np.min(y_lisse) - 3, np.max(y_lisse) + 3)
                    
                    fig.canvas.draw_idle()

                # Le battement de cœur de la fenêtre (doit tourner en permanence)
                fig.canvas.flush_events()
                await asyncio.sleep(0.05)

            await client.stop_notify(UUID_CHAR)

    except KeyboardInterrupt:
        print("\nArrêt demandé.")
    finally:
        fichier_csv.close()
        print("Données sauvegardées.")

if __name__ == "__main__":
    asyncio.run(main())