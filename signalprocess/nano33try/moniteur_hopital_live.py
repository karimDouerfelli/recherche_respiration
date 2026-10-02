import asyncio
from collections import deque
import csv
import struct
import time
from bleak import BleakClient, BleakScanner
import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, medfilt, savgol_filter

# Paramètres du moniteur médical
NOM_BALISE = "Nano_Respiration"
UUID_CHAR = "00002a37-0000-1000-8000-00805f9b34fb"
FENETRE_SECONDES = 15.0
FS_REEL = 39.2  # Cadence matérielle prouvée (589 éch. / 15 s)
DT = 1.0 / FS_REEL
TAILLE_MAX = int(FENETRE_SECONDES * FS_REEL)

temps_buffer = deque(maxlen=TAILLE_MAX)
rssi_buffer = deque(maxlen=TAILLE_MAX)
compteur_echantillons = 0


def filtre_passe_bande(data, lowcut, highcut, fs, order=2):
  nyq = 0.5 * fs
  b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
  return filtfilt(b, a, data)


async def main():
  global compteur_echantillons
  print(f"Recherche du capteur '{NOM_BALISE}'...")
  device = await BleakScanner.find_device_by_name(NOM_BALISE, timeout=5.0)

  if not device:
    print("Échec : Capteur introuvable. Vérifie que l'Arduino est branché.")
    return

  fichier_csv = open("respiration_live.csv", mode="w", newline="")
  writer = csv.writer(fichier_csv)
  writer.writerow(["Index", "Temps_s", "RSSI_dBm"])

  # Callback : reconstruire le temps sur la cadence matérielle régulière de l'Arduino
  def reception_temps_reel(sender, data):
    global compteur_echantillons
    rssi = struct.unpack("<i", data)[0]

    # Ignorer les valeurs aberrantes extrêmes (-105 dBm) dès la source
    if rssi < -90 or rssi >= 0:
      if len(rssi_buffer) > 0:
        rssi = rssi_buffer[-1]
      else:
        return

    t_exact = compteur_echantillons * DT
    compteur_echantillons += 1

    temps_buffer.append(t_exact)
    rssi_buffer.append(rssi)
    writer.writerow([compteur_echantillons, f"{t_exact:.4f}", rssi])

  plt.style.use("dark_background")
  plt.ion()
  fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
  fig.canvas.manager.set_window_title("Moniteur Biomédical BLE - Temps Réel")

  (ligne_brute,) = ax1.plot([], [], color="#00ffcc", linewidth=1.5)
  ax1.set_ylabel("RSSI (dBm)")
  ax1.set_title(
      "1. Signal Radio En Direct (Lissé sans embouteillage)", color="#00ffcc"
  )
  ax1.grid(True, alpha=0.2)

  (ligne_resp,) = ax2.plot([], [], color="#00aaff", linewidth=2.5)
  ax2.set_ylabel("Amplitude (dB)")
  ax2.set_title(
      "2. Onde Respiratoire Temps Réel (0.15 - 0.50 Hz)", color="#00aaff"
  )
  ax2.set_ylim(-4.0, 4.0)
  ax2.grid(True, alpha=0.2)

  (ligne_cardio,) = ax3.plot([], [], color="#ff4444", linewidth=1.5)
  ax3.set_ylabel("Amplitude (dB)")
  ax3.set_xlabel("Temps écoulé (secondes)")
  ax3.set_title(
      "3. Micro-variations / Bande Cardiaque (0.90 - 2.00 Hz)", color="#ff4444"
  )
  ax3.set_ylim(-1.5, 1.5)
  ax3.grid(True, alpha=0.2)

  plt.tight_layout()
  plt.show(block=False)

  print("Connexion au capteur et ouverture du moniteur fluide...")
  try:
    async with BleakClient(device) as client:
      await client.start_notify(UUID_CHAR, reception_temps_reel)
      print("--> MONITEUR ACTIF ! Respire amplement devant l'antenne...")

      while plt.fignum_exists(fig.number):
        if len(rssi_buffer) > 80:
          t_arr = np.array(temps_buffer)
          y_arr = np.array(rssi_buffer, dtype=float)

          # 1. Filtre médian + lissage Savitzky-Golay pour casser l'effet d'escalier 1 dB
          y_med = medfilt(y_arr, kernel_size=5)
          y_lisse = savgol_filter(y_med, window_length=15, polyorder=2)

          # 2. Extraction des ondes physiologiques sur grille temporelle régulière
          y_resp = filtre_passe_bande(y_lisse, 0.15, 0.50, FS_REEL, order=2)
          y_cardio = filtre_passe_bande(y_lisse, 0.90, 2.00, FS_REEL, order=2)

          # 3. Mise à jour des courbes
          ligne_brute.set_data(t_arr, y_lisse)
          ligne_resp.set_data(t_arr, y_resp)
          ligne_cardio.set_data(t_arr, y_cardio)

          # Défilement horizontal fluide
          t_max = t_arr[-1]
          ax3.set_xlim(
              max(0, t_max - FENETRE_SECONDES), max(FENETRE_SECONDES, t_max)
          )
          ax1.set_ylim(np.min(y_lisse) - 3, np.max(y_lisse) + 3)

          # Rendu rapide non-bloquant
          fig.canvas.draw_idle()
          fig.canvas.flush_events()
          fichier_csv.flush()

        await asyncio.sleep(0.08)

      await client.stop_notify(UUID_CHAR)

  except KeyboardInterrupt:
    print("\nArrêt demandé.")
  finally:
    fichier_csv.close()
    print("Données sauvegardées dans 'respiration_live.csv'.")


if __name__ == "__main__":
  asyncio.run(main())