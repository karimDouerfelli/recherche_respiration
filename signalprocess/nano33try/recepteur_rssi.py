import asyncio
import struct
import time
from bleak import BleakClient, BleakScanner
import pandas as pd

donnees_rssi = []
duree_ecoute = 30.0
nom_balise = "Nano_Respiration"
# UUID standard complet correspondant à "2A37" défini dans l'Arduino
UUID_CHARACTERISTIC = "00002a37-0000-1000-8000-00805f9b34fb"


# Fonction déclenchée automatiquement toutes les 25 ms à chaque paquet envoyé par l'Arduino
def reception_streaming(sender, data):
  # Décodage de l'entier 32-bits signé (little-endian)
  rssi = struct.unpack("<i", data)[0]
  timestamp = time.time()
  donnees_rssi.append({"Timestamp": timestamp, "RSSI": rssi})
  print(f"[{len(donnees_rssi)}] Signal temps réel : {rssi} dBm")


async def main():
  print(f"Recherche de la balise '{nom_balise}'...")
  device = await BleakScanner.find_device_by_name(nom_balise, timeout=5.0)

  if not device:
    print("Échec : Balise introuvable. L'Arduino a-t-il fini de téléverser ?")
    return

  print(f"Balise trouvée ({device.address}). Ouverture du tunnel de données...")

  try:
    async with BleakClient(device) as client:
      print("Connexion établie ! Démarrage du streaming à 40 Hz...")
      print("Respire calmement et profondément devant l'antenne...")

      # Abonnement au flux de notifications de l'Arduino
      await client.start_notify(UUID_CHARACTERISTIC, reception_streaming)

      # Écoute continue pendant 15 secondes
      await asyncio.sleep(duree_ecoute)

      # Arrêt propre du flux
      await client.stop_notify(UUID_CHARACTERISTIC)
      print("\nFin de l'enregistrement.")

  except Exception as e:
    print(f"Erreur de communication : {e}")

  if donnees_rssi:
    df = pd.DataFrame(donnees_rssi)
    df.to_csv("respiration_brute.csv", index=False)
    print(
        f"Succès total ! {len(df)} échantillons sauvegardés dans"
        " 'respiration_brute.csv'."
    )


if __name__ == "__main__":
  asyncio.run(main())