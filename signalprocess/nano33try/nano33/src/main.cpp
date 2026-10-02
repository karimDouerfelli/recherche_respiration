#include <Arduino.h>
#include <SPI.h>
#include <ArduinoBLE.h>

// Création d'un service et d'une caractéristique de streaming (Notify)
BLEService rssiService("180D");
BLEIntCharacteristic rssiChar("2A37", BLERead | BLENotify);

void setup() {
  Serial.begin(115200);

  if (!BLE.begin()) {
    while (1);
  }

  BLE.setLocalName("Nano_Respiration");
  BLE.setAdvertisedService(rssiService);
  
  // Ajout du canal de notification pour envoyer les entiers RSSI
  rssiService.addCharacteristic(rssiChar);
  BLE.addService(rssiService);
  rssiChar.writeValue(0);

  BLE.advertise();
}

void loop() {
  // Attente de la connexion du script Python
  BLEDevice central = BLE.central();

  if (central) {
    // Tant que le PC est connecté, on échantillonne à 40 Hz (toutes les 25 ms)
    while (central.connected()) {
      int rssi_actuel = BLE.rssi(); // Lecture directe du registre HCI de la puce NINA
      
      // On filtre les valeurs d'erreur transitoires (0 ou 127)
      if (rssi_actuel < 0) {
        rssiChar.writeValue(rssi_actuel); // Envoi immédiat au PC via BLENotify
      }
      
      delay(25); // Cadence stricte de 40 échantillons / seconde
    }
  }
}