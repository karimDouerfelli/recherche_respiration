import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

TAILLE_FENETRE = 600

# 1. Création de l'IA "Tout-en-un" (Filtre + Classifieur)
def creer_ia_complete():
    # Entrée du signal brut
    entree = layers.Input(shape=(TAILLE_FENETRE, 1), name="entree_brute")

    # --- PARTIE 1 : Extracteur de caractéristiques (L'IA "regarde" le signal) ---
    x = layers.Conv1D(filters=32, kernel_size=15, activation='relu', padding='same')(entree)
    x = layers.MaxPooling1D(pool_size=2, padding='same')(x) # Taille passe à 300
    
    x = layers.Conv1D(filters=64, kernel_size=10, activation='relu', padding='same')(x)
    encode = layers.MaxPooling1D(pool_size=2, padding='same')(x) # Taille passe à 150

    # --- PARTIE 2 (Branche A) : Classification (Type d'état) ---
    c = layers.GlobalAveragePooling1D()(encode)
    c = layers.Dense(32, activation='relu')(c)
    sortie_classe = layers.Dense(3, activation='softmax', name="sortie_classe")(c)

    # --- PARTIE 3 (Branche B) : Reconstruction (Le Filtre IA) ---
    # On "décompresse" les données pour redessiner la courbe sur 600 points
    r = layers.Conv1DTranspose(filters=64, kernel_size=10, strides=2, activation='relu', padding='same')(encode) # Remonte à 300
    r = layers.Conv1DTranspose(filters=32, kernel_size=15, strides=2, activation='relu', padding='same')(r) # Remonte à 600
    
    # Couche finale pour obtenir la courbe lisse (1 seule valeur d'amplitude par point)
    sortie_courbe = layers.Conv1D(filters=1, kernel_size=5, activation='linear', padding='same', name="sortie_courbe")(r)

    # Assemblage du modèle à double sortie
    modele = models.Model(inputs=entree, outputs=[sortie_classe, sortie_courbe])
    
    # L'IA a besoin de deux fonctions de perte : 
    # crossentropy pour apprendre les catégories, et MSE pour apprendre à tracer la courbe.
    modele.compile(
        optimizer='adam', 
        loss={'sortie_classe': 'sparse_categorical_crossentropy', 'sortie_courbe': 'mse'},
        metrics={'sortie_classe': 'accuracy'}
    )
    return modele

print("Initialisation de la nouvelle IA (Auto-encodeur + Classifieur)...")
ia_filtre = creer_ia_complete()

# 2. Chargement des données
df = pd.read_csv("respiration_brute.csv")
t = df["Timestamp"].values
t = t - t[0]
rssi_brut = df["RSSI"].values

fs = len(t) / t[-1] if t[-1] > 0 else 39.2

if len(rssi_brut) >= TAILLE_FENETRE:
    fenetre_brute = rssi_brut[:TAILLE_FENETRE]
    fenetre_ia = fenetre_brute.reshape(1, TAILLE_FENETRE, 1)
    
    # 3. L'IA fait ses deux prédictions en même temps
    probabilites, courbe_ia_3d = ia_filtre.predict(fenetre_ia, verbose=0)
    
    # Extraction de la catégorie
    classe_detectee = np.argmax(probabilites)
    etats = ["RESPIRATION NORMALE", "APNÉE DÉTECTÉE", "MOUVEMENT PARASITE"]
    etat_ia = etats[classe_detectee]
    
    # Extraction de la courbe générée par l'IA (on la remet en 1D)
    courbe_lisse_ia = courbe_ia_3d.reshape(TAILLE_FENETRE)

    # 4. Calcul du rythme sur la courbe générée par l'IA
    peaks_resp, _ = find_peaks(courbe_lisse_ia, distance=fs * 1.2)
    if len(peaks_resp) > 1 and classe_detectee == 0:
        periodes_resp = np.diff(t[:TAILLE_FENETRE][peaks_resp])
        rythme_resp = 60 / np.mean(periodes_resp)
    else:
        rythme_resp = 0.0

    # 5. Affichage
    fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.canvas.manager.set_window_title("Analyse 100% IA")

    couleur_titre = "green" if classe_detectee == 0 else ("orange" if classe_detectee == 1 else "red")
    fig.suptitle(f"Diagnostic IA : {etat_ia}", fontsize=14, fontweight='bold', color=couleur_titre)

    # Graphe 1 : Signal Brut
    axs[0].plot(t[:TAILLE_FENETRE], fenetre_brute, color="lightgray", label="RSSI Brut (Entrée de l'IA)")
    axs[0].set_ylabel("Puissance (dBm)")
    axs[0].set_title("1. Ce que l'IA regarde")
    axs[0].legend(loc="lower right")
    axs[0].grid(True, alpha=0.3)

    # Graphe 2 : Courbe dessinée par l'IA
    axs[1].plot(t[:TAILLE_FENETRE], courbe_lisse_ia, color="#007acc", linewidth=2, label="Courbe générée par l'IA")
    axs[1].plot(t[:TAILLE_FENETRE][peaks_resp], courbe_lisse_ia[peaks_resp], "x", color="black", markersize=8)
    axs[1].set_ylabel("Amplitude")
    axs[1].set_xlabel("Temps (secondes)")
    axs[1].set_title(f"2. Signal filtré EXCLUSIVEMENT par l'IA | Rythme : {rythme_resp:.1f} resp/min")
    axs[1].legend(loc="lower right")
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()
else:
    print("Pas assez de données dans le CSV pour remplir la fenêtre.")