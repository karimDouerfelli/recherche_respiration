import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

TAILLE_FENETRE = 600

# 1. Ton architecture (identique)
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

# 2. Fonction pour découper un long CSV en petits blocs de 600 points
def preparer_donnees(nom_fichier, etiquette_classe):
    df = pd.read_csv(nom_fichier)
    rssi_brut = df["RSSI"].values
    
    # Combien de fenêtres complètes de 15s on peut extraire ?
    nb_fenetres = len(rssi_brut) // TAILLE_FENETRE
    
    X = []
    y = []
    
    for i in range(nb_fenetres):
        debut = i * TAILLE_FENETRE
        fin = debut + TAILLE_FENETRE
        fenetre = rssi_brut[debut:fin]
        X.append(fenetre)
        y.append(etiquette_classe)
        
    # Conversion en format TensorFlow (Matrice 3D)
    X = np.array(X).reshape(-1, TAILLE_FENETRE, 1)
    y = np.array(y)
    
    return X, y

# 3. Chargement et fusion des 3 datasets
print("Chargement des données...")
X0, y0 = preparer_donnees(r"C:\Users\kaoue67\recherche_respiration\data-sem6\entrainementia\dataset_0_normal.csv", 0)
X1, y1 = preparer_donnees(r"C:\Users\kaoue67\recherche_respiration\data-sem6\entrainementia\dataset_1_apnee.csv", 1)
X2, y2 = preparer_donnees(r"C:\Users\kaoue67\recherche_respiration\data-sem6\entrainementia\dataset_2_bruit.csv", 2)

X_train = np.concatenate([X0, X1, X2])
y_train = np.concatenate([y0, y1, y2])

print(f"Total de fenêtres pour l'entraînement : {len(X_train)}")

# 4. Entraînement de l'IA (C'est ici que la magie opère)
mon_ia = creer_filtre_ia()

print("Début de l'apprentissage...")
mon_ia.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.2)

# 5. Sauvegarde du cerveau
mon_ia.save('modele_respiratoire_v1.keras')
print("Modèle sauvegardé sous 'modele_respiratoire_v1.keras' !")