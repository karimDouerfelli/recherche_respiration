import os
import sys
import time
import struct
import asyncio
import threading
from collections import deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
from matplotlib.widgets import Button
from scipy.signal import butter, filtfilt, medfilt, find_peaks
from scipy.integrate import trapezoid

try:
    from bleak import BleakClient, BleakScanner
    BLEAK_DISPONIBLE = True
except ImportError:
    BLEAK_DISPONIBLE = False

# ==============================================================================
# 1. PARAMETRES DU SYSTEME ET DU CAPTEUR
# ==============================================================================
NOM_BALISE = "Nano_Respiration"
UUID_CHAR = "00002a37-0000-1000-8000-00805f9b34fb"

# Cadence matérielle de l'Arduino Nano 33 BLE
FS_REEL = 39.2              # Hz (~40 Hz)
DT = 1.0 / FS_REEL

# Paramètres de filtrage respiratoire identiques au code original
FS_CIBLE = 20.0             # Hz, fréquence commune de rééchantillonnage
BANDE_RESP = (0.15, 0.80)   # Hz, bande respiratoire (9 à 48 resp/min)
AGREGATION = "mediane"      # "mediane" ou "moyenne"

DUREE_ENREGISTREMENT = 25.0 # secondes par zone
TAILLE_BUFFER_LIVE = int(FS_REEL * 35.0)  # ~35 secondes en mémoire

REP_BASE = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# 2. DEFINITION DES 15 ZONES (MATRICE 5x3 : L1 en haut, L5 en bas)
# ==============================================================================
ZONES = {
    # Ligne 1 (Très Haut) - Y: 8.1 à 9.9
    "L1_Gauche": {"fichier": "zone_l1_gauche.csv", "coords": [(-2.9, 8.1), (-2.9, 9.9), (-1.1, 9.9), (-1.1, 8.1)]},
    "L1_Centre": {"fichier": "zone_l1_centre.csv", "coords": [(-0.9, 8.1), (-0.9, 9.9), ( 0.9, 9.9), ( 0.9, 8.1)]},
    "L1_Droit":  {"fichier": "zone_l1_droit.csv",  "coords": [( 1.1, 8.1), ( 1.1, 9.9), ( 2.9, 9.9), ( 2.9, 8.1)]},

    # Ligne 2 (Haut) - Y: 6.1 à 7.9
    "L2_Gauche": {"fichier": "zone_l2_gauche.csv", "coords": [(-2.9, 6.1), (-2.9, 7.9), (-1.1, 7.9), (-1.1, 6.1)]},
    "L2_Centre": {"fichier": "zone_l2_centre.csv", "coords": [(-0.9, 6.1), (-0.9, 7.9), ( 0.9, 7.9), ( 0.9, 6.1)]},
    "L2_Droit":  {"fichier": "zone_l2_droit.csv",  "coords": [( 1.1, 6.1), ( 1.1, 7.9), ( 2.9, 7.9), ( 2.9, 6.1)]},

    # Ligne 3 (Milieu) - Y: 4.1 à 5.9
    "L3_Gauche": {"fichier": "zone_l3_gauche.csv", "coords": [(-2.9, 4.1), (-2.9, 5.9), (-1.1, 5.9), (-1.1, 4.1)]},
    "L3_Centre": {"fichier": "zone_l3_centre.csv", "coords": [(-0.9, 4.1), (-0.9, 5.9), ( 0.9, 5.9), ( 0.9, 4.1)]},
    "L3_Droit":  {"fichier": "zone_l3_droit.csv",  "coords": [( 1.1, 4.1), ( 1.1, 5.9), ( 2.9, 5.9), ( 2.9, 4.1)]},

    # Ligne 4 (Bas) - Y: 2.1 à 3.9
    "L4_Gauche": {"fichier": "zone_l4_gauche.csv", "coords": [(-2.9, 2.1), (-2.9, 3.9), (-1.1, 3.9), (-1.1, 2.1)]},
    "L4_Centre": {"fichier": "zone_l4_centre.csv", "coords": [(-0.9, 2.1), (-0.9, 3.9), ( 0.9, 3.9), ( 0.9, 2.1)]},
    "L4_Droit":  {"fichier": "zone_l4_droit.csv",  "coords": [( 1.1, 2.1), ( 1.1, 3.9), ( 2.9, 3.9), ( 2.9, 2.1)]},

    # Ligne 5 (Très Bas) - Y: 0.1 à 1.9
    "L5_Gauche": {"fichier": "zone_l5_gauche.csv", "coords": [(-2.9, 0.1), (-2.9, 1.9), (-1.1, 1.9), (-1.1, 0.1)]},
    "L5_Centre": {"fichier": "zone_l5_centre.csv", "coords": [(-0.9, 0.1), (-0.9, 1.9), ( 0.9, 1.9), ( 0.9, 0.1)]},
    "L5_Droit":  {"fichier": "zone_l5_droit.csv",  "coords": [( 1.1, 0.1), ( 1.1, 1.9), ( 2.9, 1.9), ( 2.9, 0.1)]},
}
NOMS_ZONES = list(ZONES.keys())

# ==============================================================================
# 3. FONCTIONS DE TRAITEMENT IDENTIQUES AU CODE ORIGINAL
# ==============================================================================
def reechantillonner(t, x, fs=FS_CIBLE):
    """Interpole le signal sur une grille temporelle régulière à fs Hz."""
    t_uniq, idx = np.unique(t, return_index=True)
    x_uniq = x[idx]
    if len(t_uniq) < 2 or t_uniq[-1] <= t_uniq[0]:
        return t, x
    t_reg = np.arange(0, t_uniq[-1] - t_uniq[0], 1.0 / fs)
    return t_reg, np.interp(t_reg, t_uniq - t_uniq[0], x_uniq)

def filtre_passe_bande(data, lowcut, highcut, fs=FS_CIBLE, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, data)

def calculer_aire_inspiration(t, signal, fs=FS_CIBLE):
    """
    Intègre l'aire sous la courbe uniquement pendant l'inspiration (creux -> pic).
    Formule exacte de poumons-5pos.py.
    """
    if len(signal) < int(fs * 3.5):
        return 0.0, [], []

    creux, _ = find_peaks(-signal, distance=int(fs * 1.5))
    aires = []
    pics_indices = []

    for i in range(len(creux) - 1):
        a, b = creux[i], creux[i + 1]
        seg_t = t[a:b + 1]
        seg_s = signal[a:b + 1]

        ipic = np.argmax(seg_s)
        pics_indices.append(a + ipic)

        seg_t_inspi = seg_t[:ipic + 1]
        seg_s_inspi = seg_s[:ipic + 1]

        if len(seg_t_inspi) >= 2:
            aire = trapezoid(seg_s_inspi - seg_s_inspi[0], seg_t_inspi)
            aires.append(aire)

    if not aires:
        return 0.0, creux, pics_indices

    aire_finale = float(np.median(aires) if AGREGATION == "mediane" else np.mean(aires))
    return max(0.0, aire_finale), creux, pics_indices

# ==============================================================================
# 4. GESTIONNAIRE DE FLUX TEMPS REEL (BLE & SIMULATION)
# ==============================================================================
class GestionnaireFlux:
    def __init__(self):
        self.mode_simu = False
        self.ble_connecte = False
        self.etat_connexion = "Initialisation..."
        self.temps_flux = deque(maxlen=TAILLE_BUFFER_LIVE)
        self.rssi_flux = deque(maxlen=TAILLE_BUFFER_LIVE)
        self.verrou = threading.Lock()
        self.compteur = 0
        self.en_cours_execution = True

        self.en_enregistrement = False
        self.debut_enregistrement = 0.0
        self.temps_rec_zone = []
        self.rssi_rec_zone = []

        self.thread = None

    def demarrer(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self._boucle_acquisition, daemon=True)
            self.thread.start()

    def _boucle_acquisition(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._gestionnaire_ble_ou_simu())

    async def _gestionnaire_ble_ou_simu(self):
        while self.en_cours_execution:
            if not self.mode_simu and BLEAK_DISPONIBLE:
                self.etat_connexion = f"Recherche de '{NOM_BALISE}'..."
                try:
                    device = await BleakScanner.find_device_by_name(NOM_BALISE, timeout=3.5)
                    if device:
                        self.etat_connexion = f"Connexion à {device.name}..."
                        async with BleakClient(device) as client:
                            self.ble_connecte = True
                            self.etat_connexion = f"Connecté BLE : {NOM_BALISE}"

                            def callback_ble(sender, data):
                                val = struct.unpack("<i", data)[0]
                                # Validation RSSI réelle (-120 à 0 dBm)
                                if val < -120 or val >= 0:
                                    with self.verrou:
                                        if len(self.rssi_flux) > 0:
                                            val = self.rssi_flux[-1]
                                        else:
                                            return
                                self._ajouter_echantillon(val)

                            await client.start_notify(UUID_CHAR, callback_ble)
                            while client.is_connected and self.en_cours_execution and not self.mode_simu:
                                await asyncio.sleep(0.1)

                            await client.stop_notify(UUID_CHAR)
                    else:
                        self.etat_connexion = "Capteur introuvable -> Mode Simulation (40 Hz)"
                        self.mode_simu = True
                except Exception as e:
                    self.etat_connexion = f"BLE déconnecté ({e}) -> Mode Simulation"
                    self.mode_simu = True
                finally:
                    self.ble_connecte = False
            else:
                # Mode simulation réaliste : centré sur un RSSI réel de -60 dBm avec amplitude respiratoire
                self.etat_connexion = "Mode Simulation actif (40 Hz)"
                while self.en_cours_execution and self.mode_simu:
                    t_actuel = self.compteur * DT
                    onde_resp = 2.2 * np.sin(2 * np.pi * 0.25 * t_actuel)
                    onde_cardio = 0.25 * np.sin(2 * np.pi * 1.2 * t_actuel)
                    bruit = np.random.normal(0, 0.4)
                    val_rssi = -60.0 + onde_resp + onde_cardio + bruit
                    self._ajouter_echantillon(val_rssi)
                    await asyncio.sleep(DT)

            await asyncio.sleep(0.5)

    def _ajouter_echantillon(self, rssi):
        with self.verrou:
            t = self.compteur * DT
            self.compteur += 1
            self.temps_flux.append(t)
            self.rssi_flux.append(rssi)

            if self.en_enregistrement:
                self.temps_rec_zone.append(t)
                self.rssi_rec_zone.append(rssi)

    def demarrer_enregistrement(self):
        with self.verrou:
            self.temps_rec_zone = []
            self.rssi_rec_zone = []
            self.debut_enregistrement = time.time()
            self.en_enregistrement = True

    def arreter_enregistrement(self):
        with self.verrou:
            self.en_enregistrement = False
            return np.array(self.temps_rec_zone), np.array(self.rssi_rec_zone)

    def obtenir_donnees_recentes(self):
        with self.verrou:
            t = np.array(self.temps_flux)
            y = np.array(self.rssi_flux)
            en_rec = self.en_enregistrement
            t_debut = self.debut_enregistrement
            n_pts_rec = len(self.temps_rec_zone)
        return t, y, en_rec, t_debut, n_pts_rec

# ==============================================================================
# 5. APPLICATION PRINCIPALE
# ==============================================================================
def main():
    flux = GestionnaireFlux()

    # Charger les zones existantes ou générer la simulation (identique au code original)
    aires_zones = {}
    signaux_enregistres = {}

    for nom, cfg in ZONES.items():
        chemin_csv = os.path.join(REP_BASE, cfg["fichier"])
        if os.path.exists(chemin_csv):
            try:
                df = pd.read_csv(chemin_csv)
                t_f = df["Timestamp"].values.astype(float)
                t_f = t_f - t_f[0]
                r_f = df["RSSI"].values.astype(float)
            except Exception:
                t_f, r_f = None, None
        else:
            t_f, r_f = None, None

        if t_f is None or len(t_f) < 20:
            # Simulation de secours pour les zones non encore mesurées
            np.random.seed(abs(hash(nom)) % 1000)
            t_f = np.linspace(0, 30, 1200)
            amp = np.random.uniform(1.5, 3.5)
            r_f = -55.0 + amp * np.sin(2 * np.pi * 0.25 * t_f) + np.random.normal(0, 0.5, len(t_f))

        t_reg, r_reg = reechantillonner(t_f, r_f, FS_CIBLE)
        r_propre = medfilt(r_reg, kernel_size=5)
        sig_f = filtre_passe_bande(r_propre, BANDE_RESP[0], BANDE_RESP[1], FS_CIBLE)
        aire, _, _ = calculer_aire_inspiration(t_reg, sig_f, FS_CIBLE)

        aires_zones[nom] = aire
        signaux_enregistres[nom] = (t_reg, sig_f)

    # Index de la zone sélectionnée
    idx_zone = 0
    zone_active = NOMS_ZONES[idx_zone]

    # Configuration de l'affichage Matplotlib
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 9))
    fig.canvas.manager.set_window_title("Poumons 5 Positions - Moniteur Temps Réel")

    plt.subplots_adjust(bottom=0.14, top=0.92, left=0.07, right=0.95, wspace=0.28, hspace=0.32)

    # 1. Panneau gauche haut : Signal Radio Brut RSSI Réel (dBm)
    ax_rssi = plt.subplot(3, 2, 1)
    ax_rssi.set_title("1. Signal Radio Brut en Direct (RSSI réel)", fontsize=11, color="#55ffff")
    ax_rssi.set_ylabel("RSSI (dBm)", fontsize=9)
    ax_rssi.grid(True, alpha=0.2)
    (ligne_rssi,) = ax_rssi.plot([], [], color="#55ffff", linewidth=1.5, label="RSSI direct")
    txt_rssi_instantane = ax_rssi.text(0.02, 0.85, "RSSI: -- dBm", transform=ax_rssi.transAxes,
                                       color="#55ffff", fontsize=10, fontweight="bold")

    # 2. Panneau gauche milieu : Onde Respiratoire Filtrée (0.15 - 0.80 Hz)
    ax_resp = plt.subplot(3, 2, 3, sharex=ax_rssi)
    ax_resp.set_title("2. Onde Respiratoire Filtrée & Détection Creux / Pics", fontsize=11, color="#00ffcc")
    ax_resp.set_ylabel("Amplitude (dB)", fontsize=9)
    ax_resp.grid(True, alpha=0.2)
    (ligne_resp,) = ax_resp.plot([], [], color="#00ffcc", linewidth=2.2, label="Onde respiratoire")
    (pts_creux,) = ax_resp.plot([], [], "rv", markersize=6, label="Creux")
    (pts_pics,) = ax_resp.plot([], [], "g^", markersize=6, label="Pics (Inspi)")
    ax_resp.legend(loc="upper right", fontsize=8)

    # 3. Panneau gauche bas : Faisceau des 15 zones (signaux enregistrés + moyenne)
    ax_faisceau = plt.subplot(3, 2, 5)
    ax_faisceau.set_title("3. Faisceau des 15 zones & Moyenne synchronisée", fontsize=11, color="#ffffff")
    ax_faisceau.set_xlabel("Temps (secondes)", fontsize=9)
    ax_faisceau.set_ylabel("Amplitude filtrée (dB)", fontsize=9)
    ax_faisceau.grid(True, alpha=0.2)

    def mettre_a_jour_faisceau():
        ax_faisceau.cla()
        ax_faisceau.set_title("3. Faisceau des 15 zones & Moyenne synchronisée", fontsize=11, color="#ffffff")
        ax_faisceau.set_xlabel("Temps (secondes)", fontsize=9)
        ax_faisceau.set_ylabel("Amplitude filtrée (dB)", fontsize=9)
        ax_faisceau.grid(True, alpha=0.2)

        sigs = list(signaux_enregistres.values())
        if sigs:
            for t_loc, s_loc in sigs:
                ax_faisceau.plot(t_loc, s_loc, color="#00ffcc", alpha=0.3, linewidth=1.1)

            longueur_min = min(len(s) for _, s in sigs)
            if longueur_min > 10:
                tronques = [s[:longueur_min] for _, s in sigs]
                t_m = sigs[0][0][:longueur_min]
                ax_faisceau.plot(t_m, np.mean(tronques, axis=0), color="white", linewidth=2.2, label="Moyenne")
                ax_faisceau.legend(loc="upper right", fontsize=8)

    mettre_a_jour_faisceau()

    # 4. Panneau droit : Cartographie 5x3 des Poumons
    ax_map = plt.subplot(1, 2, 2)
    ax_map.set_title("4. Cartographie 5x3 (Aire d'inspiration normalisée)", fontsize=12, color="#ffffff")
    ax_map.set_xlim(-3.6, 3.6)
    ax_map.set_ylim(-0.5, 10.5)
    ax_map.set_aspect("equal")
    ax_map.axis("off")

    cmap = plt.get_cmap("coolwarm")
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    patches_polygones = {}
    textes_zones = {}
    textes_valeurs = {}

    for nom, cfg in ZONES.items():
        coords = cfg["coords"]
        poly = Polygon(coords, closed=True, facecolor="#222222", edgecolor="#444444", linewidth=2.0, alpha=0.9)
        ax_map.add_patch(poly)
        patches_polygones[nom] = poly

        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        cx, cy = sum(xs) / 4.0, sum(ys) / 4.0

        t_nom = ax_map.text(cx, cy + 0.35, nom.replace("_", "\n"), color="white", fontsize=8, ha="center", va="center")
        t_val = ax_map.text(cx, cy - 0.25, "--", color="#aaaaaa", fontsize=10, fontweight="bold", ha="center", va="center",
                            bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", boxstyle="round,pad=0.2"))
        textes_zones[nom] = t_nom
        textes_valeurs[nom] = t_val

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_map, fraction=0.04, pad=0.04)
    cbar.set_label("Aire d'inspiration normalisée (0 à 1)", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")

    # Boutons d'interaction
    ax_btn_rec = plt.axes([0.08, 0.035, 0.18, 0.05])
    btn_rec = Button(ax_btn_rec, "⏺ Enregistrer Zone", color="#2a52be", hovercolor="#4169e1")

    ax_btn_suiv = plt.axes([0.28, 0.035, 0.16, 0.05])
    btn_suiv = Button(ax_btn_suiv, "⏭ Zone Suivante", color="#333333", hovercolor="#555555")

    ax_btn_mode = plt.axes([0.46, 0.035, 0.16, 0.05])
    btn_mode = Button(ax_btn_mode, "📶 Basculer BLE/Simu", color="#333333", hovercolor="#555555")

    ax_btn_reset = plt.axes([0.64, 0.035, 0.14, 0.05])
    btn_reset = Button(ax_btn_reset, "🔄 Reset Zone", color="#552222", hovercolor="#773333")

    texte_statut = fig.text(0.5, 0.965, "Initialisation...", color="#ffff55", fontsize=11, ha="center", fontweight="bold")

    etat = {
        "zone_active": zone_active,
        "idx_zone": idx_zone,
        "aires": aires_zones,
        "derniere_aire_live": 0.0,
    }

    def selectionner_zone(nom_zone):
        if nom_zone in NOMS_ZONES:
            etat["zone_active"] = nom_zone
            etat["idx_zone"] = NOMS_ZONES.index(nom_zone)

    def zone_suivante(event=None):
        nouvel_idx = (etat["idx_zone"] + 1) % len(NOMS_ZONES)
        selectionner_zone(NOMS_ZONES[nouvel_idx])

    def basculer_rec(event=None):
        if not flux.en_enregistrement:
            flux.demarrer_enregistrement()
            btn_rec.label.set_text("⏹ Valider & Sauvegarder")
            btn_rec.color = "#bb2222"
        else:
            t_rec, r_rec = flux.arreter_enregistrement()
            btn_rec.label.set_text("⏺ Enregistrer Zone")
            btn_rec.color = "#2a52be"

            if len(t_rec) > int(FS_REEL * 3.5):
                t_reg, r_reg = reechantillonner(t_rec, r_rec, FS_CIBLE)
                r_propre = medfilt(r_reg, kernel_size=5)
                sig_f = filtre_passe_bande(r_propre, BANDE_RESP[0], BANDE_RESP[1], FS_CIBLE)

                aire, _, _ = calculer_aire_inspiration(t_reg, sig_f, FS_CIBLE)
                nom_z = etat["zone_active"]
                etat["aires"][nom_z] = aire
                signaux_enregistres[nom_z] = (t_reg, sig_f)
                mettre_a_jour_faisceau()

                chemin = os.path.join(REP_BASE, ZONES[nom_z]["fichier"])
                df_save = pd.DataFrame({"Timestamp": t_rec, "RSSI": r_rec})
                df_save.to_csv(chemin, index=False)
                print(f"[SUCCES] Zone '{nom_z}' enregistrée : Aire = {aire:.3f} -> '{ZONES[nom_z]['fichier']}'")

                zone_suivante()

    def basculer_mode(event=None):
        flux.mode_simu = not flux.mode_simu
        mode_str = "Simulation" if flux.mode_simu else "BLE"
        print(f"[MODE] Basculement en mode {mode_str}")

    def reset_zone(event=None):
        nom_z = etat["zone_active"]
        etat["aires"][nom_z] = 0.0
        if nom_z in signaux_enregistres:
            del signaux_enregistres[nom_z]
        mettre_a_jour_faisceau()
        chemin = os.path.join(REP_BASE, ZONES[nom_z]["fichier"])
        if os.path.exists(chemin):
            try:
                os.remove(chemin)
            except OSError:
                pass
        print(f"[RESET] Zone '{nom_z}' réinitialisée.")

    def on_click(event):
        if event.inaxes == ax_map:
            for nom, poly in patches_polygones.items():
                contient, _ = poly.contains(event)
                if contient:
                    selectionner_zone(nom)
                    break

    def on_key(event):
        if event.key == " ":
            basculer_rec()
        elif event.key in ("right", "down", "tab"):
            zone_suivante()
        elif event.key in ("left", "up"):
            nouvel_idx = (etat["idx_zone"] - 1) % len(NOMS_ZONES)
            selectionner_zone(NOMS_ZONES[nouvel_idx])
        elif event.key in ("s", "S"):
            basculer_mode()

    btn_rec.on_clicked(basculer_rec)
    btn_suiv.on_clicked(zone_suivante)
    btn_mode.on_clicked(basculer_mode)
    btn_reset.on_clicked(reset_zone)
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.ion()
    plt.show(block=False)
    fig.canvas.flush_events()

    # Démarrage sécurisé du thread BLE une fois le GUI complètement initialisé
    flux.demarrer()
    fig.canvas.mpl_connect("close_event", lambda event: setattr(flux, "en_cours_execution", False))

    print("=================================================================")
    print("MONITEUR 5 POSITIONS EN TEMPS REEL DEMARRE !")
    print(" - Clic sur une zone pour la sélectionner")
    print(" - Touche [Espace] ou bouton 'Enregistrer' pour capturer une zone")
    print(" - Touche [Flèches] pour changer de zone")
    print("=================================================================")

    dernier_temps_affiche = time.time()
    try:
        while plt.fignum_exists(fig.number):
            t_maintenant = time.time()
            if t_maintenant - dernier_temps_affiche >= 0.06:
                dernier_temps_affiche = t_maintenant

                t_arr, y_arr, en_rec, t_deb, n_pts = flux.obtenir_donnees_recentes()
                nom_z_active = etat["zone_active"]

                if len(y_arr) > int(FS_REEL * 2.5):
                    # 1. Mise à jour du signal direct RSSI en dBm réels
                    t_fin = t_arr[-1]
                    t_debut_fenetre = max(0, t_fin - 15.0)
                    masque = t_arr >= t_debut_fenetre

                    t_fenetre = t_arr[masque]
                    y_fenetre = y_arr[masque]

                    ligne_rssi.set_data(t_fenetre, y_fenetre)
                    ax_rssi.set_xlim(t_debut_fenetre, max(t_debut_fenetre + 15.0, t_fin))
                    y_min, y_max = np.min(y_fenetre), np.max(y_fenetre)
                    ax_rssi.set_ylim(min(y_min - 3.0, -90.0), max(y_max + 3.0, -45.0))
                    txt_rssi_instantane.set_text(f"RSSI direct : {y_fenetre[-1]:.1f} dBm")

                    # 2. Rééchantillonnage à 20 Hz et calcul respiratoire (exact comme poumons-5pos)
                    t_reg, r_reg = reechantillonner(t_fenetre, y_fenetre, FS_CIBLE)
                    if len(r_reg) > 20:
                        r_propre = medfilt(r_reg, kernel_size=5)
                        sig_resp = filtre_passe_bande(r_propre, BANDE_RESP[0], BANDE_RESP[1], FS_CIBLE)

                        aire_live, creux, pics = calculer_aire_inspiration(t_reg, sig_resp, FS_CIBLE)
                        etat["derniere_aire_live"] = aire_live

                        t_resp_absolu = t_debut_fenetre + t_reg
                        ligne_resp.set_data(t_resp_absolu, sig_resp)

                        if len(creux) > 0:
                            creux_valides = creux[creux < len(t_reg)]
                            pts_creux.set_data(t_debut_fenetre + t_reg[creux_valides], sig_resp[creux_valides])
                        else:
                            pts_creux.set_data([], [])

                        if len(pics) > 0:
                            pics_valides = [p for p in pics if p < len(t_reg)]
                            pts_pics.set_data(t_debut_fenetre + t_reg[pics_valides], sig_resp[pics_valides])
                        else:
                            pts_pics.set_data([], [])

                        y_amp_max = max(np.max(np.abs(sig_resp)) * 1.3, 1.8)
                        ax_resp.set_ylim(-y_amp_max, y_amp_max)

                # Bandeau de statut
                if en_rec:
                    duree_ecoulee = t_maintenant - t_deb
                    pct = min(100, int((duree_ecoulee / DUREE_ENREGISTREMENT) * 100))
                    statut_str = (
                        f"🔴 ENREGISTREMENT : Zone [{nom_z_active}] | "
                        f"{duree_ecoulee:.1f}s / {DUREE_ENREGISTREMENT:.0f}s ({pct}%) | "
                        f"Aire live: {etat['derniere_aire_live']:.2f} | {flux.etat_connexion}"
                    )
                    texte_statut.set_color("#ff4444")
                    if duree_ecoulee >= DUREE_ENREGISTREMENT:
                        basculer_rec()
                else:
                    statut_str = (
                        f"Zone active : [{nom_z_active}] | "
                        f"Dernière aire live: {etat['derniere_aire_live']:.2f} | {flux.etat_connexion}"
                    )
                    texte_statut.set_color("#ffff55")

                texte_statut.set_text(statut_str)

                # 3. Mise à jour de la cartographie 5x3
                valeurs_pos = [v for v in etat["aires"].values() if v > 0]
                aire_max = max(valeurs_pos) if valeurs_pos else 1.0
                if aire_max <= 0:
                    aire_max = 1.0

                for nom, poly in patches_polygones.items():
                    val = etat["aires"].get(nom, 0.0)
                    est_active = (nom == nom_z_active)

                    if est_active and en_rec:
                        val_affichee = etat["derniere_aire_live"]
                    else:
                        val_affichee = val

                    val_norm = np.clip(val_affichee / aire_max, 0.0, 1.0) if aire_max > 0 else 0.0

                    if val_affichee > 0:
                        poly.set_facecolor(cmap(norm(val_norm)))
                        textes_valeurs[nom].set_text(f"{val_norm:.2f}")
                        textes_valeurs[nom].set_color("white")
                    else:
                        poly.set_facecolor("#1e1e1e")
                        textes_valeurs[nom].set_text("--")
                        textes_valeurs[nom].set_color("#777777")

                    if est_active:
                        poly.set_edgecolor("#ffd700" if (int(t_maintenant * 3) % 2 == 0 or not en_rec) else "#ff3333")
                        poly.set_linewidth(4.0)
                        textes_zones[nom].set_color("#ffd700")
                        textes_zones[nom].set_fontweight("bold")
                    else:
                        poly.set_edgecolor("#333333")
                        poly.set_linewidth(1.8)
                        textes_zones[nom].set_color("white")
                        textes_zones[nom].set_fontweight("normal")

                fig.canvas.draw_idle()
                fig.canvas.flush_events()

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\nArrêt de l'application...")
    finally:
        flux.en_cours_execution = False
        plt.close("all")
        print("Application fermée proprement.")

if __name__ == "__main__":
    main()
