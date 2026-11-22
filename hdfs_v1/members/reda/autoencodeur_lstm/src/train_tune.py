# Étape 2 : comparer les réglages sur val (F1), un réglage à la fois.
# Pour aller vite, chaque essai s'entraîne sur un échantillon de 100 000 blocs normaux de train.
# Le meilleur réglage est réentraîné sur tout train normal à l'étape 3.
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (RESULTS, FIGS, RAPIDE, SEED, QUANTILE_SEUIL, journal, charger_sequences, fixer_graine,
                    garder_pc_eveille, to_tensor, LSTMAutoencoder, entrainer, scores_par_bloc, mesures,
                    ecrire_json)

log = journal("2_reglages")
TABLE = RESULTS / "reglages_val.csv"
TAILLE_ECHANTILLON = 3000 if RAPIDE else 100_000
MAX_EPOQUES = 2 if RAPIDE else 20
PATIENCE = 3
L_MAX = 120


def main():
    garder_pc_eveille()
    df, officiel = charger_sequences()
    log.info(f"données : {'sequences.parquet (officiel)' if officiel else 'sequences_provisoire.parquet'}, {len(df)} blocs")
    train_normal = df[(df["split"] == "train") & (df["label"] == 0)]
    val = df[df["split"] == "val"]
    val_normal = val[val["label"] == 0]
    if RAPIDE:
        val = val.sample(min(len(val), 4000), random_state=SEED)
        val_normal = val[val["label"] == 0]
    assert (train_normal["label"] == 0).all()
    log.info(f"train normal {len(train_normal)}, val {len(val)} dont {int(val['label'].sum())} anomalies")

    # Longueur fixe L, calculée sur train normal uniquement
    lg = train_normal["events"].apply(len)
    centiles = {"p95": 95, "p99": 99, "p99.9": 99.9}
    L = {k: int(min(L_MAX, max(5, np.ceil(np.percentile(lg, v))))) for k, v in centiles.items()}
    log.info(f"longueurs train normal : min {lg.min()}, médiane {int(lg.median())}, max {lg.max()} ; L = {L}")
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.hist(lg.clip(upper=150), bins=range(0, 152), color="#2A7F8E")
    for k, c in zip(L, ["#999999", "#D9822B", "#555555"]):
        ax.axvline(L[k], color=c, lw=1.8, label=f"{k} : L = {L[k]}")
    ax.set_xlabel("nombre d'événements dans le bloc (train normal)")
    ax.set_ylabel("nombre de blocs")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGS / "longueurs.png", dpi=150)
    plt.close(fig)

    essais = [
        {"essai": 1, "latent": 32, "hidden": 64, "L_nom": "p99"},
        {"essai": 2, "latent": 16, "hidden": 64, "L_nom": "p99"},
        {"essai": 3, "latent": 64, "hidden": 64, "L_nom": "p99"},
        {"essai": 4, "latent": 32, "hidden": 128, "L_nom": "p99"},
        {"essai": 5, "latent": 32, "hidden": 64, "L_nom": "p95"},
        {"essai": 6, "latent": 32, "hidden": 64, "L_nom": "p99.9"},
    ]
    # Si L p95 = p99 (ou p99.9 = p99), l'essai serait identique : on le saute
    vus = set()
    for e in essais:
        e["L"] = L[e["L_nom"]]
        cle = (e["latent"], e["hidden"], e["L"])
        e["doublon"] = cle in vus
        vus.add(cle)

    deja = pd.read_csv(TABLE) if TABLE.exists() else pd.DataFrame()
    faits = set(deja["essai"]) if len(deja) else set()
    lignes = deja.to_dict("records") if len(deja) else []

    echantillon = train_normal.sample(min(TAILLE_ECHANTILLON, len(train_normal)), random_state=SEED)
    cache = {}
    for e in essais:
        if e["essai"] in faits:
            log.info(f"essai {e['essai']} déjà fait, sauté")
            continue
        if e["doublon"]:
            log.info(f"essai {e['essai']} identique à un essai précédent (même L), sauté")
            continue
        log.info(f"===== essai {e['essai']} : latent {e['latent']}, hidden {e['hidden']}, L {e['L']} ({e['L_nom']}) =====")
        Lx = e["L"]
        if Lx not in cache:
            cache[Lx] = (to_tensor(echantillon, Lx), to_tensor(val_normal, Lx), to_tensor(val, Lx))
        X_tr, X_vn, X_v = cache[Lx]
        fixer_graine(SEED)
        model = LSTMAutoencoder(hidden=e["hidden"], latent=e["latent"])
        hist, duree = entrainer(model, X_tr, X_vn, MAX_EPOQUES, PATIENCE, log, prefixe=f"[essai {e['essai']}] ")
        seuil = float(np.quantile(scores_par_bloc(model, X_vn), QUANTILE_SEUIL))
        s_val = scores_par_bloc(model, X_v)
        m = mesures(val["label"].values, s_val > seuil)
        meilleure = min(h["val"] for h in hist)
        ligne = {"essai": e["essai"], "latent": e["latent"], "hidden": e["hidden"], "L_nom": e["L_nom"], "L": Lx,
                 "epoques": len(hist), "perte_val_normal": round(meilleure, 5), "seuil": round(seuil, 5),
                 "precision_val": round(m["precision"], 4), "rappel_val": round(m["rappel"], 4),
                 "f1_val": round(m["f1"], 4), "temps_min": round(duree / 60, 1)}
        log.info(f"[essai {e['essai']}] F1 val {m['f1']:.4f}  précision {m['precision']:.4f}  rappel {m['rappel']:.4f}"
                 f"  seuil {seuil:.4f}  ({duree / 60:.1f} min)")
        lignes.append(ligne)
        pd.DataFrame(lignes).sort_values("essai").to_csv(TABLE, index=False)

    tab = pd.DataFrame(lignes).sort_values("essai")
    log.info("tableau des réglages :\n" + tab.to_string(index=False))
    # Meilleur F1 val ; à égalité, le modèle le plus simple
    tab["_cout"] = tab["hidden"] * 1000 + tab["latent"] * 10 + tab["L"] / 1000
    best = tab.sort_values(["f1_val", "_cout"], ascending=[False, True]).iloc[0]
    choix = {"latent": int(best["latent"]), "hidden": int(best["hidden"]), "L": int(best["L"]),
             "L_nom": best["L_nom"], "essai": int(best["essai"]), "f1_val_essai": float(best["f1_val"])}
    ecrire_json(RESULTS / "meilleur_reglage.json", choix)
    log.info(f"réglage retenu : {choix}")


if __name__ == "__main__":
    try:
        t0 = time.time()
        main()
        log.info(f"étape 2 terminée en {(time.time() - t0) / 60:.0f} min")
    except Exception as e:
        log.exception(f"ERREUR : {e}")
        sys.exit(1)
