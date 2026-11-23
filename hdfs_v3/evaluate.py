# Script de score commun de l'étape 2 : le seul autorisé pour les chiffres du tableau final.
#
# Utilisation (depuis le dossier hdfs_v3) :
#   python evaluate.py --pred results/random_forest_test.csv --model "Random Forest" --member Reda
# Options :
#   --train-time 12.5     temps d'entraînement en minutes
#   --pred-time 3.2       temps de prédiction sur test en secondes
#   --machine "PC i5 16 Go, CPU"
#   --force               réécrire une ligne déjà présente (à éviter : le test se passe une seule fois)
#
# Le fichier de prédictions contient une ligne par trace de test, avec les colonnes trace_id, pred
# (pred = le label du type de panne prédit, un entier ; voir data/vocabulaire.json). Des colonnes proba_k
# peuvent suivre, elles sont gardées pour la chaîne finale mais ne changent pas le score.
import sys
import json
import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description="Score commun LogSentinel (HDFS_v3, diagnostic, partie test)")
    ap.add_argument("--pred", required=True, help="fichier CSV trace_id,pred")
    ap.add_argument("--model", required=True, help="nom du modèle, par exemple Random Forest")
    ap.add_argument("--member", required=True, help="prénom du membre")
    ap.add_argument("--data", default=str(RACINE / "data"), help="dossier des fichiers communs")
    ap.add_argument("--summary", default=str(RACINE / "results" / "summary.csv"))
    ap.add_argument("--train-time", type=float, default=None, help="minutes")
    ap.add_argument("--pred-time", type=float, default=None, help="secondes")
    ap.add_argument("--machine", default="")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    data = Path(a.data)
    pannes = json.loads((data / "vocabulaire.json").read_text(encoding="utf-8"))["pannes"]
    k = len(pannes)
    t = pd.read_parquet(data / "traces.parquet", columns=["trace_id", "label", "split"])
    verite = t[t["split"] == "test"].set_index("trace_id")["label"].astype(int)
    pred = pd.read_csv(a.pred, dtype={"trace_id": str})

    # Vérifications du format
    erreurs = []
    manquantes = {"trace_id", "pred"} - set(pred.columns)
    if manquantes:
        sys.exit(f"colonnes manquantes : {sorted(manquantes)} (attendu : trace_id,pred)")
    if pred["trace_id"].duplicated().any():
        erreurs.append(f"{int(pred['trace_id'].duplicated().sum())} trace_id en double")
    absents = set(verite.index) - set(pred["trace_id"])
    en_trop = set(pred["trace_id"]) - set(verite.index)
    if absents:
        erreurs.append(f"{len(absents)} traces de test absentes du fichier")
    if en_trop:
        erreurs.append(f"{len(en_trop)} traces qui ne sont pas dans test (train ou val ?)")
    if pred["pred"].isna().any() or not set(pred["pred"].astype(int).unique()) <= set(range(k)):
        erreurs.append(f"pred doit être un entier de 0 à {k - 1}")
    if erreurs:
        sys.exit("fichier refusé :\n  - " + "\n  - ".join(erreurs))

    # Mesures
    y = verite.loc[pred["trace_id"]].values
    p = pred["pred"].astype(int).values
    mc = np.zeros((k, k), dtype=int)
    np.add.at(mc, (y, p), 1)
    vp = np.diag(mc).astype(float)
    prec = np.divide(vp, mc.sum(axis=0), out=np.zeros(k), where=mc.sum(axis=0) > 0)
    rapp = np.divide(vp, mc.sum(axis=1), out=np.zeros(k), where=mc.sum(axis=1) > 0)
    f1 = np.divide(2 * prec * rapp, prec + rapp, out=np.zeros(k), where=(prec + rapp) > 0)
    p_macro, r_macro, f1_macro = prec.mean(), rapp.mean(), f1.mean()
    pire = int(np.argmin(f1))

    print(f"\n{a.member} | {a.model} | {len(y)} traces de test, {k} types de panne")
    print(f"précision macro {p_macro:.4f}   rappel macro {r_macro:.4f}   F1 macro {f1_macro:.4f}\n")
    print(f"{'type de panne':20s} {'précision':>9s} {'rappel':>7s} {'F1':>6s} {'traces':>7s}")
    for i, nom in enumerate(pannes):
        print(f"{nom:20s} {prec[i]:9.3f} {rapp[i]:7.3f} {f1[i]:6.3f} {int(mc[i].sum()):7d}")
    court = [f"{i:>5d}" for i in range(k)]
    print("\nmatrice de confusion (lignes : vrai type, colonnes : type prédit, numéros = labels)")
    print(f"{'':>24s}" + "".join(court))
    for i, nom in enumerate(pannes):
        print(f"{i:2d} {nom:20s} " + "".join(f"{v:5d}" for v in mc[i]))
    print()

    # Matrice de confusion gardée pour le rapport
    dossier_mc = Path(a.summary).parent / "confusion"
    dossier_mc.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(mc, index=pannes, columns=pannes).to_csv(dossier_mc / f"{Path(a.pred).stem}.csv")

    # Ajout au tableau commun
    chemin = Path(a.summary)
    tab = pd.read_csv(chemin) if chemin.exists() else pd.DataFrame()
    deja = len(tab) and ((tab["member"] == a.member) & (tab["model"] == a.model)).any()
    if deja and not a.force:
        sys.exit(f"{a.member} / {a.model} est déjà dans {chemin.name} : le test ne se passe qu'une fois. "
                 "Utiliser --force seulement en cas d'erreur de fichier, et prévenir le groupe.")
    if deja:
        tab = tab[~((tab["member"] == a.member) & (tab["model"] == a.model))]
    ligne = {"date": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "member": a.member, "model": a.model,
             "precision_macro": round(p_macro, 4), "recall_macro": round(r_macro, 4),
             "f1_macro": round(f1_macro, 4), "pire_type": pannes[pire], "f1_pire_type": round(f1[pire], 4),
             "n_test": len(y), "train_time_min": a.train_time, "pred_time_s": a.pred_time, "machine": a.machine,
             "pred_file": Path(a.pred).name}
    ligne.update({f"f1_{nom}": round(f1[i], 4) for i, nom in enumerate(pannes)})
    tab = pd.concat([tab, pd.DataFrame([ligne])], ignore_index=True)
    tab.to_csv(chemin, index=False)
    print(f"ligne ajoutée à {chemin}\n")
    print(tab[["member", "model", "precision_macro", "recall_macro", "f1_macro"]].to_string(index=False))


if __name__ == "__main__":
    main()
