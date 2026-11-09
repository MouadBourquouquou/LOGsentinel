# Script de score commun : le seul autorisé pour les chiffres du tableau final.
#
# Utilisation :
#   python evaluate.py --pred results/pca_test.csv --model PCA --member Mouad
# Options :
#   --train-time 12.5     temps d'entraînement en minutes
#   --pred-time 3.2       temps de prédiction sur test en secondes
#   --machine "PC i5 16 Go, CPU"
#   --force               réécrire une ligne déjà présente (à éviter : le test se passe une seule fois)
#
# Le fichier de prédictions doit contenir une ligne par bloc de test, avec les colonnes
# block_id, score, pred (score plus haut = moins normal ; pred = 1 si alerte, 0 sinon).
import sys
import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parent


def charger_verite(chemin_data):
    for nom in ("sequences.parquet", "counts.parquet"):
        p = Path(chemin_data) / nom
        if p.exists():
            df = pd.read_parquet(p, columns=["block_id", "label", "split"])
            return df[df["split"] == "test"].set_index("block_id")["label"].astype(int)
    sys.exit(f"ni sequences.parquet ni counts.parquet dans {chemin_data}")


def main():
    ap = argparse.ArgumentParser(description="Score commun LogSentinel (HDFS_v1, partie test)")
    ap.add_argument("--pred", required=True, help="fichier CSV block_id,score,pred")
    ap.add_argument("--model", required=True, help="nom du modèle, par exemple PCA")
    ap.add_argument("--member", required=True, help="prénom du membre")
    ap.add_argument("--data", default=str(RACINE / "data"), help="dossier des fichiers communs")
    ap.add_argument("--summary", default=str(RACINE / "results" / "summary.csv"))
    ap.add_argument("--train-time", type=float, default=None, help="minutes")
    ap.add_argument("--pred-time", type=float, default=None, help="secondes")
    ap.add_argument("--machine", default="")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    verite = charger_verite(a.data)
    pred = pd.read_csv(a.pred, dtype={"block_id": str})

    # Vérifications du format
    erreurs = []
    manquantes = {"block_id", "score", "pred"} - set(pred.columns)
    if manquantes:
        sys.exit(f"colonnes manquantes : {sorted(manquantes)} (attendu : block_id,score,pred)")
    if pred["block_id"].duplicated().any():
        erreurs.append(f"{int(pred['block_id'].duplicated().sum())} block_id en double")
    absents = set(verite.index) - set(pred["block_id"])
    en_trop = set(pred["block_id"]) - set(verite.index)
    if absents:
        erreurs.append(f"{len(absents)} blocs de test absents du fichier")
    if en_trop:
        erreurs.append(f"{len(en_trop)} blocs qui ne sont pas dans test (train ou val ?)")
    if not set(pred["pred"].unique()) <= {0, 1}:
        erreurs.append(f"pred doit valoir 0 ou 1, trouvé : {sorted(pred['pred'].unique())[:5]}")
    if pred["score"].isna().any() or not np.isfinite(pred["score"].astype(float)).all():
        erreurs.append("score contient des valeurs vides ou infinies")
    if erreurs:
        sys.exit("fichier refusé :\n  - " + "\n  - ".join(erreurs))

    # Cohérence score / pred (avertissement seulement)
    alertes, normaux = pred.loc[pred["pred"] == 1, "score"], pred.loc[pred["pred"] == 0, "score"]
    if len(alertes) and len(normaux) and alertes.min() < normaux.max():
        print("attention : certains blocs en alerte ont un score plus bas que des blocs non alertés ;"
              " vérifier que pred = (score > seuil)")

    # Mesures
    y = verite.loc[pred["block_id"]].values
    p = pred["pred"].astype(int).values
    vp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    vn = int(((y == 0) & (p == 0)).sum())
    precision = vp / (vp + fp) if vp + fp else 0.0
    rappel = vp / (vp + fn) if vp + fn else 0.0
    f1 = 2 * precision * rappel / (precision + rappel) if precision + rappel else 0.0

    print(f"\n{a.member} | {a.model} | {len(y)} blocs de test, {int(y.sum())} anomalies")
    print(f"précision {precision:.4f}   rappel {rappel:.4f}   F1 {f1:.4f}\n")
    print("matrice de confusion       prédit normal   prédit alerte")
    print(f"  réellement normal        {vn:13d}   {fp:13d}")
    print(f"  réellement anomalie      {fn:13d}   {vp:13d}\n")

    # Ajout au tableau commun
    chemin = Path(a.summary)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    tab = pd.read_csv(chemin) if chemin.exists() else pd.DataFrame()
    deja = len(tab) and ((tab["member"] == a.member) & (tab["model"] == a.model)).any()
    if deja and not a.force:
        sys.exit(f"{a.member} / {a.model} est déjà dans {chemin.name} : le test ne se passe qu'une fois. "
                 "Utiliser --force seulement en cas d'erreur de fichier, et prévenir le groupe.")
    if deja:
        tab = tab[~((tab["member"] == a.member) & (tab["model"] == a.model))]
    ligne = {"date": dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "member": a.member, "model": a.model,
             "precision": round(precision, 4), "recall": round(rappel, 4), "f1": round(f1, 4),
             "VP": vp, "FP": fp, "FN": fn, "VN": vn, "n_test": len(y), "n_anomalies": int(y.sum()),
             "train_time_min": a.train_time, "pred_time_s": a.pred_time, "machine": a.machine,
             "pred_file": Path(a.pred).name}
    tab = pd.concat([tab, pd.DataFrame([ligne])], ignore_index=True)
    tab.to_csv(chemin, index=False)
    print(f"ligne ajoutée à {chemin}\n")
    print(tab[["member", "model", "precision", "recall", "f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
