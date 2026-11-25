# Petit outil commun pour charger les fichiers de l'étape 2 de la même façon partout.
#
# Depuis un notebook dans members/<prenom>/ :
#   import sys; sys.path.append("../..")
#   from donnees import charger_traces, parties, colonnes_caract, ecrire_predictions
#   df = charger_traces()
#   p = parties(df)
#   C = colonnes_caract()
#   X_train, y_train = p["train"][C].values, p["train"]["label"].values
#   X_val, y_val = p["val"][C].values, p["val"]["label"].values
from pathlib import Path
import json

import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parent
DATA = RACINE / "data"
GRAINE = 42

# Colonnes de traces.parquet qui ne sont PAS des caractéristiques : ne pas les donner aux modèles.
# "scenario" et "panne" contiennent la réponse, "split" et "trace_id" n'ont aucun sens pour un modèle.
COLONNES_META = ["trace_id", "label", "panne", "categorie", "charge", "scenario", "titre", "split"]


def vocabulaire(chemin=DATA / "vocabulaire.json"):
    """Noms des types de panne, des opérations, des agents, des rôles, des erreurs et des caractéristiques."""
    return json.loads(Path(chemin).read_text(encoding="utf-8"))


def noms_pannes():
    """Liste des types de panne : noms_pannes()[label] donne le nom."""
    return vocabulaire()["pannes"]


def colonnes_caract():
    """Les caractéristiques résumées de traces.parquet (toutes numériques)."""
    return vocabulaire()["colonnes_caract"]


def charger_traces(chemin=DATA / "traces.parquet"):
    """Une ligne par trace anormale : métadonnées et caractéristiques résumées."""
    return pd.read_parquet(chemin)


def charger_etapes(chemin=DATA / "etapes.parquet"):
    """Une ligne par trace : listes ops, agents, roles, machines, durees_ms, erreurs (même ordre d'étapes)."""
    df = pd.read_parquet(chemin)
    for c in ("ops", "agents", "roles", "machines", "erreurs"):
        df[c] = df[c].apply(lambda s: np.asarray(s, dtype=np.int64))
    df["durees_ms"] = df["durees_ms"].apply(lambda s: np.asarray(s, dtype=np.float32))
    return df


def charger_aretes(chemin=DATA / "aretes.parquet"):
    """Une ligne par arête : trace_id, split, src, dst, type (0 = appel, 1 = suite dans le même fil).
    src et dst sont des positions dans les listes de etapes.parquet."""
    return pd.read_parquet(chemin)


def aretes_par_trace(aretes):
    """Dictionnaire trace_id -> tableau 2 x n_aretes (src, dst) et tableau des types, pour construire les graphes."""
    res = {}
    for tid, g in aretes.groupby("trace_id", sort=False):
        res[tid] = (np.vstack([g["src"].values, g["dst"].values]).astype(np.int64), g["type"].values.astype(np.int64))
    return res


def parties(df):
    """Renvoie les parties utilisées par tout le monde (même découpage pour les trois fichiers)."""
    return {
        "train": df[df["split"] == "train"],   # entraînement
        "val": df[df["split"] == "val"],       # choix des réglages (F1 macro)
        "test": df[df["split"] == "test"],     # une seule fois, à la fin
    }


def poids_classes(y):
    """Poids par type de panne, inversement proportionnels à leur fréquence (pour PyTorch ou XGBoost)."""
    y = np.asarray(y)
    n_classes = len(noms_pannes())
    compte = np.bincount(y, minlength=n_classes).astype(float)
    return len(y) / (n_classes * np.maximum(compte, 1))


def ecrire_predictions(test, pred, chemin, proba=None):
    """Écrit le fichier attendu par evaluate.py : trace_id, pred (le label prédit, un entier).
    proba (facultatif) : tableau n_traces x n_pannes, ajouté en colonnes proba_0, proba_1..."""
    out = pd.DataFrame({"trace_id": test["trace_id"].values, "pred": np.asarray(pred, dtype=int)})
    if proba is not None:
        proba = np.asarray(proba, dtype=float)
        for k in range(proba.shape[1]):
            out[f"proba_{k}"] = proba[:, k]
    Path(chemin).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(chemin, index=False)
    return out
