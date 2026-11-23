# Affiche le classement des modèles de l'étape 2 à partir de results/summary.csv (rempli par evaluate.py)
# et l'écrit dans results/classement.md, prêt à copier dans le README.
#   python scripts/classement.py
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parent.parent
SUMMARY = RACINE / "results" / "summary.csv"


def main():
    if not SUMMARY.exists():
        print("results/summary.csv n'existe pas encore : aucun modèle noté")
        return
    t = pd.read_csv(SUMMARY).sort_values("f1_macro", ascending=False).reset_index(drop=True)
    lignes = ["| Rang | Membre | Modèle | Précision macro | Rappel macro | F1 macro | Type le plus raté (F1) "
              "| Temps d'entraînement (min) |",
              "|---|---|---|---|---|---|---|---|"]
    for i, r in t.iterrows():
        temps = "" if pd.isna(r.get("train_time_min")) else f"{r['train_time_min']:g}"
        lignes.append(f"| {i + 1} | {r['member']} | {r['model']} | {r['precision_macro']:.3f} "
                      f"| {r['recall_macro']:.3f} | {r['f1_macro']:.3f} | {r['pire_type']} ({r['f1_pire_type']:.2f}) "
                      f"| {temps} |")
    texte = "\n".join(lignes)
    (RACINE / "results" / "classement.md").write_text(texte + "\n", encoding="utf-8")
    print(texte)


if __name__ == "__main__":
    main()
