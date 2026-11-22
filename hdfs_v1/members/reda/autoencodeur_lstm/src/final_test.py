# Étape 3 : réentraîner le réglage retenu sur tout train normal, fixer le seuil sur val normal,
# passer le test UNE seule fois, écrire results/autoencoder_test.csv et le rapport de la nuit.
import sys
import time

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (RESULTS, FIGS, RAPIDE, SEED, QUANTILE_SEUIL, DEVICE, journal, charger_sequences,
                    fixer_graine, garder_pc_eveille, to_tensor, LSTMAutoencoder, entrainer, scores_par_bloc,
                    reconstruire, mesures, ecrire_json, lire_json, templates)

log = journal("3_test_final")
MAX_EPOQUES = 2 if RAPIDE else 30
PATIENCE = 3
MODELE = RESULTS / "autoencoder_best.pt"
SORTIE = RESULTS / "autoencoder_test.csv"


def fmt_seq(seq, n=40):
    s = " ".join(f"E{e}" for e in seq[:n])
    return s + (f" ... (+{len(seq) - n})" if len(seq) > n else "")


def main():
    garder_pc_eveille()
    choix = lire_json(RESULTS / "meilleur_reglage.json")
    L = choix["L"]
    df, officiel = charger_sequences()
    train_normal = df[(df["split"] == "train") & (df["label"] == 0)]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]
    if RAPIDE:
        train_normal = train_normal.sample(min(len(train_normal), 3000), random_state=SEED)
    val_normal = val[val["label"] == 0]
    log.info(f"réglage retenu : latent {choix['latent']}, hidden {choix['hidden']}, L {L}")
    log.info(f"train normal {len(train_normal)}, val {len(val)}, test {len(test)}")

    X_train = to_tensor(train_normal, L)
    X_vn = to_tensor(val_normal, L)
    X_val = to_tensor(val, L)

    model = LSTMAutoencoder(hidden=choix["hidden"], latent=choix["latent"])
    if MODELE.exists():
        log.info("modèle final déjà entraîné : chargement")
        model.load_state_dict(torch.load(MODELE, map_location="cpu"))
        model.to(DEVICE)
        info = lire_json(RESULTS / "entrainement_final.json")
    else:
        fixer_graine(SEED)
        hist, duree = entrainer(model, X_train, X_vn, MAX_EPOQUES, PATIENCE, log, prefixe="[final] ")
        torch.save(model.state_dict(), MODELE)
        info = {"historique": hist, "temps_min": round(duree / 60, 1), "machine": str(DEVICE)}
        ecrire_json(RESULTS / "entrainement_final.json", info)
        h = pd.DataFrame(hist)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(h["epoque"], h["train"], label="train", color="#1F3A5F", lw=2)
        ax.plot(h["epoque"], h["val"], label="val normal", color="#D9822B", lw=2)
        ax.set_xlabel("époque"); ax.set_ylabel("perte"); ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(FIGS / "courbe_perte.png", dpi=150); plt.close(fig)

    # Seuil : quantile 99,5 % des scores des normaux de val
    s_vn = scores_par_bloc(model, X_vn)
    seuil = float(np.quantile(s_vn, QUANTILE_SEUIL))
    s_val = scores_par_bloc(model, X_val)
    m_val = mesures(val["label"].values, s_val > seuil)
    log.info(f"seuil {seuil:.5f} ; sur val : F1 {m_val['f1']:.4f}, précision {m_val['precision']:.4f}, rappel {m_val['rappel']:.4f}")

    y = val["label"].values
    fig, ax = plt.subplots(figsize=(8, 3.4))
    lo = max(s_val.min(), 1e-6)
    bins = np.logspace(np.log10(lo), np.log10(s_val.max() + 1e-6), 80)
    ax.hist(s_val[y == 0], bins=bins, alpha=0.75, density=True, color="#2A7F8E", label="normaux")
    ax.hist(s_val[y == 1], bins=bins, alpha=0.6, density=True, color="#B9403A", label="anomalies")
    ax.axvline(seuil, color="#D9822B", lw=2, label="seuil")
    ax.set_xscale("log"); ax.set_xlabel("score d'anomalie (val)"); ax.set_ylabel("densité")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIGS / "scores_val.png", dpi=150); plt.close(fig)

    # Test : une seule fois
    X_test = to_tensor(test, L)
    t0 = time.time()
    s_test = scores_par_bloc(model, X_test)
    duree_pred = time.time() - t0
    pred = (s_test > seuil).astype(int)
    resultat = pd.DataFrame({"block_id": test["block_id"].values, "score": s_test, "pred": pred})
    assert list(resultat.columns) == ["block_id", "score", "pred"] and len(resultat) == len(test)
    resultat.to_csv(SORTIE, index=False)
    m_test = mesures(test["label"].values, pred)
    log.info(f"TEST : F1 {m_test['f1']:.4f}, précision {m_test['precision']:.4f}, rappel {m_test['rappel']:.4f}"
             f"  (VP {m_test['VP']}, FP {m_test['FP']}, FN {m_test['FN']}, VN {m_test['VN']})")
    ecrire_json(RESULTS / "mesures.json", {"seuil": seuil, "val": m_val, "test": m_test, "reglage": choix,
                                          "temps_entrainement_min": info["temps_min"],
                                          "temps_prediction_test_s": round(duree_pred, 1),
                                          "machine": info.get("machine", str(DEVICE)),
                                          "donnees": "officielles" if officiel else "provisoires"})

    # Analyse
    tpl = templates()
    vus_normaux = set(tuple(s) for s in train_normal["events"])
    t = test.reset_index(drop=True).assign(score=s_test, pred=pred)

    def bloc_txt(i, titre):
        seq = list(t.loc[i, "events"])
        rec = reconstruire(model, X_test[i:i + 1])[0][:min(len(seq), L)].tolist()
        diff = [k for k, (a, b) in enumerate(zip(seq[:L], rec)) if a != b]
        lignes = [f"### {titre} : {t.loc[i, 'block_id']}",
                  f"- label réel : {'anomalie' if t.loc[i, 'label'] else 'normal'} ; prédiction : "
                  f"{'alerte' if t.loc[i, 'pred'] else 'normal'} ; score {t.loc[i, 'score']:.4f} (seuil {seuil:.4f})",
                  f"- longueur : {len(seq)} événements" + (f", coupé à L = {L}" if len(seq) > L else ""),
                  f"- séquence identique à un bloc normal de train : {'oui' if tuple(seq) in vus_normaux else 'non'}",
                  f"- origine     : `{fmt_seq(seq)}`",
                  f"- reconstruit : `{fmt_seq(rec)}`",
                  f"- positions mal reconstruites : {len(diff)} sur {min(len(seq), L)}"]
        rares = sorted({seq[k] for k in diff})[:6]
        if tpl and rares:
            lignes.append("- événements concernés : " + " ; ".join(f"E{e} = {tpl.get(e, '?')}" for e in rares))
        return "\n".join(lignes)

    pires = list(np.argsort(-s_test)[:3])
    anomalies = t[t["label"] == 1]
    trouvees = list(anomalies[anomalies["pred"] == 1].sort_values("score", ascending=False).index[:3])
    ratees = list(anomalies[anomalies["pred"] == 0].sort_values("score").index[:3])

    r = ["# Autoencodeur LSTM : résultats de la nuit", ""]
    if not officiel:
        r += ["Données PROVISOIRES (sequences_provisoire.parquet construit depuis le zip Loghub).",
              "Les chiffres officiels viendront de sequences.parquet de Mouad et de evaluate.py.", ""]
    if RAPIDE:
        r += ["MODE RAPIDE : test de fonctionnement seulement, chiffres sans valeur.", ""]
    r += ["## Réglages comparés sur val", ""]
    tab = pd.read_csv(RESULTS / "reglages_val.csv")
    r += ["| " + " | ".join(tab.columns) + " |", "|" + "---|" * len(tab.columns)]
    r += ["| " + " | ".join(str(v) for v in row) + " |" for row in tab.values]
    r += ["", f"Réglage retenu : latent {choix['latent']}, hidden {choix['hidden']}, L {L} ({choix['L_nom']}).", "",
          "## Entraînement final", "",
          f"- époques : {len(info['historique'])} ; temps : {info['temps_min']} min ; machine : {info.get('machine', '')}",
          f"- seuil (quantile 99,5 % des normaux de val) : {seuil:.5f}",
          f"- val : F1 {m_val['f1']:.4f}, précision {m_val['precision']:.4f}, rappel {m_val['rappel']:.4f}", "",
          "## Test (une seule fois)", "",
          f"- précision {m_test['precision']:.4f}, rappel {m_test['rappel']:.4f}, F1 {m_test['f1']:.4f}",
          f"- matrice de confusion : VP {m_test['VP']}, FP {m_test['FP']}, FN {m_test['FN']}, VN {m_test['VN']}",
          f"- temps de prédiction sur le test : {duree_pred:.1f} s",
          "- fichier : results/autoencoder_test.csv", "",
          "## Les 3 blocs les plus mal reconstruits", ""]
    r += [bloc_txt(i, f"Pire n°{k + 1}") + "\n" for k, i in enumerate(pires)]
    r += ["## 3 anomalies trouvées", ""]
    r += [bloc_txt(i, f"Trouvée n°{k + 1}") + "\n" for k, i in enumerate(trouvees)]
    r += ["## 3 anomalies ratées (les scores les plus bas)", ""]
    r += [bloc_txt(i, f"Ratée n°{k + 1}") + "\n" for k, i in enumerate(ratees)]
    r += ["## Figures", "", "- results/figures/longueurs.png", "- results/figures/courbe_perte.png",
          "- results/figures/scores_val.png", ""]
    (RESULTS / "rapport_nuit.md").write_text("\n".join(r), encoding="utf-8")
    log.info("rapport écrit : results/rapport_nuit.md")


if __name__ == "__main__":
    try:
        t0 = time.time()
        main()
        log.info(f"étape 3 terminée en {(time.time() - t0) / 60:.0f} min")
    except Exception as e:
        log.exception(f"ERREUR : {e}")
        sys.exit(1)
