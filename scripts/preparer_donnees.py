# Prépare les fichiers communs de l'équipe à partir de HDFS_v1 (Loghub) :
#   data/sequences.parquet  : block_id, label, split, events
#   data/counts.parquet     : block_id, label, split, E1..E30, n_warn, n_error, length, n_unique, duration_s
#   data/templates.csv      : num, texte (le texte de chaque événement)
#   data/MANIFEST.txt       : chiffres clés et empreintes MD5 pour vérifier que tout le monde a les mêmes fichiers
#
# Utilisation (depuis la racine du dépôt) :
#   python scripts/preparer_donnees.py
# avec le dossier extrait dans data/HDFS_v1 (il contient HDFS.log et le dossier preprocessed),
# ou en donnant un autre chemin :
#   python scripts/preparer_donnees.py --source "C:\chemin\vers\HDFS_v1"
import re
import sys
import time
import hashlib
import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parent.parent
DATA = RACINE / "data"
BLK = re.compile(r"blk_-?\d+")
N_EVENEMENTS = 30


def info(msg):
    print(time.strftime("%H:%M:%S"), " ", msg, flush=True)


def trouver(dossier, motif):
    res = sorted(Path(dossier).rglob(motif), key=lambda p: len(str(p)))
    return res[0] if res else None


def colonne_blk(df):
    for c in df.columns:
        if df[c].astype(str).head(50).str.match(r"^blk_-?\d+$").mean() > 0.9:
            return c
    raise ValueError(f"aucune colonne d'identifiants blk_ dans {list(df.columns)}")


def lire_labels(chemin):
    lab = pd.read_csv(chemin)
    c_id = colonne_blk(lab)
    c_lab = next(c for c in lab.columns if c != c_id and
                 set(lab[c].astype(str).str.lower().head(2000)) & {"anomaly", "normal", "fail", "success"})
    y = lab[c_lab].astype(str).str.lower().isin(["anomaly", "fail"]).astype(int)
    info(f"labels : {len(lab)} blocs, {int(y.sum())} anomalies")
    return dict(zip(lab[c_id].astype(str), y))


def lire_traces(chemin):
    tr = pd.read_csv(chemin)
    info(f"{chemin.name} : colonnes {list(tr.columns)}")
    c_id = colonne_blk(tr)
    c_ev = next(c for c in tr.columns if c != c_id and tr[c].astype(str).head(50).str.contains(r"E\d+").mean() > 0.9)
    num = re.compile(r"E(\d+)")
    events = tr[c_ev].astype(str).apply(lambda s: [int(e) for e in num.findall(s)])
    return pd.DataFrame({"block_id": tr[c_id].astype(str), "events": events})


def lire_templates(chemin):
    t = pd.read_csv(chemin)
    c_id = next(c for c in t.columns if t[c].astype(str).str.match(r"^E\d+$").mean() > 0.9)
    c_tx = next(c for c in t.columns if c != c_id and not pd.api.types.is_numeric_dtype(t[c]))
    return pd.DataFrame({"num": t[c_id].astype(str).str[1:].astype(int), "texte": t[c_tx].astype(str)})


def parcourir_log(chemin, tpl=None):
    """Une seule lecture de HDFS.log. Pour chaque bloc : rang de première apparition, premier et dernier
    horodatage, nombre de lignes WARN et ERROR. Si tpl est donné (Event_traces.csv absent), reconstruit
    aussi les séquences."""
    info(f"lecture de {chemin} (11 millions de lignes, quelques minutes)...")
    ordre, debut, fin, n_warn, n_error = {}, {}, {}, {}, {}
    seqs = {} if tpl is not None else None
    regex = []
    if tpl is not None:
        for num, texte in zip(tpl["num"], tpl["texte"]):
            motif = re.escape(texte.strip()).replace(re.escape("[*]"), ".*?").replace(re.escape("<*>"), ".*?")
            regex.append((num, re.compile("^" + motif + "$")))
        regex.sort(key=lambda r: -len(r[1].pattern))
    jours = {}
    t0 = time.time()
    with open(chemin, "r", encoding="utf-8", errors="replace") as f:
        for i, ligne in enumerate(f):
            ids = BLK.findall(ligne)
            if ids:
                morceaux = ligne.split(" ", 5)
                try:
                    d, h = morceaux[0], morceaux[1]
                    if d not in jours:
                        jours[d] = dt.datetime(2000 + int(d[:2]), int(d[2:4]), int(d[4:6])).timestamp()
                    ts = jours[d] + int(h[:2]) * 3600 + int(h[2:4]) * 60 + int(h[4:6])
                    niveau = morceaux[3]
                except (IndexError, ValueError):
                    ts, niveau = None, ""
                for b in dict.fromkeys(ids):
                    if b not in ordre:
                        ordre[b] = len(ordre)
                        n_warn[b] = 0
                        n_error[b] = 0
                    if ts is not None:
                        if b not in debut:
                            debut[b] = ts
                        fin[b] = ts
                    if niveau == "WARN":
                        n_warn[b] += 1
                    elif niveau in ("ERROR", "FATAL"):
                        n_error[b] += 1
                if seqs is not None:
                    contenu = morceaux[5].rstrip("\n") if len(morceaux) > 5 else ligne
                    for num, rx in regex:
                        if rx.match(contenu):
                            for b in dict.fromkeys(ids):
                                seqs.setdefault(b, []).append(num)
                            break
            if i and i % 1_000_000 == 0:
                info(f"  {i // 1_000_000} M lignes lues ({time.time() - t0:.0f} s)")
    info(f"{len(ordre)} blocs trouvés dans le log")
    duree = {b: int(fin[b] - debut[b]) for b in debut}
    return ordre, n_warn, n_error, duree, seqs


def md5(chemin):
    h = hashlib.md5()
    with open(chemin, "rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Prépare les fichiers communs HDFS_v1")
    defaut = DATA / "HDFS_v1" if (DATA / "HDFS_v1").exists() else DATA / "raw" / "HDFS_v1"
    ap.add_argument("--source", default=str(defaut),
                    help="dossier HDFS_v1 extrait (contient HDFS.log et preprocessed/)")
    args = ap.parse_args()
    src = Path(args.source)
    if not src.exists():
        sys.exit(f"dossier introuvable : {src}")
    DATA.mkdir(parents=True, exist_ok=True)

    f_traces = trouver(src, "Event_traces.csv")
    f_labels = trouver(src, "anomaly_label.csv")
    f_log = trouver(src, "HDFS.log")
    f_tpl = trouver(src, "*templates*.csv")
    info(f"traces={f_traces}\n          labels={f_labels}\n          log={f_log}\n          templates={f_tpl}")
    if f_labels is None or f_log is None:
        sys.exit("il faut au moins anomaly_label.csv et HDFS.log")

    tpl = lire_templates(f_tpl) if f_tpl else None
    if tpl is not None:
        tpl.to_csv(DATA / "templates.csv", index=False)
        info(f"{len(tpl)} templates écrits dans data/templates.csv")
    labels = lire_labels(f_labels)

    if f_traces is not None:
        df = lire_traces(f_traces)
        ordre, n_warn, n_error, duree, _ = parcourir_log(f_log)
    elif tpl is not None:
        info("Event_traces.csv absent : séquences reconstruites depuis HDFS.log")
        ordre, n_warn, n_error, duree, seqs = parcourir_log(f_log, tpl)
        df = pd.DataFrame({"block_id": list(seqs.keys()), "events": list(seqs.values())})
    else:
        sys.exit("ni Event_traces.csv, ni fichier de templates")

    # Label, nettoyage, ordre d'apparition, découpage 70/15/15
    df["label"] = df["block_id"].map(labels)
    manquants = int(df["label"].isna().sum())
    if manquants:
        info(f"{manquants} blocs sans label retirés")
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    df = df[df["events"].apply(len) > 0]
    rang = df["block_id"].map(ordre)
    info(f"ordre d'apparition connu pour {int(rang.notna().sum())} blocs sur {len(df)}")
    df["rang"] = rang.fillna(pd.Series(len(ordre) + np.arange(len(df)), index=df.index))
    df = df.sort_values("rang", kind="stable").drop(columns="rang").reset_index(drop=True)

    n = len(df)
    n_train, n_val = int(0.70 * n), int(0.15 * n)
    split = np.array(["test"] * n, dtype=object)
    split[:n_train] = "train"
    split[n_train:n_train + n_val] = "val"
    df["split"] = split

    tous = sorted({e for s in df["events"] for e in s})
    if tous[0] < 1 or tous[-1] > N_EVENEMENTS:
        sys.exit(f"numéros d'événements hors de 1 à {N_EVENEMENTS} : {tous[0]} à {tous[-1]}")

    # sequences.parquet
    seq = df[["block_id", "label", "split", "events"]]
    seq.to_parquet(DATA / "sequences.parquet", index=False)
    info("data/sequences.parquet écrit")

    # counts.parquet
    M = np.zeros((n, N_EVENEMENTS), dtype=np.int32)
    for i, s in enumerate(df["events"]):
        M[i] = np.bincount(s, minlength=N_EVENEMENTS + 1)[1:]
    counts = pd.DataFrame(M, columns=[f"E{k}" for k in range(1, N_EVENEMENTS + 1)])
    counts.insert(0, "split", df["split"].values)
    counts.insert(0, "label", df["label"].values)
    counts.insert(0, "block_id", df["block_id"].values)
    counts["n_warn"] = df["block_id"].map(n_warn).fillna(0).astype(int).values
    counts["n_error"] = df["block_id"].map(n_error).fillna(0).astype(int).values
    counts["length"] = df["events"].apply(len).astype(int).values
    counts["n_unique"] = df["events"].apply(lambda s: len(set(s))).astype(int).values
    counts["duration_s"] = df["block_id"].map(duree).fillna(0).astype(int).values
    counts.to_parquet(DATA / "counts.parquet", index=False)
    info("data/counts.parquet écrit")

    # MANIFEST
    lignes = [f"Fichiers communs LogSentinel, HDFS_v1, générés le {dt.datetime.now():%Y-%m-%d %H:%M}",
              f"Source : {src}", "",
              "Découpage : blocs rangés par première apparition dans HDFS.log, puis 70 % train, 15 % val, 15 % test", ""]
    for s in ("train", "val", "test"):
        p = df[df["split"] == s]
        lignes.append(f"{s:5s} : {len(p):7d} blocs, {int(p['label'].sum()):6d} anomalies ({p['label'].mean() * 100:.2f} %)")
    lignes.append(f"total : {n} blocs, {int(df['label'].sum())} anomalies")
    lg = counts["length"]
    lignes += [f"longueurs : min {lg.min()}, médiane {int(lg.median())}, max {lg.max()}",
               f"événements présents : {len(tous)} (de E{tous[0]} à E{tous[-1]})",
               f"blocs avec au moins un WARN : {int((counts['n_warn'] > 0).sum())}, avec ERROR : {int((counts['n_error'] > 0).sum())}",
               "", "Empreintes MD5 (doivent être identiques chez tout le monde) :"]
    for nom in ("sequences.parquet", "counts.parquet", "templates.csv"):
        if (DATA / nom).exists():
            lignes.append(f"  {nom:18s} {md5(DATA / nom)}")
    (DATA / "MANIFEST.txt").write_text("\n".join(lignes) + "\n", encoding="utf-8")
    print("\n".join(lignes))
    info("terminé")


if __name__ == "__main__":
    main()
