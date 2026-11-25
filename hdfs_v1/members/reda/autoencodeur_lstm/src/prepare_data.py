# Étape 1 : télécharger HDFS_v1, construire data/sequences_provisoire.parquet
# (block_id, label, split, events), au même format que le futur fichier commun de Mouad.
# Si data/sequences.parquet (le fichier officiel) existe déjà, cette étape ne fait rien.
import re
import sys
import time
import zipfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from common import DATA, RAW, journal, chemin_sequences, garder_pc_eveille

URL = "https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1"
ZIP = RAW / "HDFS_v1.zip"
DOSSIER = RAW / "HDFS_v1"
BLK = re.compile(r"blk_-?\d+")
log = journal("1_preparation")


def telecharger():
    if ZIP.exists() and zipfile.is_zipfile(ZIP):
        log.info(f"zip déjà présent : {ZIP}")
        return
    log.info("téléchargement de HDFS_v1.zip (187 Mo) depuis Zenodo...")
    log.info("(si cela échoue : télécharger le fichier à la main depuis zenodo.org/records/8196385 "
             "et le poser dans data/raw/HDFS_v1.zip)")
    part = ZIP.with_suffix(".part")
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(part, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        lu, palier = 0, 0
        while True:
            bloc = r.read(1 << 20)
            if not bloc:
                break
            f.write(bloc)
            lu += len(bloc)
            if total and lu * 10 // total > palier:
                palier = lu * 10 // total
                log.info(f"  {palier * 10} %")
    part.replace(ZIP)
    log.info("téléchargement terminé")


def extraire():
    if DOSSIER.exists() and any(DOSSIER.rglob("*")):
        log.info("zip déjà extrait")
        return
    log.info("extraction du zip...")
    try:
        with zipfile.ZipFile(ZIP) as z:
            z.extractall(DOSSIER)
    except zipfile.BadZipFile:
        ZIP.unlink(missing_ok=True)
        raise RuntimeError("zip abîmé (téléchargement incomplet) : supprimé, relancer run_all.bat")
    log.info("extraction terminée")


def trouver(motif):
    res = sorted(DOSSIER.rglob(motif), key=lambda p: len(str(p)))
    return res[0] if res else None


def colonne_blk(df):
    for c in df.columns:
        v = df[c].astype(str).head(50)
        if v.str.match(r"^blk_-?\d+$").mean() > 0.9:
            return c
    raise ValueError(f"aucune colonne d'identifiants blk_ trouvée dans {list(df.columns)}")


def lire_labels(chemin):
    lab = pd.read_csv(chemin)
    c_id = colonne_blk(lab)
    c_lab = None
    for c in lab.columns:
        if c == c_id:
            continue
        v = set(lab[c].astype(str).str.lower().head(2000).unique())
        if v & {"anomaly", "normal", "fail", "success"}:
            c_lab = c
            break
    if c_lab is None:
        raise ValueError(f"colonne de label introuvable dans {list(lab.columns)}")
    y = lab[c_lab].astype(str).str.lower().isin(["anomaly", "fail"]).astype(int)
    log.info(f"labels : {len(lab)} blocs, {int(y.sum())} anomalies (colonnes {c_id}, {c_lab})")
    return dict(zip(lab[c_id].astype(str), y))


def lire_traces(chemin):
    log.info(f"lecture de {chemin.name}...")
    tr = pd.read_csv(chemin)
    log.info(f"colonnes : {list(tr.columns)}")
    c_id = colonne_blk(tr)
    c_ev = None
    for c in tr.columns:
        v = tr[c].astype(str).head(50)
        if v.str.contains(r"E\d+").mean() > 0.9 and c != c_id:
            c_ev = c
            break
    if c_ev is None:
        raise ValueError("colonne des séquences introuvable")
    num = re.compile(r"E(\d+)")
    events = tr[c_ev].astype(str).apply(lambda s: [int(e) for e in num.findall(s)])
    log.info(f"séquences lues dans la colonne {c_ev} : {len(tr)} blocs")
    return pd.DataFrame({"block_id": tr[c_id].astype(str), "events": events})


def lire_templates(chemin):
    t = pd.read_csv(chemin)
    c_id = [c for c in t.columns if t[c].astype(str).str.match(r"^E\d+$").mean() > 0.9][0]
    c_tx = [c for c in t.columns if c != c_id and not pd.api.types.is_numeric_dtype(t[c])][0]
    t = pd.DataFrame({"num": t[c_id].str[1:].astype(int), "texte": t[c_tx].astype(str)})
    t.to_csv(DATA / "templates.csv", index=False)
    log.info(f"{len(t)} templates enregistrés dans data/templates.csv")
    return t


def parcourir_log(chemin, tpl=None):
    """Lit HDFS.log une fois : ordre de première apparition de chaque bloc et,
    si tpl est donné (cas de secours), les séquences d'événements."""
    log.info(f"lecture de {chemin.name} (11 millions de lignes, quelques minutes)...")
    ordre = {}
    seqs = {} if tpl is not None else None
    regex = None
    if tpl is not None:
        regex = []
        for num, texte in zip(tpl["num"], tpl["texte"]):
            motif = re.escape(texte.strip())
            motif = motif.replace(re.escape("[*]"), ".*?").replace(re.escape("<*>"), ".*?")
            regex.append((num, re.compile("^" + motif + "$")))
        regex.sort(key=lambda r: -len(r[1].pattern))
    entete = re.compile(r"^\d+ \d+ \d+ \w+ [^:]+: (.*)$")
    t0 = time.time()
    with open(chemin, "r", encoding="utf-8", errors="replace") as f:
        for i, ligne in enumerate(f):
            ids = BLK.findall(ligne)
            for b in ids:
                if b not in ordre:
                    ordre[b] = len(ordre)
            if seqs is not None and ids:
                m = entete.match(ligne.rstrip("\n"))
                contenu = m.group(1) if m else ligne
                for num, rx in regex:
                    if rx.match(contenu):
                        for b in dict.fromkeys(ids):
                            seqs.setdefault(b, []).append(num)
                        break
            if i and i % 1_000_000 == 0:
                log.info(f"  {i // 1_000_000} M lignes lues ({time.time() - t0:.0f} s)")
    log.info(f"{len(ordre)} blocs trouvés dans le log")
    return ordre, seqs


def main():
    garder_pc_eveille()
    sortie, officiel = chemin_sequences()
    if officiel:
        log.info("data/sequences.parquet (fichier officiel) trouvé : rien à préparer")
        return
    if sortie.exists():
        log.info(f"{sortie.name} existe déjà : étape sautée")
        return

    telecharger()
    extraire()
    f_traces = trouver("Event_traces.csv")
    f_labels = trouver("anomaly_label.csv")
    f_log = trouver("HDFS.log")
    f_tpl = trouver("*templates*.csv")
    log.info(f"fichiers : traces={f_traces}, labels={f_labels}, log={f_log}, templates={f_tpl}")
    if f_labels is None:
        raise FileNotFoundError("anomaly_label.csv introuvable dans le zip")
    tpl = None
    if f_tpl:
        try:
            tpl = lire_templates(f_tpl)
        except Exception as e:
            log.info(f"templates non lus ({e}) : on continue sans le texte des événements")
    labels = lire_labels(f_labels)

    if f_traces is not None:
        df = lire_traces(f_traces)
        ordre = parcourir_log(f_log)[0] if f_log else None
    elif f_log is not None and tpl is not None:
        log.info("Event_traces.csv absent : séquences reconstruites depuis HDFS.log (plus long)")
        ordre, seqs = parcourir_log(f_log, tpl)
        df = pd.DataFrame({"block_id": list(seqs.keys()), "events": list(seqs.values())})
    else:
        raise FileNotFoundError("ni Event_traces.csv, ni HDFS.log avec templates")

    df["label"] = df["block_id"].map(labels)
    manquants = int(df["label"].isna().sum())
    if manquants:
        log.info(f"{manquants} blocs sans label retirés")
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    df = df[df["events"].apply(len) > 0]

    if ordre:
        rang = df["block_id"].map(ordre)
        log.info(f"ordre chronologique connu pour {int(rang.notna().sum())} blocs sur {len(df)}")
        df["rang"] = rang.fillna(pd.Series(len(ordre) + np.arange(len(df)), index=df.index))
        df = df.sort_values("rang", kind="stable").drop(columns="rang")
    else:
        log.info("HDFS.log absent : ordre des lignes du fichier utilisé (approximation)")
    df = df.reset_index(drop=True)

    n = len(df)
    n_train, n_val = int(0.70 * n), int(0.15 * n)
    split = np.array(["test"] * n, dtype=object)
    split[:n_train] = "train"
    split[n_train:n_train + n_val] = "val"
    df["split"] = split
    df = df[["block_id", "label", "split", "events"]]
    df.to_parquet(sortie, index=False)

    for s in ("train", "val", "test"):
        p = df[df["split"] == s]
        log.info(f"{s:5s} : {len(p):7d} blocs, {int(p['label'].sum()):6d} anomalies ({p['label'].mean() * 100:.2f} %)")
    lg = df["events"].apply(len)
    log.info(f"longueurs : min {lg.min()}, médiane {int(lg.median())}, max {lg.max()}")
    tous = sorted({e for s in df["events"] for e in s})
    log.info(f"numéros d'événements : {tous[0]} à {tous[-1]} ({len(tous)} différents)")
    if tous[-1] >= 31:
        raise ValueError("un numéro d'événement dépasse 30 : à vérifier avant d'entraîner")
    log.info(f"fichier écrit : {sortie}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.exception(f"ERREUR : {e}")
        sys.exit(1)
