# Prépare les fichiers communs de l'étape 2 à partir de HDFS_v3 TraceBench (Loghub) :
#   hdfs_v3/data/traces.parquet     une ligne par trace anormale : type de panne, partie, caractéristiques résumées
#   hdfs_v3/data/etapes.parquet     une ligne par trace : la suite de ses étapes, dans l'ordre
#   hdfs_v3/data/aretes.parquet     les liens entre étapes (le graphe de chaque trace)
#   hdfs_v3/data/vocabulaire.json   correspondance entre numéros et noms (pannes, opérations, erreurs...)
#   hdfs_v3/data/MANIFEST.txt       chiffres clés, choix faits et empreintes MD5
#
# Utilisation (depuis le dossier hdfs_v3) :
#   python scripts/preparer_donnees.py
# Le zip est lu directement, sans l'extraire : par défaut data/HDFS_v3_TraceBench.zip à la racine du dépôt.
# Autre emplacement :
#   python scripts/preparer_donnees.py --source "C:\chemin\vers\HDFS_v3_TraceBench.zip"
#
# Le script garde un résultat intermédiaire par dossier de TraceBench (data/HDFS_v3_cache à la racine du
# dépôt) : s'il est interrompu, il reprend là où il s'était arrêté. Le cache est supprimé à la fin.
import io
import re
import sys
import json
import time
import shutil
import hashlib
import zipfile
import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RACINE = Path(__file__).resolve().parent.parent
DEPOT = RACINE.parent
DATA = RACINE / "data"
GRAINE = 42
COLONNES_EVT = ["TaskID", "TID", "OpName", "StartTime", "EndTime", "HostAddress", "HostName", "Agent",
                "Description"]
CHIFFRES = re.compile(r"[0-9][^a-z/]*")   # même nettoyage que le script de Loghub
ROLES = ["client", "namenode", "datanode"]
TYPES_ARETE = ["appel", "suite"]


def info(msg):
    print(time.strftime("%H:%M:%S"), " ", msg, flush=True)


def slug(texte):
    s = re.sub(r"[^a-z0-9]+", "_", texte.lower()).strip("_")
    return s or "vide"


# Choix des dossiers -----------------------------------------------------------------------------------

def panne_du_dossier(nom):
    """AN_Proc_killDN_w_05FDN_... -> ("Proc_killDN", "Proc", "w"). None si le dossier n'est pas gardé."""
    morceaux = nom.split("_")
    if morceaux[0] != "AN":
        return None, "pas un dossier de panne simple (NM = normal, COM = pannes combinées)"
    if "_00FDN_" in nom:
        return None, "0 DataNode en panne : aucune panne injectée"
    if re.search(r"_slowHDFS_[a-z]+_0+ms_", nom):
        return None, "retard de 0 ms : aucune panne injectée"
    return (f"{morceaux[1]}_{morceaux[2]}", morceaux[1], morceaux[3]), ""


def lire_evenements(brut):
    """event.csv : la colonne Description contient des virgules sans guillemets, on coupe à la main."""
    lignes = brut.decode("utf-8", "replace").splitlines()[1:]
    morceaux = [l.split(",", 8) for l in lignes if l]
    morceaux = [m for m in morceaux if len(m) == 9]
    return pd.DataFrame(morceaux, columns=COLONNES_EVT)


def role(nom_machine):
    n = str(nom_machine).lower()
    if "namenode" in n:
        return "namenode"
    if n.startswith("client"):
        return "client"
    return "datanode"


def traiter_dossier(z, nom, panne, categorie, charge):
    ev = lire_evenements(z.read(f"tracebench/{nom}/event.csv"))
    ev["ordre"] = np.arange(len(ev))
    desc = ev["Description"].fillna("").str.lower()
    reussi = desc.str.contains("success:", regex=False) | desc.str.contains("a user task", regex=False)
    ev["echec"] = (~reussi).astype(np.int8)
    anormales = ev.loc[ev["echec"] == 1, "TaskID"].unique()
    ev = ev[ev["TaskID"].isin(anormales)].copy()
    if ev.empty:
        return None
    desc = ev["Description"].fillna("").str.lower()
    ev["erreur"] = np.where(ev["echec"] == 1,
                            desc.map(lambda s: re.sub(r"\s+", " ", CHIFFRES.sub("", s)).strip()[:100]), "")
    debut = pd.to_numeric(ev["StartTime"], errors="coerce")
    fin = pd.to_numeric(ev["EndTime"], errors="coerce")
    ev["duree_ms"] = ((fin - debut) / 1e6).clip(lower=0).fillna(0.0).astype("float32")
    ev["role"] = ev["HostName"].map(role)
    ev = ev.sort_values(["TaskID", "ordre"])
    ev["pos"] = ev.groupby("TaskID").cumcount().astype(np.int32)
    ev["machine"] = ev.groupby("TaskID")["HostName"].transform(lambda s: pd.factorize(s)[0]).astype(np.int16)

    # Arêtes : "appel" = étape parente vers la première étape du fil appelé ; "suite" = étapes qui se suivent
    # dans un même fil (même TID).
    ar = pd.read_csv(io.BytesIO(z.read(f"tracebench/{nom}/edge.csv")), dtype=str)
    ar = ar[ar["TaskID"].isin(anormales)]
    cle = ev.drop_duplicates(["TaskID", "TID", "StartTime"])[["TaskID", "TID", "StartTime", "pos"]]
    src = ar.merge(cle, left_on=["TaskID", "FatherTID", "FatherStartTime"],
                   right_on=["TaskID", "TID", "StartTime"], how="inner")[["TaskID", "ChildTID", "pos"]]
    premier = ev.groupby(["TaskID", "TID"])["pos"].min().rename("dst").reset_index()
    appels = src.merge(premier, left_on=["TaskID", "ChildTID"], right_on=["TaskID", "TID"], how="inner")
    appels = appels.rename(columns={"pos": "src"})[["TaskID", "src", "dst"]]
    appels = appels[appels["src"] != appels["dst"]]
    appels["type"] = 0
    suite = ev[["TaskID", "TID", "pos"]].copy()
    suite["dst"] = suite.groupby(["TaskID", "TID"])["pos"].shift(-1)
    suite = suite.dropna(subset=["dst"]).rename(columns={"pos": "src"})[["TaskID", "src", "dst"]]
    suite["type"] = 1
    aretes = pd.concat([appels, suite], ignore_index=True).drop_duplicates()
    aretes[["src", "dst"]] = aretes[["src", "dst"]].astype(np.int32)
    aretes["type"] = aretes["type"].astype(np.int8)

    tr = pd.read_csv(io.BytesIO(z.read(f"tracebench/{nom}/trace.csv")), dtype=str)
    tr = tr[tr["TaskID"].isin(anormales)].drop_duplicates("TaskID")
    d_tr = (pd.to_numeric(tr["EndTime"], errors="coerce") - pd.to_numeric(tr["StartTime"], errors="coerce")) / 1e6
    traces = pd.DataFrame({"TaskID": tr["TaskID"].values, "titre": tr["Title"].fillna("").values,
                           "duree_trace_ms": d_tr.clip(lower=0).values})
    fils = ev.groupby("TaskID")["TID"].nunique().rename("n_fils").reset_index()
    traces = pd.DataFrame({"TaskID": anormales}).merge(traces, on="TaskID", how="left").merge(fils, on="TaskID")
    traces["scenario"], traces["panne"], traces["categorie"], traces["charge"] = nom, panne, categorie, charge

    etapes = ev[["TaskID", "pos", "OpName", "Agent", "role", "machine", "duree_ms", "echec", "erreur"]]
    return {"traces": traces, "etapes": etapes.reset_index(drop=True), "aretes": aretes}


# Assemblage ---------------------------------------------------------------------------------------------

def afficher(chemin):
    """Chemin relatif au dépôt quand c'est possible, pour un MANIFEST identique chez tout le monde."""
    try:
        return Path(chemin).resolve().relative_to(DEPOT.resolve()).as_posix()
    except ValueError:
        return Path(chemin).name


def md5(chemin):
    h = hashlib.md5()
    with open(chemin, "rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


def assembler(morceaux, exclus, min_traces, source):
    traces = pd.concat([m["traces"] for m in morceaux], ignore_index=True)
    etapes = pd.concat([m["etapes"] for m in morceaux], ignore_index=True)
    aretes = pd.concat([m["aretes"] for m in morceaux], ignore_index=True)

    # Identifiant unique de trace
    doublons = traces["TaskID"].duplicated(keep=False)
    if doublons.any():
        sys.exit(f"{int(doublons.sum())} TaskID présents dans plusieurs dossiers : identifiant à revoir")
    traces = traces.rename(columns={"TaskID": "trace_id"})
    etapes = etapes.rename(columns={"TaskID": "trace_id"})
    aretes = aretes.rename(columns={"TaskID": "trace_id"})

    # Types de panne gardés
    compte = traces["panne"].value_counts()
    petits = compte[compte < min_traces]
    for p, n in petits.items():
        exclus.append((p, f"seulement {n} traces anormales (minimum {min_traces})"))
    gardes = sorted(compte[compte >= min_traces].index)
    traces = traces[traces["panne"].isin(gardes)].copy()
    etapes = etapes[etapes["trace_id"].isin(traces["trace_id"])]
    aretes = aretes[aretes["trace_id"].isin(traces["trace_id"])]
    traces["label"] = traces["panne"].map({p: i for i, p in enumerate(gardes)}).astype(np.int16)

    # Même message d'erreur pour un bloc au numéro négatif ou positif (blk_-123 et blk_123)
    etapes = etapes.assign(erreur=etapes["erreur"].str.replace("blk_-", "blk_", regex=False))

    # Vocabulaires (0 réservé : remplissage pour les opérations et agents, "pas d'erreur" pour les erreurs)
    operations = sorted(etapes["OpName"].unique())
    agents = sorted(etapes["Agent"].unique())
    freq_err = etapes.loc[etapes["echec"] == 1, "erreur"].value_counts()
    frequentes = list(freq_err[freq_err >= 20].index)
    erreurs = ["pas d'erreur"] + frequentes + ["autre erreur"]
    num_err = {e: i + 1 for i, e in enumerate(frequentes)}
    autre = len(erreurs) - 1
    etapes = etapes.assign(
        op=etapes["OpName"].map({o: i + 1 for i, o in enumerate(operations)}).astype(np.int16),
        agent=etapes["Agent"].map({a: i + 1 for i, a in enumerate(agents)}).astype(np.int8),
        role_n=etapes["role"].map({r: i for i, r in enumerate(ROLES)}).astype(np.int8),
        err=np.where(etapes["echec"] == 1, etapes["erreur"].map(num_err).fillna(autre), 0).astype(np.int16))

    # Découpage 70 / 15 / 15 stratifié par type de panne
    ids, y = traces["trace_id"].values, traces["label"].values
    i_train, i_reste, _, y_reste = train_test_split(ids, y, test_size=0.30, stratify=y, random_state=GRAINE)
    i_val, i_test = train_test_split(i_reste, test_size=0.50, stratify=y_reste, random_state=GRAINE)
    partie = {**{i: "train" for i in i_train}, **{i: "val" for i in i_val}, **{i: "test" for i in i_test}}
    traces["split"] = traces["trace_id"].map(partie)

    # Caractéristiques résumées par trace
    info("calcul des caractéristiques par trace")
    g = etapes.groupby("trace_id")
    car = pd.DataFrame({
        "n_etapes": g.size(),
        "n_machines": g["machine"].nunique(),
        "n_datanodes": etapes[etapes["role"] == "datanode"].groupby("trace_id")["machine"].nunique(),
        "n_echecs": g["echec"].sum(),
        "n_types_erreur": etapes[etapes["echec"] == 1].groupby("trace_id")["err"].nunique(),
        "duree_moy_ms": g["duree_ms"].mean(),
        "duree_max_ms": g["duree_ms"].max(),
        "duree_totale_ms": g["duree_ms"].sum(),
        "pos_premier_echec": etapes[etapes["echec"] == 1].groupby("trace_id")["pos"].min(),
    })
    car["part_echecs"] = car["n_echecs"] / car["n_etapes"]
    car["pos_premier_echec"] = car["pos_premier_echec"] / car["n_etapes"]
    car["n_aretes_appel"] = aretes[aretes["type"] == 0].groupby("trace_id").size()
    noms_op = {i + 1: f"op_{slug(o.replace('OP:', ''))}" for i, o in enumerate(operations)}
    ops = pd.crosstab(etapes["trace_id"], etapes["op"]).rename(columns=noms_op)
    ech = etapes[etapes["echec"] == 1]
    ech_role = pd.crosstab(ech["trace_id"], ech["role"]).reindex(columns=ROLES, fill_value=0)
    ech_role.columns = [f"echecs_{r}" for r in ech_role.columns]
    err = pd.crosstab(ech["trace_id"], ech["err"])
    err = err.reindex(columns=range(1, len(erreurs)), fill_value=0)
    err.columns = [f"err_{i:02d}" for i in err.columns]
    car = car.join([ops, ech_role, err]).fillna(0)
    car["ecriture"] = (traces.set_index("trace_id")["titre"] == "fs -copyFromLocal").astype(np.int8)

    meta = ["trace_id", "label", "panne", "categorie", "charge", "scenario", "titre", "split"]
    tab = traces.set_index("trace_id")[[c for c in meta if c != "trace_id"]].join(car)
    tab["duree_trace_ms"] = traces.set_index("trace_id")["duree_trace_ms"].fillna(tab["duree_totale_ms"])
    tab["n_fils"] = traces.set_index("trace_id")["n_fils"]
    colonnes_car = [c for c in tab.columns if c not in meta]
    for c in colonnes_car:
        if c.startswith(("op_", "err_", "echecs_", "n_")):
            tab[c] = tab[c].astype(np.int32)
    tab = tab.reset_index().sort_values(["split", "label", "trace_id"]).reset_index(drop=True)

    # Suites d'étapes (même ordre de traces que traces.parquet)
    info("construction des suites d'étapes")
    etapes = etapes.sort_values(["trace_id", "pos"])
    g = etapes.groupby("trace_id", sort=False)
    suites = pd.DataFrame({
        "ops": g["op"].agg(list), "agents": g["agent"].agg(list), "roles": g["role_n"].agg(list),
        "machines": g["machine"].agg(list), "durees_ms": g["duree_ms"].agg(list), "erreurs": g["err"].agg(list),
    })
    suites = tab[["trace_id", "label", "split"]].set_index("trace_id").join(suites).reset_index()
    aretes = aretes.merge(tab[["trace_id", "split"]], on="trace_id")[["trace_id", "split", "src", "dst", "type"]]
    aretes = aretes.sort_values(["trace_id", "type", "src", "dst"]).reset_index(drop=True)

    DATA.mkdir(parents=True, exist_ok=True)
    tab.to_parquet(DATA / "traces.parquet", index=False)
    suites.to_parquet(DATA / "etapes.parquet", index=False)
    aretes.to_parquet(DATA / "aretes.parquet", index=False)
    vocab = {"pannes": gardes, "operations": ["(remplissage)"] + operations, "agents": ["(remplissage)"] + agents,
             "roles": ROLES, "erreurs": erreurs, "types_arete": TYPES_ARETE, "colonnes_caract": colonnes_car}
    (DATA / "vocabulaire.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=1), encoding="utf-8")

    # MANIFEST
    l = [f"Fichiers communs LogSentinel, HDFS_v3 TraceBench, générés le {dt.datetime.now():%Y-%m-%d %H:%M}",
         f"Source : {afficher(source)}", "",
         "Une trace est gardée si au moins une de ses étapes a échoué (règle du script de Loghub : la description",
         "ne contient ni « success: » ni « a user task »). Son type de panne est celui du dossier où elle se trouve.",
         "Découpage : 70 % train, 15 % val, 15 % test, stratifié par type de panne, graine 42.", ""]
    l.append(f"{'type de panne':22s} {'label':>5s} {'train':>6s} {'val':>6s} {'test':>6s} {'total':>6s}")
    for i, p in enumerate(gardes):
        s = tab[tab["label"] == i]["split"].value_counts()
        l.append(f"{p:22s} {i:5d} {s.get('train', 0):6d} {s.get('val', 0):6d} {s.get('test', 0):6d} "
                 f"{int(s.sum()):6d}")
    s = tab["split"].value_counts()
    l.append(f"{'total':22s} {'':5s} {s.get('train', 0):6d} {s.get('val', 0):6d} {s.get('test', 0):6d} {len(tab):6d}")
    l += ["", f"étapes par trace : min {int(tab['n_etapes'].min())}, médiane {int(tab['n_etapes'].median())}, "
              f"max {int(tab['n_etapes'].max())}",
          f"opérations différentes : {len(operations)}, types d'erreur gardés : {len(frequentes)} (+ autre erreur)",
          f"caractéristiques dans traces.parquet : {len(colonnes_car)}", "", "Exclus :"]
    par_raison = {}
    for quoi, pourquoi in exclus:
        court = "_".join(quoi.split("_")[:3])
        par_raison.setdefault(pourquoi, {}).setdefault(court, 0)
        par_raison[pourquoi][court] += 1
    for pourquoi, noms in par_raison.items():
        l.append(f"  {pourquoi} :")
        l.append("    " + ", ".join(f"{n} ({k})" if k > 1 else n for n, k in sorted(noms.items())))
    l += ["", "Empreintes MD5 (doivent être identiques chez tout le monde) :"]
    for nom in ("traces.parquet", "etapes.parquet", "aretes.parquet", "vocabulaire.json"):
        l.append(f"  {nom:18s} {md5(DATA / nom)}")
    (DATA / "MANIFEST.txt").write_text("\n".join(l) + "\n", encoding="utf-8")
    print("\n".join(l))


def main():
    ap = argparse.ArgumentParser(description="Fichiers communs de l'étape 2 (HDFS_v3 TraceBench)")
    ap.add_argument("--source", default=str(DEPOT / "data" / "HDFS_v3_TraceBench.zip"))
    ap.add_argument("--cache", default=str(DEPOT / "data" / "HDFS_v3_cache"))
    ap.add_argument("--min-traces", type=int, default=30, help="minimum de traces pour garder un type de panne")
    ap.add_argument("--max-secondes", type=float, default=None, help="s'arrêter après ce temps (reprise possible)")
    ap.add_argument("--garder-cache", action="store_true")
    a = ap.parse_args()

    t0 = time.time()
    z = zipfile.ZipFile(a.source)
    dossiers = sorted({n.split("/")[1] for n in z.namelist() if n.startswith("tracebench/") and n.count("/") >= 2})
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    exclus, a_faire = [], []
    for nom in dossiers:
        res, pourquoi = panne_du_dossier(nom)
        if res is None:
            exclus.append((nom, pourquoi))
        else:
            a_faire.append((nom, *res))
    info(f"{len(dossiers)} dossiers dans le zip, {len(a_faire)} dossiers de panne à lire")

    morceaux = []
    for k, (nom, panne, categorie, charge) in enumerate(a_faire, 1):
        f = cache / f"{nom}.pkl"
        if not f.exists():
            if a.max_secondes and time.time() - t0 > a.max_secondes:
                info(f"arrêt après {a.max_secondes:.0f} s, {k - 1}/{len(a_faire)} dossiers prêts : relancer pour continuer")
                return
            m = traiter_dossier(z, nom, panne, categorie, charge)
            pd.to_pickle(m, f)
            if k % 20 == 0:
                info(f"{k}/{len(a_faire)} dossiers lus")
        m = pd.read_pickle(f)
        if m is not None:
            morceaux.append(m)
        else:
            exclus.append((nom, "aucune trace avec une étape en échec"))
    info("tous les dossiers sont lus, assemblage")
    assembler(morceaux, exclus, a.min_traces, a.source)
    if not a.garder_cache:
        shutil.rmtree(cache, ignore_errors=True)
    info(f"terminé en {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
