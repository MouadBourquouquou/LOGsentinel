# Outils partagés par les trois scripts : chemins, graine, journal, modèle, entraînement, scores.
import os
import sys
import json
import time
import copy
import random
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
LOGS = ROOT / "logs"
for d in (DATA, RAW, RESULTS, FIGS, LOGS):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
N_CLASSES = 31            # 0 = remplissage, 1 à 30 = événements
BATCH = 256
LR = 0.001
QUANTILE_SEUIL = 0.995

# Mode rapide (pour vérifier que tout tourne en quelques minutes) : LOGSENTINEL_RAPIDE=1
RAPIDE = os.environ.get("LOGSENTINEL_RAPIDE") == "1"


def fixer_graine(graine=SEED):
    random.seed(graine)
    np.random.seed(graine)
    torch.manual_seed(graine)


def garder_pc_eveille():
    # Empêche Windows de se mettre en veille tant que le script tourne (pas besoin d'être admin).
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        except Exception:
            pass


def journal(nom):
    log = logging.getLogger(nom)
    log.setLevel(logging.INFO)
    if not log.handlers:
        fmt = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")
        fh = logging.FileHandler(LOGS / f"{nom}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        log.addHandler(fh)
        log.addHandler(sh)
    return log


def chemin_sequences():
    # Le fichier officiel de Mouad est prioritaire s'il est posé dans data/
    officiel = DATA / "sequences.parquet"
    if officiel.exists():
        return officiel, True
    return DATA / "sequences_provisoire.parquet", False


def charger_sequences():
    chemin, officiel = chemin_sequences()
    df = pd.read_parquet(chemin)
    df["events"] = df["events"].apply(lambda s: [int(e) for e in s])
    df["label"] = df["label"].astype(int)
    return df, officiel


def pad_or_cut(seq, L):
    seq = list(seq)[:L]
    return seq + [0] * (L - len(seq))


def to_tensor(partie, L):
    X = np.stack([pad_or_cut(s, L) for s in partie["events"]])
    return torch.tensor(X, dtype=torch.long)


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_classes=N_CLASSES, emb_dim=32, hidden=64, latent=32):
        super().__init__()
        self.embedding = nn.Embedding(n_classes, emb_dim, padding_idx=0)
        self.encoder = nn.LSTM(emb_dim, hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent)
        self.decoder = nn.LSTM(latent, hidden, batch_first=True)
        self.output = nn.Linear(hidden, n_classes)

    def forward(self, x):                                   # x : (B, L)
        e = self.embedding(x)                               # (B, L, 32)
        _, (h, _) = self.encoder(e)                         # h : (1, B, hidden)
        z = self.to_latent(h[-1])                           # (B, latent)
        z_rep = z.unsqueeze(1).repeat(1, x.size(1), 1)      # (B, L, latent)
        d, _ = self.decoder(z_rep)                          # (B, L, hidden)
        return self.output(d)                               # (B, L, 31)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def perte_moyenne(model, loader, criterion):
    model.eval()
    total, nb = 0.0, 0
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(DEVICE)
            logits = model(x)
            loss = criterion(logits.reshape(-1, N_CLASSES), x.reshape(-1))
            total += loss.item() * x.size(0)
            nb += x.size(0)
    return total / nb


def entrainer(model, X_train, X_val_normal, max_epoques, patience, log, prefixe=""):
    """Entraîne avec arrêt précoce sur val normal. Renvoie l'historique et le temps total."""
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(TensorDataset(X_train), batch_size=BATCH, shuffle=True, generator=g)
    val_loader = DataLoader(TensorDataset(X_val_normal), batch_size=1024, shuffle=False)

    meilleure, attente, meilleur_etat = float("inf"), 0, copy.deepcopy(model.state_dict())
    historique = []
    t0 = time.time()
    for epoque in range(1, max_epoques + 1):
        debut = time.time()
        model.train()
        total, nb = 0.0, 0
        for (x,) in train_loader:
            x = x.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.reshape(-1, N_CLASSES), x.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * x.size(0)
            nb += x.size(0)
        perte_train = total / nb
        perte_val = perte_moyenne(model, val_loader, criterion)
        historique.append({"epoque": epoque, "train": perte_train, "val": perte_val})
        log.info(f"{prefixe}époque {epoque:2d}  perte train {perte_train:.4f}  perte val normal {perte_val:.4f}"
                 f"  ({time.time() - debut:.0f} s)")
        if not np.isfinite(perte_train):
            log.info(f"{prefixe}perte non finie, arrêt")
            break
        if perte_val < meilleure - 1e-4:
            meilleure, attente = perte_val, 0
            meilleur_etat = copy.deepcopy(model.state_dict())
        else:
            attente += 1
            if attente >= patience:
                log.info(f"{prefixe}arrêt précoce (meilleure perte val {meilleure:.4f})")
                break
    model.load_state_dict(meilleur_etat)
    return historique, time.time() - t0


def scores_par_bloc(model, X, batch_size=2048):
    model.eval()
    crit = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
    resultats = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x = X[i:i + batch_size].to(DEVICE)
            logits = model(x)
            pertes = crit(logits.reshape(-1, N_CLASSES), x.reshape(-1)).reshape(x.shape)
            masque = (x != 0).float()
            score = (pertes * masque).sum(dim=1) / masque.sum(dim=1).clamp(min=1)
            resultats.append(score.cpu())
    return torch.cat(resultats).numpy()


def reconstruire(model, X):
    model.eval()
    with torch.no_grad():
        return model(X.to(DEVICE)).argmax(dim=-1).cpu()


def mesures(y, pred):
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    vp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    vn = int(((y == 0) & (pred == 0)).sum())
    p = vp / (vp + fp) if vp + fp else 0.0
    r = vp / (vp + fn) if vp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "rappel": r, "f1": f1, "VP": vp, "FP": fp, "FN": fn, "VN": vn}


def ecrire_json(chemin, obj):
    Path(chemin).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def lire_json(chemin):
    return json.loads(Path(chemin).read_text(encoding="utf-8"))


def templates():
    # Texte des événements (E5 = "Receiving block ..."), si disponible
    p = DATA / "templates.csv"
    if p.exists():
        t = pd.read_csv(p)
        return dict(zip(t["num"], t["texte"]))
    return {}
