"""Fine-tuning de CamemBERT pour la classification d'intentions.

Boucle d'entraînement écrite à la main plutôt qu'avec le Trainer de
HuggingFace : c'est une vingtaine de lignes de plus, mais chaque étape est
visible et rien ne dépend d'une API qui change d'une version à l'autre.

Usage :
    python ml/train_intent.py
    python ml/train_intent.py --epochs 5 --class-weights
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

RACINE = Path(__file__).resolve().parents[1]
TRAIN = RACINE / "data" / "raw" / "intents_train.csv"
VAL = RACINE / "data" / "raw" / "intents_val.csv"
ARTEFACTS = RACINE / "ml" / "artifacts" / "intent-model"
JOURNAL = RACINE / "docs" / "training.json"

GRAINE = 42


class JeuIntentions(Dataset):
    """Enveloppe les phrases déjà tokenisées pour le DataLoader."""

    def __init__(self, encodages, etiquettes):
        self.encodages = encodages
        self.etiquettes = etiquettes

    def __len__(self):
        return len(self.etiquettes)

    def __getitem__(self, i):
        element = {c: v[i] for c, v in self.encodages.items()}
        element["labels"] = torch.tensor(self.etiquettes[i])
        return element


def charger(tokenizer, chemin, vers_id, longueur_max):
    donnees = pd.read_csv(chemin)
    encodages = tokenizer(
        donnees["texte"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=longueur_max,
        return_tensors="pt",
    )
    etiquettes = [vers_id[nom] for nom in donnees["intention"]]
    return JeuIntentions(encodages, etiquettes)


def evaluer(modele, chargeur, appareil, perte_fn):
    """Renvoie (F1 macro, perte moyenne) sur un jeu, en mode inférence.

    La perte est suivie en plus du F1 : si le F1 sature, elle reste un
    signal exploitable pour départager deux epochs.
    """
    modele.eval()
    vraies, predites, perte_totale = [], [], 0.0
    with torch.no_grad():
        for lot in chargeur:
            etiquettes = lot.pop("labels").to(appareil)
            lot = {c: v.to(appareil) for c, v in lot.items()}
            with torch.amp.autocast("cuda", enabled=appareil.type == "cuda"):
                sorties = modele(**lot)
                perte_totale += perte_fn(sorties.logits, etiquettes).item()
            predites.extend(sorties.logits.argmax(dim=-1).cpu().tolist())
            vraies.extend(etiquettes.cpu().tolist())
    return (f1_score(vraies, predites, average="macro"),
            perte_totale / len(chargeur))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--modele", default="camembert-base")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--longueur-max", type=int, default=128)
    p.add_argument("--class-weights", action="store_true",
                   help="pondère la perte pour compenser le déséquilibre")
    args = p.parse_args()

    torch.manual_seed(GRAINE)
    np.random.seed(GRAINE)

    appareil = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Appareil : {appareil}")
    if appareil.type == "cuda":
        print(f"Carte    : {torch.cuda.get_device_name(0)}")
    else:
        print("Attention : aucun GPU détecté, l'entraînement sera très lent.")

    # --- étiquettes ---------------------------------------------------
    train_brut = pd.read_csv(TRAIN)
    intentions = sorted(train_brut["intention"].unique())
    vers_id = {nom: i for i, nom in enumerate(intentions)}
    vers_nom = {i: nom for nom, i in vers_id.items()}
    print(f"Intentions : {len(intentions)}")

    # --- données ------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.modele)
    jeu_train = charger(tokenizer, TRAIN, vers_id, args.longueur_max)
    jeu_val = charger(tokenizer, VAL, vers_id, args.longueur_max)

    chargeur_train = DataLoader(jeu_train, batch_size=args.batch, shuffle=True)
    chargeur_val = DataLoader(jeu_val, batch_size=args.batch * 2)
    print(f"Entraînement : {len(jeu_train)} phrases, "
          f"validation : {len(jeu_val)}")

    # --- modèle -------------------------------------------------------
    modele = AutoModelForSequenceClassification.from_pretrained(
        args.modele, num_labels=len(intentions),
    ).to(appareil)

    # --- perte --------------------------------------------------------
    # Sans pondération, le modèle a intérêt à négliger les classes rares.
    # Le poids d'une classe est inversement proportionnel à sa fréquence.
    if args.class_weights:
        effectifs = train_brut["intention"].value_counts()
        poids = torch.tensor(
            [len(train_brut) / (len(intentions) * effectifs[nom])
             for nom in intentions],
            dtype=torch.float, device=appareil,
        )
        print("Pondération des classes activée")
    else:
        poids = None
    perte_fn = torch.nn.CrossEntropyLoss(weight=poids)

    # --- optimiseur et planificateur -----------------------------------
    # Taux d'apprentissage très faible : on ajuste un modèle déjà entraîné,
    # on ne l'apprend pas de zéro. Trop haut, il oublie son français.
    optimiseur = torch.optim.AdamW(modele.parameters(), lr=args.lr,
                                   weight_decay=0.01)
    total_pas = len(chargeur_train) * args.epochs
    planificateur = get_linear_schedule_with_warmup(
        optimiseur,
        num_warmup_steps=int(0.1 * total_pas),
        num_training_steps=total_pas,
    )

    # fp16 : Turing (RTX 2060 Super) ne gère pas le bf16.
    scaler = torch.amp.GradScaler("cuda", enabled=appareil.type == "cuda")

    # --- boucle --------------------------------------------------------
    ARTEFACTS.mkdir(parents=True, exist_ok=True)
    meilleur_score = (0.0, float("-inf"))  # (F1, -perte)
    meilleur_f1 = 0.0
    historique = []
    depart = time.time()

    for epoch in range(1, args.epochs + 1):
        modele.train()
        perte_totale = 0.0

        for i, lot in enumerate(chargeur_train, 1):
            etiquettes = lot.pop("labels").to(appareil)
            lot = {c: v.to(appareil) for c, v in lot.items()}

            optimiseur.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=appareil.type == "cuda"):
                sorties = modele(**lot)
                perte = perte_fn(sorties.logits, etiquettes)

            scaler.scale(perte).backward()
            scaler.unscale_(optimiseur)
            torch.nn.utils.clip_grad_norm_(modele.parameters(), 1.0)
            scaler.step(optimiseur)
            scaler.update()
            planificateur.step()

            perte_totale += perte.item()
            if i % 50 == 0:
                print(f"  epoch {epoch} | pas {i}/{len(chargeur_train)} "
                      f"| perte {perte_totale / i:.4f}", end="\r")

        f1_val, perte_val = evaluer(modele, chargeur_val, appareil, perte_fn)
        perte_moy = perte_totale / len(chargeur_train)
        ecoule = time.time() - depart
        print(f"  epoch {epoch} | perte train {perte_moy:.4f} "
              f"| perte val {perte_val:.4f} "
              f"| F1 macro val {f1_val:.4f} | {ecoule:.0f}s")

        historique.append({
            "epoch": epoch,
            "perte_train": round(perte_moy, 4),
            "perte_val": round(perte_val, 4),
            "f1_val": round(f1_val, 4),
        })

        # Le F1 décide, la perte départage les égalités.
        score = (round(f1_val, 4), -perte_val)
        if score > meilleur_score:
            meilleur_score = score
            meilleur_f1 = f1_val
            modele.save_pretrained(ARTEFACTS)
            tokenizer.save_pretrained(ARTEFACTS)
            (ARTEFACTS / "labels.json").write_text(
                json.dumps({"vers_id": vers_id, "vers_nom": vers_nom},
                           indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"           meilleur modèle sauvegardé ({f1_val:.4f})")

    # --- journal --------------------------------------------------------
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    JOURNAL.write_text(json.dumps({
        "modele": args.modele,
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "class_weights": args.class_weights,
        "duree_s": round(time.time() - depart),
        "meilleur_f1_val": round(meilleur_f1, 4),
        "meilleure_perte_val": round(-meilleur_score[1], 4),
        "historique": historique,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nMeilleur F1 macro validation : {meilleur_f1:.4f}")
    print(f"Modèle dans {ARTEFACTS.relative_to(RACINE)}")
    print(f"Journal dans {JOURNAL.relative_to(RACINE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())