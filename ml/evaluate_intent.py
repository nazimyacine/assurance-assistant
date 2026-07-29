"""
Évaluation du classifieur d'intentions CamemBERT.

Étape 5a : métriques globales, par classe, par niveau de difficulté, et
comparaison avec la baseline sur EXACTEMENT le même jeu de test.

Le script refuse de tourner si une phrase du test apparaît à l'entraînement.
Une évaluation qui s'interrompt vaut mieux qu'un joli chiffre faux.

Usage :
    python ml\\evaluate_intent.py
    python ml\\evaluate_intent.py --model-dir ml/artifacts/intent-model
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

RACINE = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# Lecture des données
# --------------------------------------------------------------------------

def normaliser(texte) -> str:
    """Minuscules sans accents, pour rapprocher « bruité » et « bruite »."""
    decompose = unicodedata.normalize("NFKD", str(texte))
    return "".join(c for c in decompose if not unicodedata.combining(c)).lower().strip()


def detecter_colonne(df: pd.DataFrame, candidats: list[str], role: str) -> str:
    """Retrouve une colonne parmi plusieurs noms possibles."""
    normalisees = {normaliser(c): c for c in df.columns}
    for candidat in candidats:
        if normaliser(candidat) in normalisees:
            return normalisees[normaliser(candidat)]
    raise SystemExit(
        f"Colonne « {role} » introuvable. Colonnes présentes : "
        f"{list(df.columns)}. Noms attendus : {candidats}."
    )


def verifier_absence_de_fuite(train: pd.Series, test: pd.Series) -> None:
    """Interrompt le programme si le test contient des phrases d'entraînement."""
    textes_train = {t.lower().strip() for t in train}
    fuites = [t for t in test if t.lower().strip() in textes_train]
    if fuites:
        raise SystemExit(
            f"FUITE : {len(fuites)} phrase(s) du test sont présentes à "
            f"l'entraînement. Exemple : « {fuites[0]} »"
        )
    print(f"[ok] aucune fuite entre entraînement et test ({len(test)} phrases)")


# --------------------------------------------------------------------------
# Étiquettes
# --------------------------------------------------------------------------

def _candidats_etiquettes(objet):
    """
    Parcourt un objet JSON et renvoie toutes les listes d'étiquettes
    plausibles qu'il contient, quelle que soit la forme du fichier.

    Formes reconnues :
        ["resilier", "souscrire", ...]
        {"0": "resilier", "1": "souscrire", ...}
        {"resilier": 0, "souscrire": 1, ...}
        {"vers_id": {...}, "vers_nom": {...}}   (imbriqué, on descend)
    """
    trouves = []

    if isinstance(objet, list) and objet and all(isinstance(x, str) for x in objet):
        trouves.append(list(objet))
        return trouves

    if isinstance(objet, dict) and objet:
        cles = list(objet.keys())
        valeurs = list(objet.values())

        if (all(isinstance(v, str) for v in valeurs)
                and all(str(k).lstrip("-").isdigit() for k in cles)):
            trouves.append([objet[k] for k in sorted(cles, key=lambda x: int(x))])

        elif (all(isinstance(v, int) and not isinstance(v, bool) for v in valeurs)
                and all(isinstance(k, str) for k in cles)):
            inverse = {int(v): k for k, v in objet.items()}
            if len(inverse) == len(objet):
                trouves.append([inverse[i] for i in sorted(inverse)])

        else:
            for valeur in valeurs:
                trouves.extend(_candidats_etiquettes(valeur))

    return trouves


def charger_etiquettes(model_dir: Path, model) -> list[str]:
    """
    Récupère la liste des intentions dans l'ordre des sorties du modèle.

    Un mauvais ordre produirait un rapport entièrement faux mais crédible.
    Le script s'interrompt donc plutôt que de deviner : on n'accepte qu'une
    liste dont la longueur correspond au nombre de sorties du modèle.
    """
    attendu = int(getattr(model.config, "num_labels", 0)) or None

    sources = []
    for nom in ("labels.json", "label_encoder.json", "intentions.json"):
        chemin = model_dir / nom
        if chemin.exists():
            sources.append((nom, json.loads(chemin.read_text(encoding="utf-8"))))

    id2label = getattr(model.config, "id2label", {}) or {}
    if id2label:
        sources.append(("config.json (id2label)", id2label))

    for nom, donnees in sources:
        for candidat in _candidats_etiquettes(donnees):
            if attendu is not None and len(candidat) != attendu:
                continue
            if len(set(candidat)) != len(candidat):
                continue
            if all(str(e).startswith("LABEL_") for e in candidat):
                continue
            print(f"[ok] {len(candidat)} étiquettes lues dans {nom}")
            print(f"     ordre : {candidat[0]}, {candidat[1]}, ..., {candidat[-1]}")
            return [str(e) for e in candidat]

    raise SystemExit(
        f"Étiquettes exploitables introuvables dans {model_dir}.\n"
        f"Le modèle a {attendu} sorties. Fichiers examinés : "
        f"{[nom for nom, _ in sources] or 'aucun'}.\n"
        "Déposer un labels.json contenant la liste des intentions dans "
        "l'ordre des indices, ou renseigner id2label à l'enregistrement "
        "du modèle dans train_intent.py."
    )


# --------------------------------------------------------------------------
# Modèle
# --------------------------------------------------------------------------

@torch.no_grad()
def predire(model, tokenizer, textes: list[str], device: str,
            batch: int = 64, longueur_max: int = 128) -> np.ndarray:
    """Renvoie la matrice des probabilités, une ligne par phrase."""
    model.eval()
    sorties = []

    for debut in range(0, len(textes), batch):
        lot = textes[debut:debut + batch]
        encodage = tokenizer(
            lot, truncation=True, max_length=longueur_max,
            padding=True, return_tensors="pt",
        ).to(device)
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(**encodage).logits
        else:
            logits = model(**encodage).logits
        sorties.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())

    return np.vstack(sorties)


# --------------------------------------------------------------------------
# Baseline
# --------------------------------------------------------------------------

def lire_baseline(chemin: Path) -> dict:
    """
    Lit docs/baseline.json et retient la MEILLEURE variante par F1 macro.

    Se comparer à une baseline volontairement affaiblie serait malhonnête :
    c'est la configuration la plus forte qui sert de référence. Les autres
    restent affichées à côté.

    Récupère aussi les exactitudes par difficulté quand elles existent,
    sous la forme exactitude_<difficulte>.
    """
    vide = {"variantes": {}, "retenue": None, "f1_macro": None,
            "exactitude": None, "par_difficulte": {}}

    if not chemin.exists():
        print(f"[!] {chemin} absent, comparaison laissée vide")
        return vide

    donnees = json.loads(chemin.read_text(encoding="utf-8"))

    variantes = {
        nom: contenu for nom, contenu in donnees.items()
        if isinstance(contenu, dict) and "f1_macro" in contenu
    }
    if not variantes and "f1_macro" in donnees:
        variantes = {"baseline": donnees}
    if not variantes:
        print(f"[!] aucune variante exploitable dans {chemin.name}")
        return vide

    retenue = max(variantes, key=lambda nom: variantes[nom]["f1_macro"])
    contenu = variantes[retenue]

    par_difficulte = {
        normaliser(cle[len("exactitude_"):]): float(valeur)
        for cle, valeur in contenu.items()
        if cle.startswith("exactitude_") and isinstance(valeur, (int, float))
    }

    exactitude = contenu.get("exactitude")

    print(f"[ok] baseline de référence : « {retenue} », "
          f"F1 macro {contenu['f1_macro']:.4f} "
          f"({len(variantes)} variante(s) dans le fichier)")

    return {
        "variantes": variantes,
        "retenue": retenue,
        "f1_macro": float(contenu["f1_macro"]),
        "exactitude": float(exactitude) if exactitude is not None else None,
        "par_difficulte": par_difficulte,
    }


# --------------------------------------------------------------------------

def main() -> None:
    parseur = argparse.ArgumentParser()
    parseur.add_argument("--model-dir", default="ml/artifacts/intent-model")
    parseur.add_argument("--test", default="data/eval/intents_test.csv")
    parseur.add_argument("--train", default="data/raw/intents_train.csv")
    parseur.add_argument("--baseline", default="docs/baseline.json")
    parseur.add_argument("--sortie", default="docs/eval_intent.json")
    parseur.add_argument("--predictions", default="ml/artifacts/predictions_test.npz")
    args = parseur.parse_args()

    model_dir = RACINE / args.model_dir
    chemin_test = RACINE / args.test
    chemin_train = RACINE / args.train

    # 1. données -----------------------------------------------------------
    test = pd.read_csv(chemin_test)
    col_texte = detecter_colonne(test, ["texte", "text", "message", "phrase"], "texte")
    col_label = detecter_colonne(test, ["intention", "label", "intent", "classe"], "intention")

    col_diff = None
    normalisees = {normaliser(c) for c in test.columns}
    for candidat in ("difficulte", "difficulty", "niveau"):
        if candidat in normalisees:
            col_diff = detecter_colonne(test, [candidat], "difficulté")
            break

    if not chemin_train.exists():
        raise SystemExit(
            f"{chemin_train} absent. Le garde-fou anti-fuite ne peut pas "
            "tourner ; régénérer les données avant d'évaluer."
        )
    train = pd.read_csv(chemin_train)
    verifier_absence_de_fuite(train[col_texte], test[col_texte])

    # 2. modèle ------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ok] device : {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    etiquettes = charger_etiquettes(model_dir, model)

    inconnues = set(test[col_label]) - set(etiquettes)
    if inconnues:
        raise SystemExit(
            f"Le jeu de test contient des intentions absentes du modèle : "
            f"{sorted(inconnues)}"
        )

    # 3. prédiction --------------------------------------------------------
    probas = predire(model, tokenizer, test[col_texte].tolist(), device)
    indices = probas.argmax(axis=1)
    predits = [etiquettes[i] for i in indices]
    reels = test[col_label].tolist()

    chemin_preds = RACINE / args.predictions
    chemin_preds.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        chemin_preds,
        probas=probas,
        etiquettes=np.asarray(etiquettes, dtype=str),
        reels=np.asarray(reels, dtype=str),
        textes=np.asarray(test[col_texte], dtype=str),
    )
    print(f"[ok] probabilités enregistrées dans {args.predictions} (pour l'étape 5c)")

    # 4. métriques ---------------------------------------------------------
    f1_macro = f1_score(reels, predits, average="macro", zero_division=0)
    exactitude = accuracy_score(reels, predits)
    prec_macro, rap_macro, _, _ = precision_recall_fscore_support(
        reels, predits, average="macro", zero_division=0
    )

    prec, rap, f1, support = precision_recall_fscore_support(
        reels, predits, labels=etiquettes, zero_division=0
    )
    par_classe = {
        etiquette: {
            "precision": round(float(prec[i]), 4),
            "rappel": round(float(rap[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        for i, etiquette in enumerate(etiquettes)
    }

    par_difficulte = {}
    if col_diff:
        cadre = pd.DataFrame({
            "reel": reels, "predit": predits,
            "diff": test[col_diff].map(normaliser),
        })
        for niveau, sous in cadre.groupby("diff"):
            par_difficulte[str(niveau)] = {
                "exactitude": round(float(accuracy_score(sous["reel"], sous["predit"])), 4),
                "f1_macro": round(float(f1_score(
                    sous["reel"], sous["predit"], average="macro", zero_division=0
                )), 4),
                "n": int(len(sous)),
                "classes_presentes": int(sous["reel"].nunique()),
            }

    baseline = lire_baseline(RACINE / args.baseline)

    resultats = {
        "modele": str(args.model_dir),
        "jeu_test": {"chemin": str(args.test), "n": int(len(test))},
        "global": {
            "f1_macro": round(float(f1_macro), 4),
            "exactitude": round(float(exactitude), 4),
            "precision_macro": round(float(prec_macro), 4),
            "rappel_macro": round(float(rap_macro), 4),
        },
        "par_classe": par_classe,
        "par_difficulte": par_difficulte,
        "baseline": baseline,
    }

    chemin_sortie = RACINE / args.sortie
    chemin_sortie.parent.mkdir(parents=True, exist_ok=True)
    chemin_sortie.write_text(
        json.dumps(resultats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 5. affichage ---------------------------------------------------------
    print()
    print(f"CamemBERT sur {len(test)} phrases de test")
    print(f"  F1 macro   : {f1_macro:.4f}")
    print(f"  Exactitude : {exactitude:.4f}")

    print()
    print("Comparaison sur le même jeu de test")
    print("| Modèle | F1 macro | Exactitude |")
    print("|---|---|---|")
    for nom, contenu in baseline["variantes"].items():
        marque = " (référence)" if nom == baseline["retenue"] else ""
        ex = contenu.get("exactitude")
        ex_txt = f"{ex:.4f}" if isinstance(ex, (int, float)) else "?"
        print(f"| Baseline TF-IDF {nom}{marque} | {contenu['f1_macro']:.4f} | {ex_txt} |")
    print(f"| CamemBERT | {f1_macro:.4f} | {exactitude:.4f} |")

    if baseline["f1_macro"] is not None:
        ecart = f1_macro - baseline["f1_macro"]
        print(f"\nÉcart de F1 macro contre la meilleure baseline "
              f"(« {baseline['retenue']} ») : {ecart:+.4f}")

    if par_difficulte:
        print()
        print("Par difficulté, exactitude contre exactitude")
        print("| Difficulté | Baseline | CamemBERT | n | classes |")
        print("|---|---|---|---|---|")
        for niveau, valeurs in sorted(par_difficulte.items()):
            ref = baseline["par_difficulte"].get(niveau)
            ref_txt = f"{ref:.4f}" if ref is not None else "?"
            print(f"| {niveau} | {ref_txt} | {valeurs['exactitude']:.4f} | "
                  f"{valeurs['n']} | {valeurs['classes_presentes']} |")
        print()
        print("Le F1 macro par sous-ensemble n'est pas comparable d'une ligne")
        print("à l'autre, les classes n'y ont pas les mêmes effectifs.")
        print("Pour information seulement :")
        for niveau, valeurs in sorted(par_difficulte.items()):
            print(f"  {niveau:<10} f1 macro {valeurs['f1_macro']:.3f}")

    print()
    print("Par classe, des plus faibles aux plus fortes")
    for etiquette, valeurs in sorted(par_classe.items(), key=lambda x: x[1]["f1"]):
        print(f"  {etiquette:<26} f1 {valeurs['f1']:.3f}  (n={valeurs['support']})")

    print(f"\n[ok] résultats écrits dans {args.sortie}")


if __name__ == "__main__":
    main()