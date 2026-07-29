"""
Matrices de confusion, baseline et CamemBERT, sur le même jeu de test.

Étape 5b. Le script ne recalcule aucune prédiction : il relit les deux
fichiers .npz produits par baseline.py et evaluate_intent.py, et refuse de
tourner s'ils ne portent pas exactement sur les mêmes phrases dans le même
ordre.

Les intentions sont ordonnées par type de routage, transactionnelles puis
informationnelles puis rejet, et non par ordre alphabétique. Ce choix rend
visible la seule distinction qui compte pour l'étape 9 : une confusion à
l'intérieur d'un bloc laisse le routeur sur le bon chemin, une confusion
qui traverse un trait plein l'envoie ailleurs.

Usage :
    python ml\\plot_confusion.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # aucun affichage interactif, on écrit un fichier

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

RACINE = Path(__file__).resolve().parents[1]

ORDRE_TYPES = ["transactionnelle", "informationnelle", "rejet"]


# --------------------------------------------------------------------------
# Types d'intentions
# --------------------------------------------------------------------------

def charger_types(etiquettes: list[str]) -> dict[str, str]:
    """
    Lit la correspondance intention vers type dans data/generator/intents.py.

    C'est le référentiel unique : dupliquer cette table ici garantirait
    qu'elle finisse par diverger de celle qui a servi à générer les données.
    """
    dossier = RACINE / "data" / "generator"
    sys.path.insert(0, str(dossier))
    try:
        from intents import INTENTIONS  # type: ignore
    except Exception as erreur:
        raise SystemExit(
            f"Impossible de lire INTENTIONS dans {dossier / 'intents.py'} : "
            f"{erreur}"
        )
    finally:
        sys.path.pop(0)

    types = {nom: donnees["type"] for nom, donnees in INTENTIONS.items()}

    manquantes = [e for e in etiquettes if e not in types]
    if manquantes:
        raise SystemExit(
            f"Intentions absentes de intents.py : {manquantes}. "
            "Le modèle et le générateur ne parlent pas de la même chose."
        )
    inconnus = sorted(set(types.values()) - set(ORDRE_TYPES))
    if inconnus:
        raise SystemExit(f"Types inconnus dans intents.py : {inconnus}")

    return types


def ordonner(etiquettes: list[str], types: dict[str, str]) -> list[str]:
    """Trie les intentions par type de routage, puis par nom."""
    return sorted(etiquettes, key=lambda e: (ORDRE_TYPES.index(types[e]), e))


def frontieres(ordre: list[str], types: dict[str, str]) -> list[int]:
    """Indices où le type change, pour tracer les séparateurs."""
    return [i for i in range(1, len(ordre)) if types[ordre[i]] != types[ordre[i - 1]]]


# --------------------------------------------------------------------------
# Lecture et contrôle d'alignement
# --------------------------------------------------------------------------

def charger_predictions(chemin_cam: Path, chemin_base: Path):
    for chemin in (chemin_cam, chemin_base):
        if not chemin.exists():
            raise SystemExit(
                f"{chemin} absent. Relancer evaluate_intent.py et "
                "baseline.py avant cette étape."
            )

    cam = np.load(chemin_cam)
    base = np.load(chemin_base)

    for nom, fichier, attendus in (
        ("CamemBERT", cam, ("probas", "etiquettes", "reels", "textes")),
        ("baseline", base, ("predits", "reels", "textes")),
    ):
        manquants = [c for c in attendus if c not in fichier.files]
        if manquants:
            raise SystemExit(
                f"Champs manquants dans le fichier {nom} : {manquants}. "
                "Le script qui l'a produit est une version antérieure, "
                "le relancer."
            )

    # Contrôle d'alignement : les deux séries doivent porter sur les mêmes
    # phrases dans le même ordre. Apparier deux fichiers en supposant qu'ils
    # sont triés pareil est la version silencieuse du jeu de test périmé.
    if len(cam["textes"]) != len(base["textes"]):
        raise SystemExit(
            f"Longueurs différentes : CamemBERT {len(cam['textes'])} "
            f"contre baseline {len(base['textes'])}."
        )
    if not np.array_equal(cam["textes"], base["textes"]):
        divergentes = int((cam["textes"] != base["textes"]).sum())
        raise SystemExit(
            f"ALIGNEMENT : {divergentes} phrase(s) diffèrent entre les deux "
            "fichiers de prédictions. Ils ne portent pas sur le même jeu de "
            "test ou pas dans le même ordre. Relancer les deux scripts."
        )
    if not np.array_equal(cam["reels"], base["reels"]):
        raise SystemExit(
            "ALIGNEMENT : les étiquettes réelles diffèrent entre les deux "
            "fichiers alors que les textes sont identiques."
        )

    etiquettes = [str(e) for e in cam["etiquettes"]]
    predits_cam = [etiquettes[i] for i in cam["probas"].argmax(axis=1)]
    predits_base = [str(p) for p in base["predits"]]
    reels = [str(r) for r in cam["reels"]]

    print(f"[ok] alignement vérifié sur {len(reels)} phrases")
    return reels, predits_cam, predits_base, etiquettes


# --------------------------------------------------------------------------
# Analyse
# --------------------------------------------------------------------------

def erreurs_inter_type(reels, predits, types) -> dict:
    """
    Sépare les erreurs qui changent de type de routage de celles qui n'en
    changent pas. Une confusion question_tarif contre question_garantie
    reste sur le chemin RAG ; une confusion question_delai contre resilier
    déclenche un flux métier à la place d'une réponse.
    """
    erreurs = [(r, p) for r, p in zip(reels, predits) if r != p]
    inter = [(r, p) for r, p in erreurs if types[r] != types[p]]
    return {
        "erreurs": len(erreurs),
        "inter_type": len(inter),
        "intra_type": len(erreurs) - len(inter),
        "part_inter_type": round(len(inter) / len(erreurs), 4) if erreurs else 0.0,
        "part_du_jeu": round(len(inter) / len(reels), 4),
    }


def paires_frequentes(reels, predits, n: int = 8) -> list[tuple[str, str, int]]:
    compteur = Counter((r, p) for r, p in zip(reels, predits) if r != p)
    return [(r, p, c) for (r, p), c in compteur.most_common(n)]


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------

def dessiner(axe, matrice, ordre, types, titre):
    with np.errstate(divide="ignore", invalid="ignore"):
        taux = np.nan_to_num(matrice / matrice.sum(axis=1, keepdims=True))

    image = axe.imshow(taux, cmap="Blues", vmin=0, vmax=1)

    axe.set_xticks(range(len(ordre)))
    axe.set_yticks(range(len(ordre)))
    axe.set_xticklabels(ordre, rotation=45, ha="right", fontsize=8)
    axe.set_yticklabels(ordre, fontsize=8)
    axe.set_xlabel("prédit", fontsize=9)
    axe.set_ylabel("réel", fontsize=9)
    axe.set_title(titre, fontsize=11, pad=12)

    # les effectifs bruts en clair : un taux de 0,50 sur 2 phrases et sur
    # 100 phrases ne se lisent pas de la même façon
    for i in range(len(ordre)):
        for j in range(len(ordre)):
            valeur = int(matrice[i, j])
            if valeur == 0:
                continue
            axe.text(
                j, i, str(valeur), ha="center", va="center", fontsize=7,
                color="white" if taux[i, j] > 0.55 else "#22303c",
            )

    for position in frontieres(ordre, types):
        axe.axhline(position - 0.5, color="#c0392b", linewidth=1.1)
        axe.axvline(position - 0.5, color="#c0392b", linewidth=1.1)

    return image


# --------------------------------------------------------------------------

def main() -> None:
    parseur = argparse.ArgumentParser()
    parseur.add_argument("--camembert", default="ml/artifacts/predictions_test.npz")
    parseur.add_argument("--baseline", default="ml/artifacts/predictions_baseline_test.npz")
    parseur.add_argument("--figure", default="docs/matrice_confusion.png")
    parseur.add_argument("--sortie", default="docs/confusion.json")
    args = parseur.parse_args()

    reels, predits_cam, predits_base, etiquettes = charger_predictions(
        RACINE / args.camembert, RACINE / args.baseline
    )

    types = charger_types(etiquettes)
    ordre = ordonner(etiquettes, types)
    print(f"[ok] ordre par type : "
          + ", ".join(f"{t} ({sum(1 for e in ordre if types[e] == t)})"
                      for t in ORDRE_TYPES))

    matrices = {
        "baseline": confusion_matrix(reels, predits_base, labels=ordre),
        "camembert": confusion_matrix(reels, predits_cam, labels=ordre),
    }

    figure, axes = plt.subplots(1, 2, figsize=(17, 7.5))
    image = None
    for axe, (nom, titre) in zip(axes, (
        ("baseline", "Baseline TF-IDF pondérée"),
        ("camembert", "CamemBERT fine-tuné"),
    )):
        image = dessiner(axe, matrices[nom], ordre, types, titre)

    figure.colorbar(image, ax=axes, fraction=0.025, pad=0.02,
                    label="part de la ligne (rappel)")
    figure.suptitle(
        f"Matrices de confusion sur {len(reels)} phrases de test\n"
        "traits rouges : frontières de type de routage "
        "(transactionnelle, informationnelle, rejet)",
        fontsize=12,
    )

    chemin_figure = RACINE / args.figure
    chemin_figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(chemin_figure, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"[ok] figure écrite dans {args.figure}")

    # ----------------------------------------------------------------------
    resultats = {"n": len(reels), "ordre": ordre, "types": types, "modeles": {}}

    for nom, predits in (("baseline", predits_base), ("camembert", predits_cam)):
        routage = erreurs_inter_type(reels, predits, types)
        paires = paires_frequentes(reels, predits)
        resultats["modeles"][nom] = {
            "routage": routage,
            "paires_frequentes": [
                {"reel": r, "predit": p, "n": c,
                 "inter_type": types[r] != types[p]}
                for r, p, c in paires
            ],
        }

        print()
        print(f"{nom}")
        print(f"  erreurs totales      : {routage['erreurs']}")
        print(f"  dont changement de type : {routage['inter_type']} "
              f"({routage['part_inter_type']:.1%} des erreurs, "
              f"{routage['part_du_jeu']:.1%} du jeu)")
        print("  confusions les plus fréquentes :")
        for r, p, c in paires:
            marque = "  <-- change de chemin" if types[r] != types[p] else ""
            print(f"    {r} pris pour {p} : {c}{marque}")

    chemin_sortie = RACINE / args.sortie
    chemin_sortie.write_text(
        json.dumps(resultats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[ok] résultats écrits dans {args.sortie}")


if __name__ == "__main__":
    main()