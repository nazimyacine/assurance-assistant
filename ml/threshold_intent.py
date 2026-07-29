"""
Coût du seuil de rejet du classifieur d'intentions.

Étape 5c. Le modèle sort une probabilité par intention. En dessous d'un
seuil, l'assistant demande une reformulation plutôt que d'agir. Ce script
mesure ce que ce seuil coûte et ce qu'il rapporte, au lieu de le fixer à
0,6 parce que le chiffre a l'air raisonnable.

Trois choses qu'un tableau d'exactitude ne dit pas :

  - toutes les erreurs ne se valent pas. Une erreur qui reste sur le même
    type de routage produit une réponse imparfaite ; une erreur qui prédit
    une intention transactionnelle déclenche un flux métier à tort. C'est
    la seconde que le seuil doit intercepter.

  - le jeu de test compte 536 lignes mais 64 unités indépendantes. Toutes
    les mesures sont donc données deux fois, par ligne et par unité.

  - une probabilité de softmax n'est pas une probabilité. Un modèle
    fine-tuné est notoirement trop confiant. Le script mesure l'écart entre
    confiance annoncée et exactitude réelle avant de s'appuyer dessus.

Usage :
    python ml\\threshold_intent.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Le référentiel des types vit dans data/generator/intents.py. On réutilise
# le lecteur écrit à l'étape 5b plutôt que de dupliquer la table ici.
from plot_confusion import charger_types

RACINE = Path(__file__).resolve().parents[1]

SEUILS = np.round(np.arange(0.0, 1.0, 0.025), 3)
BINS_CALIBRATION = 10


# --------------------------------------------------------------------------
# Données
# --------------------------------------------------------------------------

def charger(chemin_npz: Path, chemin_csv: Path):
    """Relit les prédictions et les rattache aux gabarits du jeu de test."""
    if not chemin_npz.exists():
        raise SystemExit(f"{chemin_npz} absent. Relancer evaluate_intent.py.")

    donnees = np.load(chemin_npz)
    for champ in ("probas", "etiquettes", "reels", "textes"):
        if champ not in donnees.files:
            raise SystemExit(
                f"Champ « {champ} » absent de {chemin_npz.name}. "
                "Relancer evaluate_intent.py dans sa version à jour."
            )

    test = pd.read_csv(chemin_csv)
    if "gabarit" not in test.columns:
        raise SystemExit(
            f"{chemin_csv} n'a pas de colonne « gabarit ». Régénérer les "
            "données avec la version à jour de generate_intents.py."
        )

    textes_npz = np.asarray(donnees["textes"], dtype=str)
    textes_csv = np.asarray(test["texte"], dtype=str)
    if len(textes_npz) != len(textes_csv) or not np.array_equal(textes_npz, textes_csv):
        raise SystemExit(
            "ALIGNEMENT : les prédictions et le CSV ne portent pas sur les "
            "mêmes phrases dans le même ordre. Relancer evaluate_intent.py "
            "après la régénération des données."
        )

    etiquettes = [str(e) for e in donnees["etiquettes"]]
    probas = donnees["probas"]

    return {
        "etiquettes": etiquettes,
        "reels": np.asarray(donnees["reels"], dtype=str),
        "predits": np.array([etiquettes[i] for i in probas.argmax(axis=1)]),
        "confiance": probas.max(axis=1),
        "gabarit": np.asarray(test["gabarit"], dtype=str),
        "difficulte": np.asarray(test["difficulte"], dtype=str),
        "n": len(test),
    }


# --------------------------------------------------------------------------
# Balayage des seuils
# --------------------------------------------------------------------------

def moyenne_par_gabarit(valeurs: np.ndarray, gabarits: np.ndarray) -> float:
    """Moyenne des moyennes par gabarit : chaque unité pèse pareil.

    Sans cela un gabarit décliné en 27 phrases pèse 27 fois plus qu'un
    gabarit décliné en 1, alors que les deux valent une observation.
    """
    if len(valeurs) == 0:
        return float("nan")
    cadre = pd.DataFrame({"v": valeurs, "g": gabarits})
    return float(cadre.groupby("g")["v"].mean().mean())


def balayer(donnees: dict, types: dict[str, str]) -> list[dict]:
    reels, predits = donnees["reels"], donnees["predits"]
    confiance, gabarits = donnees["confiance"], donnees["gabarit"]
    n = donnees["n"]

    correct = predits == reels
    type_reel = np.array([types[r] for r in reels])
    type_predit = np.array([types[p] for p in predits])

    # Une erreur qui prédit une intention transactionnelle enclenche un flux
    # métier sur un message qui ne le demandait pas. C'est l'incident que le
    # seuil existe pour éviter.
    flux_a_tort = (~correct) & (type_predit == "transactionnelle")
    change_de_chemin = (~correct) & (type_reel != type_predit)

    lignes = []
    for seuil in SEUILS:
        accepte = confiance >= seuil
        rejete = ~accepte

        n_acceptes = int(accepte.sum())
        erreurs_evitees = int((rejete & ~correct).sum())
        legitimes_perdues = int((rejete & correct).sum())

        lignes.append({
            "seuil": float(seuil),
            "couverture": round(n_acceptes / n, 4),
            "couverture_gabarit": round(moyenne_par_gabarit(accepte, gabarits), 4),
            "exactitude_acceptes": round(
                float(correct[accepte].mean()) if n_acceptes else float("nan"), 4),
            "exactitude_acceptes_gabarit": round(
                moyenne_par_gabarit(correct[accepte], gabarits[accepte])
                if n_acceptes else float("nan"), 4),
            "erreurs_evitees": erreurs_evitees,
            "legitimes_perdues": legitimes_perdues,
            "rendement": round(erreurs_evitees / legitimes_perdues, 2)
            if legitimes_perdues else None,
            "erreurs_restantes": int((accepte & ~correct).sum()),
            "flux_a_tort": int((accepte & flux_a_tort).sum()),
            "change_de_chemin": int((accepte & change_de_chemin).sum()),
        })

    return lignes


def recommander(lignes: list[dict], couverture_min: float = 0.90) -> dict:
    """
    Règle de sélection, énoncée pour pouvoir être contestée.

    Parmi les seuils qui laissent passer au moins `couverture_min` des
    messages, retenir celui qui déclenche le moins de flux métier à tort.
    À égalité, le plus bas, parce qu'un seuil élevé se paie en frictions
    que ce jeu de test ne mesure pas : chaque rejet est un aller-retour
    supplémentaire imposé à l'assuré.
    """
    eligibles = [l for l in lignes if l["couverture"] >= couverture_min]
    if not eligibles:
        return {}
    meilleur = min(eligibles, key=lambda l: (l["flux_a_tort"], l["seuil"]))
    return {"regle": f"couverture >= {couverture_min:.0%}, "
                     "puis minimum de flux metier declenches a tort",
            "seuil": meilleur["seuil"], "detail": meilleur}


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------

def calibration(donnees: dict, bins: int = BINS_CALIBRATION) -> dict:
    """
    Compare confiance annoncée et exactitude constatée, par tranche.

    L'écart moyen pondéré (ECE) résume en un chiffre à quel point la
    probabilité de softmax est utilisable telle quelle. Un modèle
    parfaitement calibré aurait 0.
    """
    confiance = donnees["confiance"]
    correct = (donnees["predits"] == donnees["reels"]).astype(float)

    bornes = np.linspace(confiance.min(), 1.0, bins + 1)
    tranches, ece = [], 0.0

    for i in range(bins):
        bas, haut = bornes[i], bornes[i + 1]
        dans = (confiance >= bas) & (confiance < haut if i < bins - 1
                                     else confiance <= haut)
        if not dans.any():
            continue
        conf_moy = float(confiance[dans].mean())
        exact = float(correct[dans].mean())
        poids = int(dans.sum())
        ece += poids / len(confiance) * abs(conf_moy - exact)
        tranches.append({
            "borne_basse": round(float(bas), 4),
            "borne_haute": round(float(haut), 4),
            "n": poids,
            "confiance_moyenne": round(conf_moy, 4),
            "exactitude": round(exact, 4),
            "ecart": round(conf_moy - exact, 4),
        })

    return {"ece": round(ece, 4), "tranches": tranches,
            "confiance_moyenne": round(float(confiance.mean()), 4),
            "exactitude": round(float(correct.mean()), 4)}


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------

def dessiner(lignes, calib, recommande, chemin: Path) -> None:
    seuils = [l["seuil"] for l in lignes]
    figure, axes = plt.subplots(1, 3, figsize=(17, 5))

    # 1. couverture et exactitude
    axe = axes[0]
    axe.plot(seuils, [l["couverture"] for l in lignes],
             label="couverture (ligne)", color="#2c7fb8")
    axe.plot(seuils, [l["couverture_gabarit"] for l in lignes],
             label="couverture (gabarit)", color="#2c7fb8", linestyle="--")
    axe.plot(seuils, [l["exactitude_acceptes"] for l in lignes],
             label="exactitude sur acceptés", color="#c0392b")
    axe.set_xlabel("seuil de confiance")
    axe.set_ylim(0, 1.02)
    axe.set_title("Ce que le seuil laisse passer")
    axe.legend(fontsize=8)
    axe.grid(alpha=0.25)

    # 2. le compromis, en effectifs
    axe = axes[1]
    axe.plot(seuils, [l["erreurs_evitees"] for l in lignes],
             label="erreurs évitées", color="#27ae60")
    axe.plot(seuils, [l["legitimes_perdues"] for l in lignes],
             label="demandes légitimes rejetées", color="#c0392b")
    axe.plot(seuils, [l["flux_a_tort"] for l in lignes],
             label="flux métier déclenchés à tort", color="#8e44ad")
    axe.set_xlabel("seuil de confiance")
    axe.set_ylabel("nombre de messages")
    axe.set_title("Le compromis, en messages")
    axe.legend(fontsize=8)
    axe.grid(alpha=0.25)

    # 3. calibration
    axe = axes[2]
    conf = [t["confiance_moyenne"] for t in calib["tranches"]]
    exact = [t["exactitude"] for t in calib["tranches"]]
    axe.plot([0, 1], [0, 1], color="#7f8c8d", linestyle=":",
             label="calibration parfaite")
    axe.plot(conf, exact, marker="o", color="#2c7fb8", label="modèle")
    axe.set_xlabel("confiance annoncée")
    axe.set_ylabel("exactitude constatée")
    axe.set_title(f"Calibration, ECE = {calib['ece']:.3f}")
    axe.set_xlim(0, 1.02)
    axe.set_ylim(0, 1.02)
    axe.legend(fontsize=8)
    axe.grid(alpha=0.25)

    if recommande:
        for axe in axes[:2]:
            axe.axvline(recommande["seuil"], color="#e67e22", linewidth=1.2)

    figure.suptitle(
        "Seuil de rejet du classifieur d'intentions"
        + (f", seuil retenu {recommande['seuil']:.3f}" if recommande else ""),
        fontsize=12,
    )
    figure.tight_layout()
    chemin.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(chemin, dpi=150, bbox_inches="tight")
    plt.close(figure)


# --------------------------------------------------------------------------

def main() -> None:
    parseur = argparse.ArgumentParser()
    parseur.add_argument("--predictions", default="ml/artifacts/predictions_test.npz")
    parseur.add_argument("--test", default="data/eval/intents_test.csv")
    parseur.add_argument("--figure", default="docs/seuil_rejet.png")
    parseur.add_argument("--sortie", default="docs/seuil_rejet.json")
    parseur.add_argument("--couverture-min", type=float, default=0.90)
    args = parseur.parse_args()

    donnees = charger(RACINE / args.predictions, RACINE / args.test)
    types = charger_types(donnees["etiquettes"])
    print(f"[ok] {donnees['n']} phrases, "
          f"{len(set(donnees['gabarit']))} unités indépendantes")

    calib = calibration(donnees)
    print(f"[ok] confiance moyenne {calib['confiance_moyenne']:.4f} "
          f"pour une exactitude de {calib['exactitude']:.4f}, "
          f"ECE {calib['ece']:.4f}")

    lignes = balayer(donnees, types)
    recommande = recommander(lignes, args.couverture_min)

    dessiner(lignes, calib, recommande, RACINE / args.figure)
    print(f"[ok] figure écrite dans {args.figure}")

    # ----------------------------------------------------------------------
    print()
    print("| seuil | couverture | exact. acceptés | erreurs évitées | "
          "légitimes perdues | flux à tort |")
    print("|---|---|---|---|---|---|")
    for ligne in lignes:
        if round(ligne["seuil"] * 1000) % 100:  # une ligne sur quatre
            continue
        print(f"| {ligne['seuil']:.2f} | {ligne['couverture']:.3f} | "
              f"{ligne['exactitude_acceptes']:.3f} | "
              f"{ligne['erreurs_evitees']} | {ligne['legitimes_perdues']} | "
              f"{ligne['flux_a_tort']} |")

    if recommande:
        detail = recommande["detail"]
        print()
        print(f"Seuil retenu : {recommande['seuil']:.3f}")
        print(f"  règle : {recommande['regle']}")
        print(f"  couverture           {detail['couverture']:.1%} par ligne, "
              f"{detail['couverture_gabarit']:.1%} par gabarit")
        print(f"  exactitude acceptés  {detail['exactitude_acceptes']:.1%}")
        print(f"  erreurs évitées      {detail['erreurs_evitees']}")
        print(f"  légitimes rejetées   {detail['legitimes_perdues']}")
        print(f"  rendement            {detail['rendement']} erreur(s) évitée(s) "
              f"par demande légitime sacrifiée")
        print(f"  flux à tort restants {detail['flux_a_tort']} "
              f"(contre {lignes[0]['flux_a_tort']} sans seuil)")

    resultats = {
        "n_lignes": donnees["n"],
        "n_gabarits": len(set(donnees["gabarit"])),
        "calibration": calib,
        "recommandation": recommande,
        "balayage": lignes,
    }
    chemin_sortie = RACINE / args.sortie
    chemin_sortie.write_text(
        json.dumps(resultats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[ok] résultats écrits dans {args.sortie}")


if __name__ == "__main__":
    main()