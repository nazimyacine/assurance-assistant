"""Étape 5d : génération de docs/metrics.md à partir des résultats chiffrés.

Commit 1 : chargement des sources et garde-fous. Le script charge tout,
vérifie la cohérence, n'écrit rien. Lancer avec --check pour un état des lieux.

Principe : metrics.md sera GÉNÉRÉ, jamais rédigé. Toute valeur affichée
provient d'un fichier source, aucun chiffre n'est écrit en dur ici.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parents[1]
DOCS = RACINE / "docs"
ARTIFACTS = RACINE / "ml" / "artifacts"

SOURCES_JSON = {
    "baseline": DOCS / "baseline.json",
    "training": DOCS / "training.json",
    "eval_intent": DOCS / "eval_intent.json",
    "confusion": DOCS / "confusion.json",
    "seuil_rejet": DOCS / "seuil_rejet.json",
}
NPZ_CAMEMBERT = ARTIFACTS / "predictions_test.npz"
NPZ_BASELINE = ARTIFACTS / "predictions_baseline_test.npz"
CSV_TEST = RACINE / "data" / "eval" / "intents_test.csv"
LABELS = ARTIFACTS / "intent-model" / "labels.json"
CAS_AMBIGUS = RACINE / "data" / "eval" / "cas_ambigus.md"


def echec(message: str) -> None:
    raise SystemExit(f"ERREUR : {message}")


def charger_labels() -> dict[int, str]:
    """labels.json est imbriqué : {"vers_id": {...}, "vers_nom": {...}}."""
    contenu = json.loads(LABELS.read_text(encoding="utf-8"))
    if "vers_nom" not in contenu:
        echec(f"forme inattendue de {LABELS} : clé 'vers_nom' absente "
              f"(clés : {sorted(contenu)})")
    return {int(k): v for k, v in contenu["vers_nom"].items()}


def en_noms(tableau: np.ndarray, vers_nom: dict[int, str], origine: str) -> np.ndarray:
    """Convertit un tableau d'étiquettes en noms, qu'il soit numérique ou textuel."""
    if np.issubdtype(tableau.dtype, np.number):
        return np.array([vers_nom[int(i)] for i in tableau])
    return tableau.astype(str)


def charger_sources() -> dict:
    manquants = [str(p) for p in [*SOURCES_JSON.values(), NPZ_CAMEMBERT,
                 NPZ_BASELINE, CSV_TEST, LABELS, CAS_AMBIGUS] if not p.exists()]
    if manquants:
        echec("fichier(s) source absent(s) :\n  " + "\n  ".join(manquants))

    src = {nom: json.loads(chemin.read_text(encoding="utf-8"))
           for nom, chemin in SOURCES_JSON.items()}
    vers_nom = charger_labels()

    npz_c = np.load(NPZ_CAMEMBERT, allow_pickle=True)
    npz_b = np.load(NPZ_BASELINE, allow_pickle=True)
    test = pd.read_csv(CSV_TEST)

    src["textes"] = npz_c["textes"].astype(str)
    src["probas"] = npz_c["probas"]
    # "etiquettes" est le vocabulaire des classes (ordre des colonnes de
    # probas), pas un vecteur de prédictions par ligne.
    src["classes"] = npz_c["etiquettes"].astype(str)
    src["camembert_predit"] = src["classes"][npz_c["probas"].argmax(axis=1)]
    src["reels"] = en_noms(npz_c["reels"], vers_nom, "camembert")
    src["baseline_predit"] = en_noms(npz_b["predits"], vers_nom, "baseline")
    src["baseline_reels"] = en_noms(npz_b["reels"], vers_nom, "baseline")
    src["baseline_variante"] = str(np.atleast_1d(npz_b["variante"])[0])
    src["textes_baseline"] = npz_b["textes"].astype(str)
    src["test"] = test
    src["vers_nom"] = vers_nom
    src["cas_ambigus"] = CAS_AMBIGUS.read_text(encoding="utf-8")
    return src


def verifier(src: dict) -> None:
    """Garde-fous. Chaque hypothèse d'alignement est vérifiée, aucune supposée."""
    textes, test = src["textes"], src["test"]

    # 1. Les deux npz portent sur les mêmes phrases, dans le même ordre.
    if not np.array_equal(textes, src["textes_baseline"]):
        echec("les vecteurs de textes des deux .npz diffèrent, "
              "comparaison appariée impossible")

    # 2. Le npz est aligné ligne à ligne avec intents_test.csv,
    #    condition pour récupérer la colonne gabarit par position.
    colonnes_attendues = {"texte", "intention", "difficulte", "origine", "gabarit"}
    if not colonnes_attendues <= set(test.columns):
        echec(f"colonnes manquantes dans {CSV_TEST} : "
              f"{sorted(colonnes_attendues - set(test.columns))}")
    if len(test) != len(textes):
        echec(f"tailles différentes : csv {len(test)}, npz {len(textes)}")
    if not np.array_equal(test["texte"].astype(str).to_numpy(), textes):
        echec("intents_test.csv et predictions_test.npz ne sont pas dans "
              "le même ordre, jonction par position impossible")

    # 3. Les étiquettes réelles concordent partout.
    if not np.array_equal(src["reels"], src["baseline_reels"]):
        echec("étiquettes réelles différentes entre les deux .npz")
    if not np.array_equal(src["reels"], test["intention"].astype(str).to_numpy()):
        echec("étiquettes réelles du npz différentes de celles du csv")

    # 4. La légende des colonnes de probas concorde avec labels.json.
    classes = src["classes"]
    if len(classes) != src["probas"].shape[1]:
        echec(f"{len(classes)} noms de classes pour "
              f"{src['probas'].shape[1]} colonnes de probas")
    attendu = np.array([src["vers_nom"][i] for i in range(len(classes))])
    if not np.array_equal(classes, attendu):
        echec("l'ordre des classes du npz diffère de labels.json :\n"
              f"  npz    : {classes.tolist()}\n"
              f"  labels : {attendu.tolist()}")

    # 5. Recoupement avec confusion.json : les comptes d'erreurs recalculés
    #    doivent être identiques à ceux de l'étape 5b.
    for modele, predit in [("camembert", src["camembert_predit"]),
                           ("baseline", src["baseline_predit"])]:
        recalcule = int((predit != src["reels"]).sum())
        reference = src["confusion"]["modeles"][modele]["routage"]["erreurs"]
        if recalcule != reference:
            echec(f"{modele} : {recalcule} erreurs recalculées contre "
                  f"{reference} dans confusion.json")

    # 6. Recoupement avec eval_intent.json sur l'exactitude globale.
    exactitude = float((src["camembert_predit"] == src["reels"]).mean())
    reference = src["eval_intent"]["global"]["exactitude"]
    if abs(exactitude - reference) > 5e-4:
        echec(f"exactitude recalculée {exactitude:.4f} contre "
              f"{reference} dans eval_intent.json")


def etat_des_lieux(src: dict) -> None:
    test = src["test"]
    erreurs = src["camembert_predit"] != src["reels"]
    groupes = (test.loc[erreurs]
               .assign(predit=src["camembert_predit"][erreurs])
               .groupby(["gabarit", "intention", "predit"])
               .size().sort_values(ascending=False))
    print(f"lignes            : {len(test)}")
    print(f"unités (gabarits + ambigus) : {test['gabarit'].nunique()}")
    print(f"erreurs camembert : {int(erreurs.sum())}")
    print(f"erreurs baseline  : {int((src['baseline_predit'] != src['reels']).sum())}")
    print(f"variante baseline : {src['baseline_variante']}")
    print(f"groupes d'erreurs (gabarit, réel, prédit) : {len(groupes)}")
    print("\naperçu des 5 premiers groupes :")
    print(groupes.head(5).to_string())


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("--check", action="store_true",
                         help="charge, vérifie, affiche un état des lieux")
    args = parseur.parse_args()

    src = charger_sources()
    verifier(src)
    print("Garde-fous : OK, sources chargées et alignées.")
    if args.check:
        print()
        etat_des_lieux(src)


if __name__ == "__main__":
    main()