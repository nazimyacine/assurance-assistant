"""Baseline de classification d'intentions : TF-IDF + régression logistique.

Ce script n'a pas vocation à produire le meilleur modèle. Il produit le
chiffre de référence sans lequel le score de CamemBERT ne veut rien dire.

Deux familles de traits sont combinées :
  - mots et bigrammes de mots : capte le vocabulaire discriminant
    (« résilier », « attestation », « cotisation »)
  - n-grammes de caractères : survit aux fautes de frappe, puisque
    « résilier » et « résilir » partagent la plupart de leurs 4-grammes

Sans les n-grammes de caractères, la baseline s'effondrerait sur le bruit
et la comparaison avec CamemBERT serait trop flatteuse.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.pipeline import FeatureUnion, Pipeline

RACINE = Path(__file__).resolve().parents[1]
TRAIN = RACINE / "data" / "raw" / "intents_train.csv"
TEST = RACINE / "data" / "eval" / "intents_test.csv"
SORTIE = RACINE / "docs" / "baseline.json"


def construire_modele(equilibrer: bool) -> Pipeline:
    traits = FeatureUnion([
        ("mots", TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2,
            sublinear_tf=True, lowercase=True, strip_accents="unicode",
        )),
        ("caracteres", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=2,
            sublinear_tf=True, lowercase=True, strip_accents="unicode",
        )),
    ])
    return Pipeline([
        ("traits", traits),
        ("classifieur", LogisticRegression(
            max_iter=2000,
            class_weight="balanced" if equilibrer else None,
        )),
    ])

def verifier_absence_de_fuite(train, test):
    """Interrompt si une phrase du test apparaît à l'entraînement.

    Une fuite ne se voit pas dans les scores, elle les embellit. Mieux
    vaut un script qui refuse de tourner qu'un chiffre flatteur et faux.
    """
    textes_train = {t.lower().strip() for t in train["texte"]}
    fuites = [t for t in test["texte"] if t.lower().strip() in textes_train]
    if fuites:
        raise SystemExit(
            f"\nFUITE : {len(fuites)} phrase(s) du jeu de test sont "
            f"présentes à l'entraînement.\n"
            f"Exemple : « {fuites[0]} »\n"
            f"Le jeu de test est probablement périmé. Relance "
            f"generate_intents.py puis renomme intents_test_a_relire.csv "
            f"en intents_test.csv."
        )


def main() -> int:
    parseur = argparse.ArgumentParser()
    parseur.add_argument("--test", default=str(TEST))
    args = parseur.parse_args()

    train = pd.read_csv(TRAIN)
    test = pd.read_csv(args.test)
    verifier_absence_de_fuite(train, test)
    print(f"Jeu de test : {len(test)} phrases")

    resultats = {}
    for equilibrer in (False, True):
        nom = "ponderee" if equilibrer else "brute"
        modele = construire_modele(equilibrer)
        modele.fit(train["texte"], train["intention"])
        pred = modele.predict(test["texte"])

        resultats[nom] = {
            "f1_macro": round(f1_score(test["intention"], pred,
                                       average="macro"), 4),
            "exactitude": round(accuracy_score(test["intention"], pred), 4),
        }

        if equilibrer:
            print("\nRapport détaillé, version pondérée :\n")
            print(classification_report(test["intention"], pred, digits=3))

            print("Par niveau de difficulté :")
            for niveau in ("facile", "bruite", "ambigu"):
                masque = test["difficulte"] == niveau
                if masque.sum() == 0:
                    continue
                exact = accuracy_score(test.loc[masque, "intention"],
                                       pred[masque])
                print(f"  {niveau:8} {masque.sum():4} phrases   "
                      f"exactitude {exact:.3f}")
                resultats[nom][f"exactitude_{niveau}"] = round(exact, 4)

    print("\nRésumé")
    for nom, scores in resultats.items():
        print(f"  {nom:9} F1 macro {scores['f1_macro']:.3f}   "
              f"exactitude {scores['exactitude']:.3f}")

    SORTIE.parent.mkdir(parents=True, exist_ok=True)
    SORTIE.write_text(json.dumps(resultats, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    print(f"\nRésultats écrits dans {SORTIE.relative_to(RACINE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())