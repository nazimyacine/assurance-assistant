"""Vérifie que le corpus ne contredit pas le référentiel de l'offre.

Deux contrôles :
  1. couverture   : chaque montant du référentiel apparaît dans le
                    document de garanties correspondant
  2. intrusion    : chaque montant présent dans un document de garanties
                    existe bien dans le référentiel pour cette formule
"""

import re
import sys
from pathlib import Path

import yaml

RACINE = Path(__file__).resolve().parents[2]
REFERENTIEL = RACINE / "data" / "reference" / "offre.yaml"
CORPUS = RACINE / "data" / "raw" / "corpus"

DOC_PAR_FORMULE = {
    "essentiel": "02-garanties-essentiel.md",
    "confort": "03-garanties-confort.md",
    "premium": "04-garanties-premium.md",
}

# "150% BR", "100 €", "30 jours"
MOTIF_MONTANT = re.compile(r"(\d+)\s*(%|€|euros|jours?)", re.IGNORECASE)


def montants_attendus(valeur: str) -> set[str]:
    """Extrait les nombres d'une valeur du référentiel. '50 € par jour, 30
    jours maximum' donne {'50', '30'}."""
    return {m.group(1) for m in MOTIF_MONTANT.finditer(str(valeur))}


def montants_presents(texte: str) -> set[str]:
    return {m.group(1) for m in MOTIF_MONTANT.finditer(texte)}


def main() -> int:
    offre = yaml.safe_load(REFERENTIEL.read_text(encoding="utf-8"))
    anomalies = []

    for formule, nom_fichier in DOC_PAR_FORMULE.items():
        chemin = CORPUS / nom_fichier
        if not chemin.exists():
            anomalies.append(f"[manquant] {nom_fichier}")
            continue

        texte = chemin.read_text(encoding="utf-8")
        presents = montants_presents(texte)
        attendus_total = set()

        # 1. couverture
        for cle, garantie in offre["garanties"].items():
            valeur = garantie[formule]
            attendus = montants_attendus(valeur)
            attendus_total |= attendus
            manquants = attendus - presents
            if manquants:
                anomalies.append(
                    f"[couverture] {nom_fichier} : {cle} attend "
                    f"{sorted(manquants)} (référentiel : « {valeur} »)"
                )

        # 2. intrusion
        intrus = presents - attendus_total - {"16", "2026", "100"}
        if intrus:
            anomalies.append(
                f"[intrusion] {nom_fichier} : montants absents du "
                f"référentiel {sorted(intrus)}"
            )

    if anomalies:
        print(f"{len(anomalies)} anomalie(s) :\n")
        for a in anomalies:
            print("  " + a)
        return 1

    print("Corpus cohérent avec le référentiel.")
    return 0


if __name__ == "__main__":
    sys.exit(main())