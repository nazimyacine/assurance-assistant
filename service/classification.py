"""Étape 9 : classification d'intentions en service.

Version unitaire du classifieur évalué à l'étape 5 : le modèle est chargé
une fois, puis répond message par message. La longueur maximale et le
traitement des logits sont IDENTIQUES à ml/evaluate_intent.py ; s'en
écarter invaliderait silencieusement les chiffres publiés (F1 macro
0,8425, exactitude 0,8787).

Ce module ne décide rien : il rend une intention et une confiance. Le
seuil de rejet et le routage appartiennent à service/router.py.

Contrôle : python -m service.classification --message "..."
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

RACINE = Path(__file__).resolve().parents[1]
MODELE_PAR_DEFAUT = RACINE / "ml" / "artifacts" / "intent-model"

# Identique à predire() dans ml/evaluate_intent.py. Une valeur différente
# tronquerait autrement les messages longs et donnerait en service des
# prédictions que l'évaluation n'a jamais mesurées.
LONGUEUR_MAX = 128


# ---------------------------------------------------------------------------
# Étiquettes
# ---------------------------------------------------------------------------
# Repris de ml/evaluate_intent.py plutôt qu'importé : le service ne doit
# pas dépendre de ml/, qui tire pandas, scikit-learn et matplotlib.

def _candidats_etiquettes(objet):
    """Toutes les listes d'étiquettes plausibles contenues dans un JSON,
    quelle que soit sa forme (liste, dictionnaire dans un sens ou dans
    l'autre, structure imbriquée dans laquelle on descend)."""
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


def charger_etiquettes(model_dir: Path, modele) -> list[str]:
    """Liste des intentions dans l'ordre des sorties du modèle.

    Un mauvais ordre produirait un service entièrement faux mais crédible.
    On n'accepte donc qu'une liste sans doublon dont la longueur
    correspond exactement au nombre de sorties.
    """
    attendu = int(getattr(modele.config, "num_labels", 0)) or None

    sources = []
    chemin = model_dir / "labels.json"
    if chemin.exists():
        sources.append(("labels.json", json.loads(chemin.read_text(encoding="utf-8"))))
    id2label = getattr(modele.config, "id2label", {}) or {}
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
            return [str(e) for e in candidat]

    raise SystemExit(
        f"Étiquettes exploitables introuvables dans {model_dir}. "
        f"Le modèle a {attendu} sorties, fichiers examinés : "
        f"{[nom for nom, _ in sources] or 'aucun'}."
    )


# ---------------------------------------------------------------------------

class Classifieur:
    """Charge CamemBERT une fois, classe un message à la fois.

    Le service FastAPI construira un seul objet au démarrage : le
    chargement coûte environ 4 s, la prédiction environ 25 ms.
    """

    def __init__(self, model_dir: Path | str = MODELE_PAR_DEFAUT,
                 device: str | None = None):
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise SystemExit(f"ERREUR : modèle absent de {model_dir} ; "
                             f"lancer ml/train_intent.py")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.modele = AutoModelForSequenceClassification.from_pretrained(
            model_dir).to(self.device)
        self.modele.eval()
        self.etiquettes = charger_etiquettes(model_dir, self.modele)
        # Tour à blanc : la première inférence CUDA paie l'initialisation
        # des noyaux. Sans lui, la latence affichée pour le premier
        # message de la journée serait fausse d'un facteur dix.
        self.predire("bonjour")

    @torch.no_grad()
    def predire(self, message: str, n_top: int = 3) -> dict:
        """Retourne l'intention la plus probable, sa confiance, les n_top
        meilleures pour l'inspection, et la latence en millisecondes."""
        debut = time.perf_counter()
        encodage = self.tokenizer(
            message, truncation=True, max_length=LONGUEUR_MAX,
            return_tensors="pt").to(self.device)
        if self.device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = self.modele(**encodage).logits
        else:
            logits = self.modele(**encodage).logits
        probas = torch.softmax(logits.float(), dim=-1)[0]
        valeurs, indices = torch.topk(probas, k=min(n_top, probas.numel()))
        top = [(self.etiquettes[int(i)], float(v))
               for v, i in zip(valeurs, indices)]
        return {"intention": top[0][0],
                "confiance": top[0][1],
                "top": top,
                "latence_ms": round((time.perf_counter() - debut) * 1000)}


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("--message", required=True)
    parseur.add_argument("--model-dir", default=str(MODELE_PAR_DEFAUT))
    args = parseur.parse_args()

    debut = time.perf_counter()
    classifieur = Classifieur(args.model_dir)
    charge = round((time.perf_counter() - debut) * 1000)
    print(f"[ok] {len(classifieur.etiquettes)} intentions, device "
          f"{classifieur.device}, chargé en {charge} ms")
    print(f"     ordre : {classifieur.etiquettes[0]}, "
          f"{classifieur.etiquettes[1]}, ..., {classifieur.etiquettes[-1]}")

    resultat = classifieur.predire(args.message)
    print(f"\nmessage : {args.message}")
    print(f"\n{'proba':>6}  intention")
    for nom, proba in resultat["top"]:
        print(f"{proba:>6.3f}  {nom}")
    print(f"\nretenue : {resultat['intention']} "
          f"(confiance {resultat['confiance']:.3f}, "
          f"{resultat['latence_ms']} ms)")


if __name__ == "__main__":
    main()