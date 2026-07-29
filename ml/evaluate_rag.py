"""Étape 8, volet recherche : les 4 configurations sur les 50 questions.

1. naïf, vectoriel seul          3. titres, hybride
2. titres, vectoriel seul        4. titres + fil d'Ariane, hybride

Métriques : recall@5 et MRR@10 sur les questions à réponse. Les questions
sans réponse ne comptent pas dans ces métriques mais leurs scores de tête
sont mesurés : c'est la matière du futur SEUIL_PERTINENCE du routeur.

Correspondance des chunks naïfs (fenetre-N, sans section) : une fenêtre
vaut le chunk attendu si elle recouvre au moins la moitié des mots de la
section attendue, par intervalles de positions dans le texte du document.

50 questions = 50 unités indépendantes : les taux sont publiés avec leurs
effectifs, sans fausse précision.

Écrit docs/eval_rag.json. La génération LLM viendra dans un second volet.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests
import yaml

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE))

from ml.index_corpus import (CHEVAUCHEMENT, CORPUS, FENETRE_NAIF,  # noqa: E402
                             decouper_titres, lire_document)
from service.retrieval import Recherche  # noqa: E402

QUESTIONS = RACINE / "data" / "eval" / "rag_questions.yaml"
SORTIE = RACINE / "docs" / "eval_rag.json"
PAS = FENETRE_NAIF - CHEVAUCHEMENT

PAUSE_S = 2.0   # palier gratuit Mistral, marge sous la limite de débit
CACHE = RACINE / "ml" / "artifacts" / "rag_generations.json"
VERDICTS = ("correct", "incorrect", "refus")

CONFIGS = [
    ("naif, vectoriel", "naif", False, "vectoriel"),
    ("titres, vectoriel", "titres", False, "vectoriel"),
    ("titres + fil, vectoriel", "titres", True, "vectoriel"),
    ("titres, hybride", "titres", False, "hybride"),
    ("titres + fil, hybride", "titres", True, "hybride"),
]


def charger_questions() -> list[dict]:
    questions = yaml.safe_load(QUESTIONS.read_text(encoding="utf-8"))
    if len(questions) != 50:
        raise SystemExit(f"ERREUR : {len(questions)} questions, 50 attendues")
    if len({q["id"] for q in questions}) != 50:
        raise SystemExit("ERREUR : identifiants de questions non uniques")
    for q in questions:
        sans = q["difficulte"] == "sans_reponse"
        if sans != (q["chunk_attendu"] is None):
            raise SystemExit(f"ERREUR : {q['id']} incohérente entre "
                             "difficulte et chunk_attendu")
    return questions


def verifier_references(questions: list[dict], rech: Recherche) -> None:
    """Chaque (doc_id, section) attendu doit exister dans l'index. Un
    titre mal recopié ferait un échec fantôme silencieux."""
    existants = {(c["doc_id"], c["section"]) for c in rech.chunks}
    inconnus = [q["id"] for q in questions if q["chunk_attendu"] and
                (q["chunk_attendu"]["doc_id"],
                 q["chunk_attendu"]["section"]) not in existants]
    if inconnus:
        raise SystemExit("ERREUR : chunk_attendu introuvable dans l'index "
                         f"pour : {', '.join(inconnus)}")


def spans_sections() -> dict[tuple[str, str], tuple[int, int]]:
    """Position (début, fin) en mots de chaque section dans le texte sans
    titres de son document, le même texte que celui des fenêtres naïves."""
    spans: dict[tuple[str, str], tuple[int, int]] = {}
    for chemin in sorted(CORPUS.glob("*.md")):
        meta, corps = lire_document(chemin)
        position = 0
        for section, texte in decouper_titres(meta, corps):
            n = len(texte.split())
            spans[(meta["doc_id"], section)] = (position, position + n)
            position += n
    return spans


def top_chunks(rech: Recherche, question: str, mode: str, k: int = 10) -> list[dict]:
    if mode == "vectoriel":
        return [rech.par_id[id_] for id_, _ in rech.vectoriel(question, k)]
    return rech.hybride(question, k)


def rang_attendu(chunks: list[dict], attendu: dict, decoupage: str,
                 spans: dict) -> int | None:
    """Rang (0 = premier) du chunk attendu dans la liste, sinon None."""
    cible = (attendu["doc_id"], attendu["section"])
    if decoupage == "titres":
        for rang, c in enumerate(chunks):
            if (c["doc_id"], c["section"]) == cible:
                return rang
        return None
    a_debut, a_fin = spans[cible]
    for rang, c in enumerate(chunks):
        if c["doc_id"] != attendu["doc_id"]:
            continue
        fenetre = int(c["section"].split("-")[1])
        debut, fin = fenetre * PAS, fenetre * PAS + FENETRE_NAIF
        recouvrement = max(0, min(fin, a_fin) - max(debut, a_debut))
        if recouvrement >= 0.5 * (a_fin - a_debut):
            return rang
    return None

def appel_juge(question: str, attendue: str, reponse: str) -> str:
    """Fait juger une réponse par le LLM : correct, incorrect ou refus.
    Le juge est le même modèle que le générateur, limite assumée et
    notée dans limites.md."""
    from service.generation import BASE_URL, CLE, MODELE

    consignes = (
        "Tu évalues la réponse d'un assistant d'assurance santé. "
        "Réponds par UN SEUL mot, exactement : correct, incorrect ou refus.\n"
        "- refus : la réponse dit que l'information n'est pas disponible, "
        "ou renvoie vers un conseiller sans donner l'information demandée\n"
        "- correct : la réponse donne le fait attendu, chiffres et "
        "conditions compris\n"
        "- incorrect : tout le reste, fait faux, chiffre différent, "
        "réponse à côté de la question")
    contenu = (f"Question : {question}\n\n"
               f"Fait attendu : {attendue}\n\n"
               f"Réponse à évaluer : {reponse}")
    for tentative in range(6):
        http = requests.post(
            f"{BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {CLE}"},
            json={"model": MODELE,
                  "messages": [{"role": "system", "content": consignes},
                               {"role": "user", "content": contenu}],
                  "temperature": 0.0, "max_tokens": 5},
            timeout=60)
        if http.status_code == 429:
            attente = 15 * (tentative + 1)
            print(f"  débit limité, pause {attente} s")
            time.sleep(attente)
            continue
        http.raise_for_status()
        mot = http.json()["choices"][0]["message"]["content"].strip().lower()
        for verdict in VERDICTS:
            if mot.startswith(verdict):
                return verdict
        return "incorrect"
    raise SystemExit("ERREUR : limite de débit Mistral persistante")


def volet_generation(questions: list[dict], recherches: dict,
                     resultats: dict, filtre: str) -> None:
    """Génère et juge les réponses, configuration par configuration.
    Chaque appel est mis en cache dans ml/artifacts (non versionné) :
    une interruption ne coûte que l'appel en cours."""
    from service.generation import generer

    cache = (json.loads(CACHE.read_text(encoding="utf-8"))
             if CACHE.exists() else {})
    resultats["generation"] = {}
    print("\n| Configuration | réponses correctes | refus corrects | faux refus |")
    print("|---|---|---|---|")
    for nom, decoupage, fil, mode in CONFIGS:
        if filtre and filtre not in nom:
            continue
        rech = recherches[(decoupage, fil)]
        verdicts = {}
        for q in questions:
            cle = f"{nom}|{q['id']}"
            if cle not in cache:
                for tentative in range(4):
                    try:
                        chunks = top_chunks(rech, q["question"], mode, k=5)
                        sortie = generer(q["question"], chunks)
                        time.sleep(PAUSE_S)
                        attendue = (q["reponse_attendue"] or
                                    "AUCUN : l'information n'est pas dans la "
                                    "documentation, la bonne conduite est un refus")
                        sortie["verdict"] = appel_juge(q["question"], attendue,
                                                      sortie["reponse"])
                        time.sleep(PAUSE_S)
                        break
                    except requests.RequestException as erreur:
                        if tentative == 3:
                            raise
                        print(f"  réseau instable sur {q['id']} "
                              f"({type(erreur).__name__}), nouvel essai dans 15 s")
                        time.sleep(15)
                cache[cle] = sortie
                CACHE.write_text(json.dumps(cache, ensure_ascii=False,
                                            indent=2), encoding="utf-8")
            verdicts[q["id"]] = cache[cle]["verdict"]
        avec = [q for q in questions if q["chunk_attendu"]]
        sans = [q for q in questions if not q["chunk_attendu"]]
        correctes = sum(1 for q in avec if verdicts[q["id"]] == "correct")
        faux_refus = sum(1 for q in avec if verdicts[q["id"]] == "refus")
        refus_ok = sum(1 for q in sans if verdicts[q["id"]] == "refus")
        resultats["generation"][nom] = {
            "reponses_correctes": correctes, "sur": len(avec),
            "faux_refus": faux_refus,
            "refus_corrects": refus_ok, "sur_sans": len(sans),
            "verdicts": verdicts}
        print(f"| {nom} | {correctes}/{len(avec)} | {refus_ok}/{len(sans)} "
              f"| {faux_refus} |")


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("--generation", action="store_true",
                         help="ajoute le volet LLM : réponses et refus jugés")
    parseur.add_argument("--config", default="",
                         help="limite le volet génération aux configurations "
                              "dont le nom contient ce texte")
    args = parseur.parse_args()

    questions = charger_questions()
    spans = spans_sections()
    avec_reponse = [q for q in questions if q["chunk_attendu"]]
    sans_reponse = [q for q in questions if not q["chunk_attendu"]]
    print(f"{len(questions)} questions : {len(avec_reponse)} à réponse, "
          f"{len(sans_reponse)} sans réponse")

    recherches = {(d, f): Recherche(d, avec_fil=f)
                  for _, d, f, _ in CONFIGS for d, f in [(d, f)]}
    verifier_references(questions, recherches[("titres", False)])

    resultats: dict = {"configs": {}, "seuil_pertinence": {}}
    print(f"\n| Configuration | recall@5 | MRR@10 |")
    print("|---|---|---|")
    for nom, decoupage, fil, mode in CONFIGS:
        rech = recherches[(decoupage, fil)]
        rangs = {}
        for q in avec_reponse:
            chunks = top_chunks(rech, q["question"], mode)
            rangs[q["id"]] = rang_attendu(chunks, q["chunk_attendu"],
                                          decoupage, spans)
        touches = sum(1 for r in rangs.values() if r is not None and r < 5)
        mrr = sum(1 / (r + 1) for r in rangs.values() if r is not None)
        mrr /= len(avec_reponse)
        resultats["configs"][nom] = {
            "recall_a_5": touches, "sur": len(avec_reponse),
            "mrr_a_10": round(mrr, 4), "rangs": rangs}
        print(f"| {nom} | {touches}/{len(avec_reponse)} | {mrr:.3f} |")

    rech = recherches[("titres", True)]
    print("\nMeilleur cosinus par difficulté (variante titres + fil) :")
    for groupe in ["facile", "nuance", "sans_reponse"]:
        sous_ensemble = [q for q in questions if q["difficulte"] == groupe]
        scores = [rech.vectoriel(q["question"], 1)[0][1] for q in sous_ensemble]
        resultats["seuil_pertinence"][groupe] = {
            "n": len(scores), "min": round(min(scores), 4),
            "moyenne": round(sum(scores) / len(scores), 4),
            "max": round(max(scores), 4),
            "scores": {q["id"]: round(s, 4)
                       for q, s in zip(sous_ensemble, scores)}}
        r = resultats["seuil_pertinence"][groupe]
        print(f"  {groupe:13s} n={r['n']:2d}  min {r['min']:.3f}  "
              f"moyenne {r['moyenne']:.3f}  max {r['max']:.3f}")

    if args.generation:
        volet_generation(questions, recherches, resultats, args.config)

    SORTIE.write_text(json.dumps(resultats, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(f"\nÉcrit : {SORTIE.relative_to(RACINE)}")


if __name__ == "__main__":
    main()