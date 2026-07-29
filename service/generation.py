"""Étape 7 : génération de la réponse à partir des chunks retenus.

Le fournisseur est interchangeable : tout endpoint au format OpenAI
chat/completions convient. Mistral par défaut ; Ollama expose le même
format en local sur http://localhost:11434/v1 (alternative documentée
au README). Surcharger via LLM_BASE_URL et LLM_MODELE dans le .env.

Le garde-fou de pertinence (ne pas appeler le LLM si le meilleur score
est trop bas) appartient au routeur de l'étape 9, pas à ce module.

Contrôle : python -m service.generation --question "..."
"""

from __future__ import annotations

import argparse
import time

import requests

from service.retrieval import ENV, Recherche

BASE_URL = ENV.get("LLM_BASE_URL", "https://api.mistral.ai/v1")
MODELE = ENV.get("LLM_MODELE", "mistral-small-latest")
CLE = ENV.get("MISTRAL_API_KEY", "")

CONSIGNES = """Tu es l'assistant de Mutuelle Solstice, une complémentaire santé.
Tu réponds à la question d'un assuré en t'appuyant UNIQUEMENT sur les
extraits fournis. Règles impératives :
- si l'information demandée n'est pas dans les extraits, dis clairement
  que tu ne disposes pas de cette information et propose de contacter un
  conseiller ; ne réponds jamais de mémoire
- n'invente jamais un montant, un taux ou un délai
- cite le document et la section d'où vient chaque information, entre
  parenthèses, par exemple (garanties-confort > Optique, formule Confort)
- réponds en français, en 2 à 5 phrases, sans formule de politesse finale
- vouvoie toujours l'assuré"""


def construire_prompt(question: str, chunks: list[dict]) -> list[dict]:
    extraits = "\n\n".join(
        f"[{c['doc_id']} > {c['section']}]\n{c['texte']}" for c in chunks)
    return [
        {"role": "system", "content": CONSIGNES},
        {"role": "user",
         "content": f"Extraits :\n\n{extraits}\n\nQuestion : {question}"},
    ]


def generer(question: str, chunks: list[dict]) -> dict:
    """Appelle le LLM et retourne {"reponse": str, "latence_ms": int}."""
    if not CLE and "mistral" in BASE_URL:
        raise SystemExit("ERREUR : MISTRAL_API_KEY absente du .env")
    debut = time.perf_counter()
    http = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {CLE}"},
        json={"model": MODELE,
              "messages": construire_prompt(question, chunks),
              "temperature": 0.2,
              "max_tokens": 400},
        timeout=60)
    http.raise_for_status()
    texte = http.json()["choices"][0]["message"]["content"]
    return {"reponse": texte,
            "latence_ms": round((time.perf_counter() - debut) * 1000)}


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("--question", required=True)
    args = parseur.parse_args()

    recherche = Recherche()
    chunks = recherche.hybride(args.question)
    print(f"\nquestion : {args.question}")
    print("\nchunks retenus :")
    for c in chunks:
        cos = f"{c['score_cosinus']:.3f}" if c["score_cosinus"] is not None else "  -  "
        print(f"  {cos}  {c['doc_id']} > {c['section']}")
    resultat = generer(args.question, chunks)
    print(f"\nréponse ({resultat['latence_ms']} ms) :\n")
    print(resultat["reponse"])


if __name__ == "__main__":
    main()