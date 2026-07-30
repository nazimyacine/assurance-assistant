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

# Délais d'attente en secondes, (connexion, lecture).
#
# RÈGLE : les délais doivent CROÎTRE du plus interne vers le plus
# externe. Ici 5 et 10 secondes, sous les 15 secondes de lecture de la
# passerelle Spring, elles-mêmes sous les 20 secondes du front. Un délai
# interne plus long que celui qui l'englobe est pire qu'inutile : la
# couche supérieure abandonne d'abord et rend une erreur brute, tandis
# que la dégradation soignée de ce module s'exécute pour personne.
#
# Découvert à l'usage : avec 60 secondes de connexion et une panne DNS,
# la passerelle rendait un 503 nu là où le routeur aurait affiché un
# message lisible. 10 secondes de lecture couvrent largement la
# génération mesurée à l'étape 8, entre 838 et 1600 ms.
DELAIS = (5, 10)

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
    """Appelle le LLM et retourne {"reponse": str, "latence_ms": int}.

    Toute panne remonte en Exception ordinaire : le routeur l'attrape,
    la journalise et dégrade. SystemExit serait ici un piège, la classe
    héritant de BaseException et traversant donc le `except Exception`
    du routeur au lieu d'y être traitée.
    """
    if not CLE and "mistral" in BASE_URL:
        raise RuntimeError("MISTRAL_API_KEY absente du .env")
    debut = time.perf_counter()
    http = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {CLE}"},
        json={"model": MODELE,
              "messages": construire_prompt(question, chunks),
              "temperature": 0.2,
              "max_tokens": 400},
        timeout=DELAIS)
    http.raise_for_status()
    texte = http.json()["choices"][0]["message"]["content"]
    return {"reponse": texte,
            "latence_ms": round((time.perf_counter() - debut) * 1000)}


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("--question", required=True)
    args = parseur.parse_args()

    # Configuration de service, identique à celle du routeur : le
    # contrôle de ce module doit exercer ce que le service exécute, pas
    # une autre configuration. L'hybride a été mesuré perdant à l'étape 8.
    recherche = Recherche(decoupage="titres", avec_fil=True)
    chunks = recherche.vectoriel_chunks(args.question, k=5)
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