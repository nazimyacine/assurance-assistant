"""Étape 7 : recherche hybride sur les chunks du corpus.

Deux moteurs fusionnés par Reciprocal Rank Fusion :
- vectoriel : e5 + pgvector, robuste aux fautes et aux reformulations
- lexical : BM25 (rank_bm25) en mémoire, précis sur les termes exacts
  et les montants

Le score de pertinence exposé pour le futur SEUIL_PERTINENCE du routeur
est le score cosinus du meilleur chunk, pas le score RRF : l'échelle RRF
(environ 0,033 au maximum) n'a pas de sens absolu.

Contrôle : python service\\retrieval.py --requete "..."
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

RACINE = Path(__file__).resolve().parents[1]
RRF_K = 60          # constante de la fusion, 1 / (60 + rang), non réglée
CANDIDATS = 20      # profondeur des listes avant fusion


def lire_env() -> dict[str, str]:
    """Lit le .env à la racine du dépôt (format CLE=valeur)."""
    valeurs: dict[str, str] = {}
    fichier = RACINE / ".env"
    if fichier.exists():
        for ligne in fichier.read_text(encoding="utf-8").splitlines():
            ligne = ligne.strip()
            if ligne and not ligne.startswith("#") and "=" in ligne:
                cle, _, valeur = ligne.partition("=")
                valeurs[cle.strip()] = valeur.strip()
    return valeurs


ENV = lire_env()
DSN = (f"postgresql://{ENV.get('POSTGRES_USER', 'assurance')}"
       f":{ENV.get('POSTGRES_PASSWORD', 'assurance')}"
       f"@localhost:{ENV.get('POSTGRES_PORT', '5433')}"
       f"/{ENV.get('POSTGRES_DB', 'assurance')}")


def tokeniser(texte: str) -> list[str]:
    """Minuscules, accents retirés, mots et nombres. Le retrait des
    accents aligne les questions en style SMS sur le corpus accentué."""
    texte = unicodedata.normalize("NFKD", texte.lower())
    texte = "".join(c for c in texte if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9]+", texte)


class Recherche:
    """Charge une variante de chunks et sert les trois modes de recherche.

    Tout est chargé une fois à la construction : les chunks, l'index
    BM25 et le modèle d'embeddings. Le service FastAPI construira un
    seul objet Recherche au démarrage.
    """

    def __init__(self, decoupage: str = "titres", avec_fil: bool = True):
        self.decoupage, self.avec_fil = decoupage, avec_fil
        self.modele = SentenceTransformer("intfloat/multilingual-e5-base")
        with psycopg.connect(DSN) as conn:
            lignes = conn.execute(
                """SELECT id, doc_id, section, texte FROM chunks
                   WHERE decoupage = %s AND avec_fil = %s
                   ORDER BY id""", (decoupage, avec_fil)).fetchall()
        if not lignes:
            raise SystemExit(f"ERREUR : aucun chunk pour decoupage={decoupage}, "
                             f"avec_fil={avec_fil} ; lancer ml/index_corpus.py")
        self.chunks = [{"id": i, "doc_id": d, "section": s, "texte": t}
                       for i, d, s, t in lignes]
        self.par_id = {c["id"]: c for c in self.chunks}
        self.bm25 = BM25Okapi([tokeniser(c["texte"]) for c in self.chunks])

    def vectoriel(self, question: str, k: int = CANDIDATS) -> list[tuple[int, float]]:
        """Liste (id, score cosinus) triée, via l'index HNSW de Postgres."""
        vecteur = self.modele.encode(f"query: {question}",
                                     normalize_embeddings=True)
        with psycopg.connect(DSN) as conn:
            register_vector(conn)
            lignes = conn.execute(
                """SELECT id, 1 - (embedding <=> %s) AS score FROM chunks
                   WHERE decoupage = %s AND avec_fil = %s
                   ORDER BY embedding <=> %s LIMIT %s""",
                (vecteur, self.decoupage, self.avec_fil, vecteur, k)).fetchall()
        return [(id_, float(score)) for id_, score in lignes]

    def lexical(self, question: str, k: int = CANDIDATS) -> list[tuple[int, float]]:
        """Liste (id, score BM25) triée, calculée en mémoire."""
        scores = self.bm25.get_scores(tokeniser(question))
        ordre = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i]["id"], float(scores[i])) for i in ordre]

    def hybride(self, question: str, k: int = 5) -> list[dict]:
        """Fusion RRF des deux listes. Retourne les k meilleurs chunks,
        chacun avec son score RRF, son score cosinus (si présent dans la
        liste vectorielle) et son rang dans chaque liste."""
        vec = self.vectoriel(question)
        lex = self.lexical(question)
        rrf: dict[int, float] = {}
        for liste in (vec, lex):
            for rang, (id_, _) in enumerate(liste):
                rrf[id_] = rrf.get(id_, 0.0) + 1.0 / (RRF_K + rang + 1)
        rang_vec = {id_: r for r, (id_, _) in enumerate(vec)}
        rang_lex = {id_: r for r, (id_, _) in enumerate(lex)}
        score_vec = dict(vec)
        meilleurs = sorted(rrf.items(), key=lambda x: -x[1])[:k]
        return [{**self.par_id[id_],
                 "score_rrf": score,
                 "score_cosinus": score_vec.get(id_),
                 "rang_vectoriel": rang_vec.get(id_),
                 "rang_lexical": rang_lex.get(id_)}
                for id_, score in meilleurs]


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("--requete", required=True)
    parseur.add_argument("--decoupage", default="titres",
                         choices=["naif", "titres"])
    parseur.add_argument("--sans-fil", action="store_true")
    args = parseur.parse_args()

    recherche = Recherche(args.decoupage, avec_fil=not args.sans_fil)
    print(f"\nrequête : {args.requete}")
    print(f"{'rrf':>7} {'cos':>6} {'vec':>4} {'lex':>4}  document / section")
    for c in recherche.hybride(args.requete):
        cos = f"{c['score_cosinus']:.3f}" if c["score_cosinus"] is not None else "  -  "
        rv = "-" if c["rang_vectoriel"] is None else c["rang_vectoriel"] + 1
        rl = "-" if c["rang_lexical"] is None else c["rang_lexical"] + 1
        print(f"{c['score_rrf']:.4f} {cos:>6} {rv:>4} {rl:>4}  "
              f"{c['doc_id']} > {c['section']}")


if __name__ == "__main__":
    main()