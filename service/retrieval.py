"""Étape 7 : recherche sur les chunks du corpus.

Deux moteurs fusionnables par Reciprocal Rank Fusion :
- vectoriel : e5 + pgvector, robuste aux fautes et aux reformulations
- lexical : BM25 (rank_bm25) en mémoire, précis sur les termes exacts
  et les montants

La configuration de SERVICE retenue à l'étape 8 est le vectoriel seul
(titres + fil) : sur ce corpus, l'hybride mesure en dessous.

Étape 11 : la recherche vectorielle accepte une formule optionnelle. Elle
restreint alors les résultats aux garanties de cette formule et aux
documents valables pour toutes. Sans ce paramètre, le comportement est
identique à celui qui a produit les chiffres de l'étape 8.

Contrôles :
    python service\\retrieval.py --requete "..."
    python service\\retrieval.py --requete "..." --service
    python service\\retrieval.py --requete "..." --formule confort
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

# Marqueur des chunks valables quelle que soit la formule souscrite,
# c'est-à-dire les documents de procédure. Vérifié au démarrage contre le
# contenu réel de la table plutôt que supposé.
FORMULE_UNIVERSELLE = "toutes"

# Mots vides du français, formes après minuscules et retrait des accents.
# Sans eux, BM25 score "je", "la", "encore" sur tous les chunks et le
# classement lexical d'une question sans vocabulaire commun avec le
# corpus est du bruit confiant (constaté sur le jeu d'évaluation RAG).
MOTS_VIDES = frozenset("""
le la les l un une des de du d au aux et ou mais donc or ni car
que qu qui quoi dont ce cet cette ces c se s ne n pas plus moins
je j tu il elle on nous vous ils elles me m te t
mon ma mes ton ta tes son sa ses notre nos votre vos leur leurs
y en a ai as avons avez ont est es suis sommes etes sont etre avoir
pour par avec sans dans sur sous chez vers apres avant pendant
si comme quand tres tout toute tous toutes meme aussi bien encore deja
peux peut pouvez faut fait faire
""".split())


def lire_env() -> dict[str, str]:
    """Lit le .env à la racine du dépôt (format CLE=valeur)."""
    valeurs: dict[str, str] = {}
    fichier = RACINE / ".env"
    if fichier.exists():
        for ligne in fichier.read_text(encoding="utf-8-sig").splitlines():
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
    """Minuscules, accents retirés, mots et nombres, sans mots vides.
    Le retrait des accents aligne les questions en style SMS sur le
    corpus accentué ; le retrait des mots vides évite que BM25 classe
    les 94 chunks sur "je" et "la"."""
    texte = unicodedata.normalize("NFKD", texte.lower())
    texte = "".join(c for c in texte if not unicodedata.combining(c))
    return [m for m in re.findall(r"[a-z0-9]+", texte)
            if m not in MOTS_VIDES]


class Recherche:
    """Charge une variante de chunks et sert les trois modes de recherche.

    Tout est chargé une fois à la construction : les chunks, l'index
    BM25 et le modèle d'embeddings. Le service FastAPI construit un seul
    objet Recherche au démarrage.
    """

    def __init__(self, decoupage: str = "titres", avec_fil: bool = True):
        self.decoupage, self.avec_fil = decoupage, avec_fil
        self.modele = SentenceTransformer("intfloat/multilingual-e5-base")
        with psycopg.connect(DSN) as conn:
            lignes = conn.execute(
                """SELECT id, doc_id, section, formule, texte FROM chunks
                   WHERE decoupage = %s AND avec_fil = %s
                   ORDER BY id""", (decoupage, avec_fil)).fetchall()
        if not lignes:
            raise SystemExit(f"ERREUR : aucun chunk pour decoupage={decoupage}, "
                             f"avec_fil={avec_fil} ; lancer ml/index_corpus.py")
        self.chunks = [{"id": i, "doc_id": d, "section": s, "formule": f,
                        "texte": t} for i, d, s, f, t in lignes]
        self.par_id = {c["id"]: c for c in self.chunks}
        self.bm25 = BM25Okapi([tokeniser(c["texte"]) for c in self.chunks])

        # Formules réellement présentes dans l'index, plutôt qu'une liste
        # écrite à la main qui divergerait du corpus. Le marqueur
        # universel doit s'y trouver : sans lui, un filtrage écarterait
        # tous les documents de procédure sans que rien ne le signale.
        presentes = sorted({c["formule"] for c in self.chunks})
        if FORMULE_UNIVERSELLE not in presentes:
            raise SystemExit(
                f"ERREUR : marqueur « {FORMULE_UNIVERSELLE} » absent de la "
                f"colonne formule ; valeurs trouvées : {presentes}")
        self.formules = [f for f in presentes if f != FORMULE_UNIVERSELLE]

    def valider_formule(self, formule: str | None) -> str | None:
        """Normalise une formule demandée, ou lève ValueError.

        Une formule inconnue doit être une erreur bruyante : acceptée en
        silence, elle produirait une recherche vide, donc une réponse
        « je ne dispose pas de cette information » parfaitement trompeuse.
        """
        if not formule:
            return None
        normalisee = formule.strip().lower()
        if normalisee not in self.formules:
            raise ValueError(f"formule inconnue : {formule!r} ; "
                             f"attendu l'une de {self.formules}")
        return normalisee

    def vectoriel(self, question: str, k: int = CANDIDATS,
                  formule: str | None = None) -> list[tuple[int, float]]:
        """Liste (id, score cosinus) triée, via l'index HNSW de Postgres.

        Avec `formule`, la recherche est restreinte aux chunks de cette
        formule et à ceux valables pour toutes. Sans elle, aucun filtre.
        """
        formule = self.valider_formule(formule)
        vecteur = self.modele.encode(f"query: {question}",
                                     normalize_embeddings=True)
        # Le fragment interpolé ne contient aucune donnée utilisateur,
        # seulement une condition fixe ; les valeurs restent paramétrées.
        conditions = "decoupage = %s AND avec_fil = %s"
        parametres: list = [vecteur, self.decoupage, self.avec_fil]
        if formule:
            conditions += " AND formule IN (%s, %s)"
            parametres += [FORMULE_UNIVERSELLE, formule]
        parametres += [vecteur, k]
        with psycopg.connect(DSN) as conn:
            register_vector(conn)
            lignes = conn.execute(
                f"""SELECT id, 1 - (embedding <=> %s) AS score FROM chunks
                    WHERE {conditions}
                    ORDER BY embedding <=> %s LIMIT %s""",
                parametres).fetchall()
        return [(id_, float(score)) for id_, score in lignes]

    def vectoriel_chunks(self, question: str, k: int = 5,
                         formule: str | None = None) -> list[dict]:
        """Les k meilleurs chunks en vectoriel pur, dictionnaires complets.

        C'est la configuration de service retenue à l'étape 8. Même forme
        de sortie que hybride(), moins les champs de fusion, pour que le
        routeur n'ait pas à recomposer avec par_id.
        """
        return [{**self.par_id[id_], "score_cosinus": score}
                for id_, score in self.vectoriel(question, k=k, formule=formule)]

    def lexical(self, question: str, k: int = CANDIDATS) -> list[tuple[int, float]]:
        """Liste (id, score BM25) triée, calculée en mémoire. Les chunks
        à score nul sont exclus : sans recouvrement lexical, un rang
        BM25 n'a pas de sens et polluerait la fusion RRF."""
        scores = self.bm25.get_scores(tokeniser(question))
        ordre = [i for i in np.argsort(scores)[::-1] if scores[i] > 0][:k]
        return [(self.chunks[i]["id"], float(scores[i])) for i in ordre]

    def hybride(self, question: str, k: int = 5) -> list[dict]:
        """Fusion RRF des deux listes. Retourne les k meilleurs chunks,
        chacun avec son score RRF, son score cosinus (si présent dans la
        liste vectorielle) et son rang dans chaque liste.

        Pas de filtrage par formule ici : l'hybride n'est pas la
        configuration de service, il ne sert qu'au diagnostic et à la
        comparaison de l'étape 8.
        """
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
    parseur.add_argument("--service", action="store_true",
                         help="vectoriel pur, la configuration servie")
    parseur.add_argument("--formule", default=None,
                         help="contexte client, implique --service")
    args = parseur.parse_args()

    recherche = Recherche(args.decoupage, avec_fil=not args.sans_fil)
    print(f"\n{len(recherche.chunks)} chunks chargés, "
          f"formules disponibles : {recherche.formules}")
    print(f"requête : {args.requete}")

    if args.service or args.formule:
        contexte = args.formule or "aucun filtre"
        print(f"mode service, vectoriel pur, contexte : {contexte}\n")
        print(f"{'cos':>6}  {'formule':<10} document / section")
        for c in recherche.vectoriel_chunks(args.requete, formule=args.formule):
            print(f"{c['score_cosinus']:.3f}  {c['formule']:<10} "
                  f"{c['doc_id']} > {c['section']}")
        return

    print("mode diagnostic, fusion hybride\n")
    print(f"{'rrf':>7} {'cos':>6} {'vec':>4} {'lex':>4}  document / section")
    for c in recherche.hybride(args.requete):
        cos = f"{c['score_cosinus']:.3f}" if c["score_cosinus"] is not None else "  -  "
        rv = "-" if c["rang_vectoriel"] is None else c["rang_vectoriel"] + 1
        rl = "-" if c["rang_lexical"] is None else c["rang_lexical"] + 1
        print(f"{c['score_rrf']:.4f} {cos:>6} {rv:>4} {rl:>4}  "
              f"{c['doc_id']} > {c['section']}")


if __name__ == "__main__":
    main()