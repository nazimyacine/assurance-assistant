"""Étape 6 : indexation du corpus. Partie 1, le découpage.

Trois variantes de chunks pour servir les 4 configurations de l'étape 8 :
- decoupage=naif : fenêtres de mots à taille fixe, structure ignorée
- decoupage=titres, avec_fil=False : une section (titre de niveau 2) par chunk
- decoupage=titres, avec_fil=True : idem, fil d'Ariane en tête du texte

Chaque section étant rédigée pour se suffire à elle-même (convention du
corpus), le découpage par titres donne un chunk par section ; MAX_MOTS est
un plafond de refente, pas une cible.

--dry-run : découpe, affiche les statistiques, n'écrit rien.
"""

from __future__ import annotations

import argparse
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import yaml

RACINE = Path(__file__).resolve().parents[1]
CORPUS = RACINE / "data" / "raw" / "corpus"
MAX_MOTS = 500
FENETRE_NAIF = 400
CHEVAUCHEMENT = 50

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


@dataclass
class Chunk:
    doc_id: str
    type: str
    formule: str
    section: str
    decoupage: str          # "naif" ou "titres"
    avec_fil: bool
    texte: str
    n_mots: int = field(init=False)

    def __post_init__(self):
        self.n_mots = len(self.texte.split())


def lire_document(chemin: Path) -> tuple[dict, str]:
    brut = chemin.read_text(encoding="utf-8")
    m = re.match(r"^---\r?\n(.*?)\r?\n---\r?\n(.*)$", brut, re.S)
    if not m:
        raise SystemExit(f"ERREUR : frontmatter absent ou malformé dans {chemin.name}")
    meta = yaml.safe_load(m.group(1))
    for cle in ("doc_id", "titre", "type", "formule"):
        if cle not in meta:
            raise SystemExit(f"ERREUR : clé '{cle}' absente du frontmatter de {chemin.name}")
    return meta, m.group(2)


def fenetres(mots: list[str], taille: int, chevauchement: int) -> list[str]:
    pas = taille - chevauchement
    return [" ".join(mots[i:i + taille])
            for i in range(0, max(len(mots) - chevauchement, 1), pas)]


def decouper_titres(meta: dict, corps: str) -> list[tuple[str, str]]:
    """Retourne des paires (section, texte), le préambule inclus."""
    morceaux = re.split(r"^##\s+", corps, flags=re.M)
    preambule = re.sub(r"^#\s+.*$", "", morceaux[0], flags=re.M).strip()
    paires = []
    if preambule:
        paires.append(("Introduction", preambule))
    for morceau in morceaux[1:]:
        titre_section, _, texte = morceau.partition("\n")
        paires.append((titre_section.strip(), texte.strip()))
    return paires


def chunks_document(meta: dict, corps: str) -> list[Chunk]:
    commun = dict(doc_id=meta["doc_id"], type=meta["type"], formule=str(meta["formule"]))
    resultat: list[Chunk] = []

    # Variante naïve : fenêtres fixes sur le texte débarrassé des titres.
    plein = re.sub(r"^#{1,2}\s+.*$", "", corps, flags=re.M)
    for i, texte in enumerate(fenetres(plein.split(), FENETRE_NAIF, CHEVAUCHEMENT)):
        resultat.append(Chunk(**commun, section=f"fenetre-{i}",
                              decoupage="naif", avec_fil=False, texte=texte))

    # Variantes par titres, sans puis avec fil d'Ariane.
    for section, texte in decouper_titres(meta, corps):
        sous_textes = ([texte] if len(texte.split()) <= MAX_MOTS
                       else fenetres(texte.split(), MAX_MOTS, CHEVAUCHEMENT))
        for sous in sous_textes:
            fil = f"{meta['titre']} > {section}"
            resultat.append(Chunk(**commun, section=section, decoupage="titres",
                                  avec_fil=False, texte=sous))
            resultat.append(Chunk(**commun, section=section, decoupage="titres",
                                  avec_fil=True, texte=f"{fil}\n\n{sous}"))
    return resultat


def statistiques(chunks: list[Chunk]) -> None:
    for variante in [("naif", False), ("titres", False), ("titres", True)]:
        sel = [c.n_mots for c in chunks
               if (c.decoupage, c.avec_fil) == variante]
        nom = f"{variante[0]}{', fil' if variante[1] else ''}"
        print(f"{nom:12s} : {len(sel):3d} chunks, mots min {min(sel)}, "
              f"médiane {statistics.median(sel):.0f}, max {max(sel)}")
    trop_longs = [c for c in chunks if c.n_mots > MAX_MOTS]
    if trop_longs:
        raise SystemExit(f"ERREUR : {len(trop_longs)} chunk(s) au dessus de {MAX_MOTS} mots")
    print("Plafond de mots : OK")

def indexer(chunks: list[Chunk]) -> None:
    import psycopg
    from pgvector.psycopg import register_vector
    from sentence_transformers import SentenceTransformer

    modele = SentenceTransformer("intfloat/multilingual-e5-base")
    # e5 exige le préfixe "passage: " à l'encodage ; le texte stocké
    # en base reste sans préfixe.
    embeddings = modele.encode([f"passage: {c.texte}" for c in chunks],
                               batch_size=64, normalize_embeddings=True,
                               show_progress_bar=True)

    with psycopg.connect(DSN, autocommit=True) as conn:
        register_vector(conn)
        conn.execute("DROP TABLE IF EXISTS chunks")
        conn.execute("""
            CREATE TABLE chunks (
                id        serial PRIMARY KEY,
                doc_id    text    NOT NULL,
                type      text    NOT NULL,
                formule   text    NOT NULL,
                section   text    NOT NULL,
                decoupage text    NOT NULL,
                avec_fil  boolean NOT NULL,
                n_mots    int     NOT NULL,
                texte     text    NOT NULL,
                embedding vector(768) NOT NULL
            )""")
        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO chunks (doc_id, type, formule, section,
                       decoupage, avec_fil, n_mots, texte, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                [(c.doc_id, c.type, c.formule, c.section, c.decoupage,
                  c.avec_fil, c.n_mots, c.texte, e)
                 for c, e in zip(chunks, embeddings)])
        conn.execute("CREATE INDEX ON chunks "
                     "USING hnsw (embedding vector_cosine_ops)")
        n = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        print(f"{n} chunks indexés dans Postgres")


def chercher(requete: str, k: int = 5) -> None:
    """Contrôle de l'étape : les k chunks les plus proches, variante
    titres + fil, celle de la configuration cible."""
    import psycopg
    from pgvector.psycopg import register_vector
    from sentence_transformers import SentenceTransformer

    modele = SentenceTransformer("intfloat/multilingual-e5-base")
    vecteur = modele.encode(f"query: {requete}", normalize_embeddings=True)
    with psycopg.connect(DSN) as conn:
        register_vector(conn)
        resultats = conn.execute(
            """SELECT doc_id, section, 1 - (embedding <=> %s) AS score,
                      left(texte, 90)
               FROM chunks
               WHERE decoupage = 'titres' AND avec_fil
               ORDER BY embedding <=> %s
               LIMIT %s""", (vecteur, vecteur, k)).fetchall()
    print(f"\nrequête : {requete}")
    for doc_id, section, score, extrait in resultats:
        print(f"  {score:.3f}  {doc_id:24s} {section:32s} {extrait}...")


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("--dry-run", action="store_true",
                         help="découpe et affiche les statistiques, n'écrit rien")
    parseur.add_argument("--requete",
                         help="cherche cette question dans l'index au lieu d'indexer")
    args = parseur.parse_args()

    if args.requete:
        chercher(args.requete)
        return

    documents = sorted(CORPUS.glob("*.md"))
    if len(documents) != 11:
        raise SystemExit(f"ERREUR : {len(documents)} documents trouvés, 11 attendus")
    chunks: list[Chunk] = []
    for chemin in documents:
        meta, corps = lire_document(chemin)
        chunks.extend(chunks_document(meta, corps))
    print(f"{len(documents)} documents, {len(chunks)} chunks toutes variantes")
    statistiques(chunks)
    if args.dry_run:
        return
    indexer(chunks)


if __name__ == "__main__":
    main()