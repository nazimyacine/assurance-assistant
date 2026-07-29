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


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("--dry-run", action="store_true")
    args = parseur.parse_args()

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
    raise SystemExit("Partie embeddings et Postgres : commit suivant.")


if __name__ == "__main__":
    main()