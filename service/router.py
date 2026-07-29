"""Étape 9 : routeur de l'assistant Mutuelle Solstice.

Assemble les briques mesurées aux étapes 5 à 8 :
- classification CamemBERT et seuil de rejet 0,80 (étape 5c)
- flux métier guidés, dialogue à état piloté par flux_metier.yaml
- RAG en configuration de service, titres + fil en vectoriel pur (étape 8)

DEUX PRINCIPES À NE PAS PERDRE.

1. Le service est SANS ÉTAT. L'état de conversation entre par le
   paramètre `etat` et ressort dans la réponse ; il ne vit jamais dans
   cet objet. La passerelle Spring le persistera (étape 11), le front le
   gardera en mémoire en mode dégradé. Le service reste donc réplicable
   et redémarrable sans perdre une conversation en cours.

2. Le classifieur ne tourne PAS pendant un flux. Si un dialogue est en
   cours, "Dupont" ou "le 12 septembre" sont des réponses attendues, pas
   des messages à classer : les classer donnerait une intention au hasard
   à faible confiance. La sortie de secours est donc lexicale et
   déterministe, pas probabiliste.

Contrôle : python -m service.router --scenario
           python -m service.router --interactif
           python -m service.router --message "..."
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import time
import unicodedata
import logging
from datetime import date, datetime
from pathlib import Path

import yaml

from service.classification import Classifieur
from service.generation import generer
from service.retrieval import Recherche

RACINE = Path(__file__).resolve().parents[1]
CHEMIN_FLUX = RACINE / "data" / "reference" / "flux_metier.yaml"
CHEMIN_OFFRE = RACINE / "data" / "reference" / "offre.yaml"
CHEMIN_TAXONOMIE = RACINE / "data" / "generator" / "intents.py"

# Étape 5c : couverture 82,7%, exactitude sur acceptés 97,5%, flux métier
# déclenchés à tort ramenés de 31 à 4. Ne pas modifier sans regénérer
# docs/seuil_rejet.json.
SEUIL_CONFIANCE = 0.80

# Étape 8 : il n'existe PAS de seuil de pertinence séparateur (minimum
# des questions faciles 0,814, maximum des questions sans réponse 0,872).
# Le refus des questions hors documentation repose sur les consignes de
# génération, mesurées à 16/16. Ce filet grossier reste désactivé ; s'il
# était activé un jour, il ne devrait jamais être présenté comme la
# protection principale.
COSINUS_PLANCHER = None

RELANCES_MAX = 2
CHUNKS_SERVIS = 5

MOTS_ABANDON = frozenset({"annuler", "annule", "annulez", "stop", "abandonner",
                          "abandon", "quitter", "sortir"})
PHRASES_ABANDON = ("laisse tomber", "laissez tomber", "tant pis",
                   "je ne veux plus", "finalement non")

MOTS_OUI = frozenset({"oui", "o", "ok", "okay", "yes", "exact", "exactement",
                      "confirme", "affirmatif", "carrement", "volontiers"})
MOTS_NON = frozenset({"non", "nan", "negatif", "jamais"})
PHRASES_OUI = ("d accord", "tout a fait", "bien sur", "je confirme", "c est ca")
PHRASES_NON = ("pas du tout", "non merci", "surtout pas")

MOIS = {"janvier": 1, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5,
        "juin": 6, "juillet": 7, "aout": 8, "septembre": 9, "octobre": 10,
        "novembre": 11, "decembre": 12}

MOTIF_OFFRE = re.compile(r"\{offre:([a-z0-9_.]+)\}")
MOTIF_CHAMP = re.compile(r"\{([a-z0-9_]+)\}")
MOTIF_DATE_NUM = re.compile(r"(\d{1,2})\s*[/\-.\s]\s*(\d{1,2})"
                            r"(?:\s*[/\-.\s]\s*(\d{2,4}))?")
MOTIF_DATE_MOT = re.compile(r"(\d{1,2})\s+([a-z]+)(?:\s+(\d{4}))?")

# Aucune des 12 intentions ne couvre la politesse : un message purement
# civil est hors taxonomie par construction, et le modèle le range alors
# dans la classe la plus proche avec une confiance élevée ("c bon merci"
# sort à 0,876 sur suivre_remboursement). Pré-filtre déterministe, liste
# FERMÉE : l'allonger au fil des essais reviendrait à calibrer des règles
# ad hoc sur ses propres observations. Sans effet sur les métriques, le
# jeu de test ne contenant aucune formule de politesse.
COURTOISIE = frozenset({"merci", "merci beaucoup", "c bon merci", "cest bon merci",
                        "ok merci", "d accord merci", "parfait merci", "super merci",
                        "au revoir", "bonne journee", "a bientot", "c est bon merci",
                        "bonjour", "bonsoir", "salut", "coucou", "hello"})

journal = logging.getLogger("assistant")


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def normaliser(texte: str) -> str:
    """Minuscules, accents retirés, apostrophes ramenées à un espace."""
    texte = texte.replace("'", " ").replace("\u2019", " ")
    decompose = unicodedata.normalize("NFKD", texte.lower())
    return "".join(c for c in decompose if not unicodedata.combining(c))


def mots_de(texte: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normaliser(texte))


def charger_taxonomie() -> dict[str, str]:
    """Type de routage de chaque intention, lu depuis le module qui a
    servi à générer les données. Source unique : si une intention change
    de type, elle change ici et le routeur suit sans modification."""
    spec = importlib.util.spec_from_file_location("intents_taxonomie",
                                                  CHEMIN_TAXONOMIE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {nom: contenu["type"] for nom, contenu in module.INTENTIONS.items()}


def resoudre_offre(valeur, offre: dict):
    """Remplace récursivement les emplacements {offre:chemin.vers.cle}.

    Interrompt le démarrage si une clé est absente : mieux vaut un
    service qui refuse de démarrer qu'un service qui répond à un assuré
    avec un emplacement non résolu dans le texte.
    """
    if isinstance(valeur, str):
        def remplacer(motif: re.Match) -> str:
            courant = offre
            for cle in motif.group(1).split("."):
                if not isinstance(courant, dict) or cle not in courant:
                    raise SystemExit(
                        f"ERREUR : la clé « {motif.group(1)} » est absente de "
                        f"{CHEMIN_OFFRE.name} ; corriger flux_metier.yaml")
                courant = courant[cle]
            return str(courant)
        return MOTIF_OFFRE.sub(remplacer, valeur)
    if isinstance(valeur, dict):
        return {c: resoudre_offre(v, offre) for c, v in valeur.items()}
    if isinstance(valeur, list):
        return [resoudre_offre(v, offre) for v in valeur]
    return valeur


# ---------------------------------------------------------------------------
# Lecture des réponses de l'utilisateur, déterministe
# ---------------------------------------------------------------------------

def veut_abandonner(message: str) -> bool:
    norm = normaliser(message).strip(" .!?,;")
    return norm in MOTS_ABANDON or any(p in norm for p in PHRASES_ABANDON)


def lire_oui_non(message: str) -> str | None:
    norm = normaliser(message)
    mots = mots_de(message)
    if mots and mots[0] in MOTS_OUI:
        return "oui"
    if mots and mots[0] in MOTS_NON:
        return "non"
    if any(p in norm for p in PHRASES_OUI):
        return "oui"
    if any(p in norm for p in PHRASES_NON):
        return "non"
    return None


def lire_choix(champ: dict, message: str) -> str | None:
    """Numéro de l'option, nom de la clé, ou l'un des mots déclarés.

    Retourne None si aucune option ne correspond OU si plusieurs
    correspondent : une réponse ambiguë vaut mieux relancée que devinée.
    """
    mots = set(mots_de(message))
    retenus = set()
    for rang, valeur in enumerate(champ["valeurs"], start=1):
        if str(rang) in mots:
            retenus.add(valeur["cle"])
        if valeur["cle"] in mots or set(valeur["cle"].split("_")) <= mots:
            retenus.add(valeur["cle"])
        if mots & set(valeur.get("mots", [])):
            retenus.add(valeur["cle"])
    return retenus.pop() if len(retenus) == 1 else None


def lire_date(message: str) -> str | None:
    """jj/mm/aaaa, jj/mm, ou « 12 septembre 2026 ». Année omise : l'année
    en cours. Retourne une date normalisée jj/mm/aaaa, ou None."""
    norm = normaliser(message)
    trouve = MOTIF_DATE_NUM.search(norm)
    if trouve:
        jour, mois, annee = trouve.groups()
    else:
        trouve = MOTIF_DATE_MOT.search(norm)
        if not trouve or trouve.group(2) not in MOIS:
            return None
        jour, mois, annee = trouve.group(1), MOIS[trouve.group(2)], trouve.group(3)
    annee = int(annee) if annee else date.today().year
    if annee < 100:
        annee += 2000
    try:
        return datetime(annee, int(mois), int(jour)).strftime("%d/%m/%Y")
    except ValueError:
        return None


def lire_texte(message: str) -> str | None:
    propre = " ".join(message.split())
    return propre[:120] if len(propre) >= 2 else None


# ---------------------------------------------------------------------------

class Routeur:
    """Un seul objet, construit au démarrage de FastAPI (étape 10).

    Le chargement coûte quelques secondes, la réponse quelques dizaines
    de millisecondes hors appel au modèle de langage.
    """

    def __init__(self):
        offre = yaml.safe_load(CHEMIN_OFFRE.read_text(encoding="utf-8"))
        brut = yaml.safe_load(CHEMIN_FLUX.read_text(encoding="utf-8"))
        self.messages = brut.pop("messages")
        self.flux = resoudre_offre(brut, offre)
        self.messages = resoudre_offre(self.messages, offre)

        self.taxonomie = charger_taxonomie()
        self.classifieur = Classifieur()
        self.recherche = Recherche(decoupage="titres", avec_fil=True)
        self._verifier()

    def _verifier(self) -> None:
        """Trois garde-fous au démarrage. Une incohérence entre le modèle,
        la taxonomie et les flux doit arrêter le service, pas produire un
        routage silencieusement faux."""
        connues = set(self.classifieur.etiquettes)
        declarees = set(self.taxonomie)
        if connues != declarees:
            raise SystemExit(
                f"ERREUR : le modèle et la taxonomie divergent. "
                f"Modèle seul : {sorted(connues - declarees)}. "
                f"Taxonomie seule : {sorted(declarees - connues)}.")

        transactionnelles = {n for n, t in self.taxonomie.items()
                             if t == "transactionnelle"}
        if manquants := sorted(transactionnelles - set(self.flux)):
            raise SystemExit(f"ERREUR : intentions transactionnelles sans flux "
                             f"dans {CHEMIN_FLUX.name} : {manquants}")
        if inconnus := sorted(set(self.flux) - transactionnelles):
            raise SystemExit(f"ERREUR : flux déclarés pour des intentions non "
                             f"transactionnelles : {inconnus}")

    # -- contrat de sortie -------------------------------------------------

    @staticmethod
    def _nettoyer(texte: str) -> str:
        """Replie les retours à la ligne introduits par les blocs > du
        YAML, mais CONSERVE les sauts volontaires : ce sont eux qui
        séparent les options numérotées, que le front rendra en boutons.
        Les lignes vides consécutives sont ramenées à une seule."""
        propres: list[str] = []
        for ligne in texte.split("\n"):
            ligne = " ".join(ligne.split())
            if ligne or (propres and propres[-1]):
                propres.append(ligne)
        return "\n".join(propres).strip()
    
    @staticmethod
    def _sortie(reponse: str, chemin: str, intention: str | None = None,
                confiance: float | None = None, sources: list | None = None,
                etat: dict | None = None, latences: dict | None = None) -> dict:
        base = {"classification": 0, "recherche": 0, "generation": 0}
        return {"reponse": Routeur._nettoyer(reponse),
                "intention": intention,
                "confiance": None if confiance is None else round(confiance, 4),
                "chemin": chemin,
                "sources": sources or [],
                "etat": etat,
                "latence_ms": base | (latences or {})}

    # -- point d'entrée ----------------------------------------------------

    def repondre(self, message: str, etat: dict | None = None,
                 formule: str | None = None) -> dict:
        """Point d'entrée. `formule` est le contexte client, injecté par
        la passerelle Spring à l'étape 11 : il restreint la recherche
        documentaire aux garanties de cette formule et aux documents de
        procédure. Absent, le comportement est celui des étapes 9 et 10.

        Validation en un seul endroit, à l'entrée : une formule inconnue
        lève ValueError ici, que l'API traduira en 400.
        """
        formule = self.recherche.valider_formule(formule)
        sortie = self._router(message, etat, formule)
        sortie["formule"] = formule
        return sortie

    def _router(self, message: str, etat: dict | None,
                formule: str | None) -> dict:
        """Chemins possibles : flux_metier, rag, reformulation, recadrage."""
        if normaliser(message).strip(" .!?,;") in COURTOISIE:
            return self._sortie(self.messages["reformulation"], "reformulation")

        if etat and etat.get("flux"):
            return self._poursuivre(message, etat)

        prediction = self.classifieur.predire(message)
        intention = prediction["intention"]
        confiance = prediction["confiance"]
        latences = {"classification": prediction["latence_ms"]}

        if confiance < SEUIL_CONFIANCE:
            return self._sortie(self.messages["reformulation"], "reformulation",
                                intention=intention, confiance=confiance,
                                latences=latences)

        type_routage = self.taxonomie[intention]
        if type_routage == "rejet":
            return self._sortie(self.messages["recadrage"], "recadrage",
                                intention=intention, confiance=confiance,
                                latences=latences)
        if type_routage == "transactionnelle":
            return self._ouvrir(intention, confiance, latences)
        return self._rag(message, intention, confiance, latences, formule)

    # -- chemin documentaire ----------------------------------------------

    def _rag(self, message: str, intention: str, confiance: float,
             latences: dict, formule: str | None = None) -> dict:
        debut = time.perf_counter()
        chunks = self.recherche.vectoriel_chunks(message, k=CHUNKS_SERVIS,
                                                 formule=formule)
        latences["recherche"] = round((time.perf_counter() - debut) * 1000)

        sources = [{"document": c["doc_id"],
                    "section": c["section"],
                    "score": round(c["score_cosinus"], 3),
                    "extrait": " ".join(c["texte"].split())[:220]}
                   for c in chunks]

        if COSINUS_PLANCHER is not None and (
                not chunks or chunks[0]["score_cosinus"] < COSINUS_PLANCHER):
            return self._sortie(self.messages["recadrage"], "recadrage",
                                intention=intention, confiance=confiance,
                                sources=sources, latences=latences)

        try:
            resultat = generer(message, chunks)
        except Exception as erreur:
            # Dégradation propre : la passerelle Spring traduira en 503
            # (étape 11). L'erreur est journalisée, jamais avalée : un 429
            # du palier gratuit et une clé absente ne se soignent pas
            # pareil. Les sources sont tout de même remontées, le panneau
            # d'inspection reste utile.
            journal.warning("generation en echec : %s", erreur)
            return self._sortie(self.messages["erreur_generation"], "rag",
                                intention=intention, confiance=confiance,
                                sources=sources, latences=latences)

        latences["generation"] = resultat["latence_ms"]
        return self._sortie(resultat["reponse"], "rag", intention=intention,
                            confiance=confiance, sources=sources,
                            latences=latences)

    # -- moteur de flux ----------------------------------------------------

    @staticmethod
    def _applicable(champ: dict, donnees: dict) -> bool:
        condition = champ.get("si")
        return not condition or donnees.get(condition["champ"]) == condition["vaut"]

    def _prochain(self, flux: dict, donnees: dict) -> dict | None:
        for champ in flux.get("champs", []):
            if champ["nom"] not in donnees and self._applicable(champ, donnees):
                return champ
        return None

    @staticmethod
    def _question(champ: dict) -> str:
        """La question, suivie des options numérotées pour un champ de
        type choix. Le front pourra les rendre en boutons (étape 12)."""
        texte = " ".join(champ["question"].split())
        if champ["type"] == "choix":
            options = "\n".join(f"{rang}. {' '.join(v['libelle'].split())}"
                                for rang, v in enumerate(champ["valeurs"], 1))
            texte = f"{texte}\n{options}"
        return texte

    def _ouvrir(self, intention: str, confiance: float, latences: dict) -> dict:
        flux = self.flux[intention]
        champ = self._prochain(flux, {})
        if champ is None:                      # flux sans champ, un seul tour
            return self._sortie(flux["reponse"], "flux_metier",
                                intention=intention, confiance=confiance,
                                latences=latences)
        etat = {"flux": intention, "attente": champ["nom"], "donnees": {},
                "relances": 0, "confiance_ouverture": confiance}
        entete = " ".join(flux.get("ouverture", "").split())
        return self._sortie(f"{entete}\n\n{self._question(champ)}",
                            "flux_metier", intention=intention,
                            confiance=confiance, etat=etat, latences=latences)

    def _poursuivre(self, message: str, etat: dict) -> dict:
        intention = etat["flux"]
        flux = self.flux[intention]
        donnees = dict(etat.get("donnees", {}))
        confiance = etat.get("confiance_ouverture")

        if veut_abandonner(message):
            return self._sortie(self.messages["abandon"], "flux_metier",
                                intention=intention, confiance=confiance)

        champ = next(c for c in flux["champs"] if c["nom"] == etat["attente"])
        valeur = {"oui_non": lambda: lire_oui_non(message),
                  "choix": lambda: lire_choix(champ, message),
                  "date": lambda: lire_date(message),
                  "texte": lambda: lire_texte(message)}[champ["type"]]()

        if valeur is None:
            relances = etat.get("relances", 0) + 1
            if relances > RELANCES_MAX:
                return self._sortie(self.messages["echec_champ"], "flux_metier",
                                    intention=intention, confiance=confiance)
            suite = dict(etat, relances=relances)
            return self._sortie(f"{champ['relance']}\n\n{self._question(champ)}",
                                "flux_metier", intention=intention,
                                confiance=confiance, etat=suite)

        if champ["type"] == "oui_non" and valeur == "non" and "si_non" in champ:
            return self._sortie(champ["si_non"], "flux_metier",
                                intention=intention, confiance=confiance)

        donnees[champ["nom"]] = valeur
        suivant = self._prochain(flux, donnees)
        if suivant is not None:
            suite = {"flux": intention, "attente": suivant["nom"],
                     "donnees": donnees, "relances": 0,
                     "confiance_ouverture": confiance}
            return self._sortie(self._question(suivant), "flux_metier",
                                intention=intention, confiance=confiance,
                                etat=suite)

        return self._sortie(self._clore(flux, donnees), "flux_metier",
                            intention=intention, confiance=confiance)

    @staticmethod
    def _clore(flux: dict, donnees: dict) -> str:
        if "reponse_par" in flux:
            texte = flux["reponses"][donnees[flux["reponse_par"]]]
        else:
            texte = flux["reponse"]
        if flux.get("cloture"):
            texte = f"{texte.rstrip()} {flux['cloture']}"
        return MOTIF_CHAMP.sub(
            lambda m: str(donnees.get(m.group(1), m.group(0))), texte)


# ---------------------------------------------------------------------------
# Contrôles
# ---------------------------------------------------------------------------

SCENARIO = [
    "je souhaite mettre un terme a mon contrat des le mois prochain",
    "oui",
    "2",
    "bonjour",
    "je voudrais assurer ma voiture",
    "il me faudrait un papier pour prouver que je suis affilie",
    "1",
    "je dois etre hospitalise la semaine prochaine",
    "n importe quoi",
    "annuler",
]


def afficher(message: str, sortie: dict) -> None:
    confiance = ("-" if sortie["confiance"] is None
                 else f"{sortie['confiance']:.3f}")
    print(f"\n> {message}")
    print(f"  [{sortie['chemin']:<13} {str(sortie['intention']):<24} "
          f"conf {confiance}  etat "
          f"{sortie['etat']['attente'] if sortie['etat'] else '-'}]")
    for ligne in sortie["reponse"].split("\n"):
        print(f"  {ligne}")
    for source in sortie["sources"]:
        print(f"    source {source['score']:.3f}  "
              f"{source['document']} > {source['section']}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("--message")
    parseur.add_argument("--formule", default=None,
                         help="contexte client, filtre la recherche RAG")
    parseur.add_argument("--scenario", action="store_true")
    parseur.add_argument("--interactif", action="store_true")
    parseur.add_argument("--json", action="store_true",
                         help="sortie brute, pour vérifier le contrat")
    args = parseur.parse_args()

    debut = time.perf_counter()
    routeur = Routeur()
    print(f"[ok] routeur prêt en {round(time.perf_counter() - debut, 1)} s, "
          f"{len(routeur.flux)} flux, {len(routeur.taxonomie)} intentions, "
          f"formules {routeur.recherche.formules}")

    if args.scenario:
        etat = None
        for message in SCENARIO:
            sortie = routeur.repondre(message, etat)
            afficher(message, sortie)
            etat = sortie["etat"]
        return

    if args.interactif:
        etat = None
        while True:
            try:
                message = input("\nvous > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not message:
                break
            sortie = routeur.repondre(message, etat, args.formule)
            afficher(message, sortie)
            etat = sortie["etat"]
        return

    if not args.message:
        raise SystemExit("Indiquer --message, --scenario ou --interactif")
    sortie = routeur.repondre(args.message, None, args.formule)
    if args.json:
        print(json.dumps(sortie, ensure_ascii=False, indent=2))
    else:
        afficher(args.message, sortie)


if __name__ == "__main__":
    main()