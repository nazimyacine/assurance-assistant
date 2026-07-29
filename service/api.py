"""Étape 10 : service HTTP FastAPI.

Deux routes, POST /chat et GET /health.

Le routeur, donc CamemBERT, le modèle e5 et l'index BM25, est construit
UNE FOIS au démarrage, environ 4 secondes, et vit pour toute la durée du
processus. Un message coûte ensuite 25 ms de classification et 100 ms de
recherche, contre plusieurs secondes s'il fallait recharger à chaque
appel.

Le service reste SANS ÉTAT : l'état de conversation entre par le corps de
la requête et ressort dans la réponse. Deux instances derrière un
répartiteur se comportent identiquement, et un redémarrage ne perd aucune
conversation en cours.

Étape 11 : la requête accepte une formule optionnelle, le contexte client
injecté par la passerelle Spring. Elle restreint la recherche documentaire
aux garanties de cette formule et aux documents de procédure.

Lancement :
    $env:HF_HUB_OFFLINE = "1"
    python -m uvicorn service.api:app --port 8000

Documentation interactive et contrat de sortie : http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager

import psycopg
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from service.retrieval import DSN
from service.router import (CHUNKS_SERVIS, COSINUS_PLANCHER, SEUIL_CONFIANCE,
                            Routeur)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
journal = logging.getLogger("assistant")

# Le front Angular (4200) appelle ce service directement en mode dégradé,
# c'est-à-dire si la passerelle Spring (8080) est coupée. Sans ces
# origines déclarées, le navigateur bloque la requête et l'erreur est
# peu lisible côté front.
ORIGINES = ["http://localhost:4200", "http://localhost:8080"]

# Une seule carte graphique, un seul modèle en mémoire : les messages
# sont traités un par un. FastAPI exécutant les routes synchrones dans un
# fil d'exécution séparé, /health reste joignable pendant qu'un message
# occupe le verrou. Un service de production placerait le modèle derrière
# une file avec regroupement des requêtes ; à l'échelle d'une
# démonstration, ce verrou est le bon compromis, et ce sont les délais
# d'attente explicites de la passerelle Spring qui protégeront le front.
VERROU = threading.Lock()

SERVICE: dict = {"routeur": None}


@asynccontextmanager
async def cycle_de_vie(app: FastAPI):
    journal.info("chargement du routeur")
    SERVICE["routeur"] = Routeur()
    journal.info("routeur prêt : %d flux, %d intentions, formules %s",
                 len(SERVICE["routeur"].flux),
                 len(SERVICE["routeur"].taxonomie),
                 SERVICE["routeur"].recherche.formules)
    yield
    SERVICE["routeur"] = None


app = FastAPI(
    title="Assistant Mutuelle Solstice",
    description="Classification d'intentions CamemBERT, flux métier guidés "
                "et recherche documentaire RAG avec contexte client.",
    version="0.11.0",
    lifespan=cycle_de_vie,
)
app.add_middleware(CORSMiddleware, allow_origins=ORIGINES,
                   allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# Contrat, typé pour être documenté automatiquement sur /docs
# ---------------------------------------------------------------------------

class Etat(BaseModel):
    """État du dialogue en cours. Le service ne le conserve pas : il le
    rend au client, qui le renvoie au tour suivant.

    ATTENTION : tout champ ajouté à l'état dans router.py doit être
    ajouté ici. FastAPI filtre silencieusement les champs inconnus d'un
    modèle de réponse, et un dialogue perdrait sa mémoire sans qu'aucune
    erreur ne soit levée.
    """
    flux: str
    attente: str
    donnees: dict[str, str] = {}
    relances: int = 0
    confiance_ouverture: float | None = None


class Source(BaseModel):
    document: str
    section: str
    score: float
    extrait: str


class Latences(BaseModel):
    classification: int = 0
    recherche: int = 0
    generation: int = 0


class Requete(BaseModel):
    # La borne haute n'est pas décorative : elle empêche qu'un corps de
    # requête démesuré parte vers le modèle de langage.
    message: str = Field(min_length=1, max_length=1000)
    etat: Etat | None = None
    # Contexte client injecté par la passerelle Spring. Absent en appel
    # direct, la recherche porte alors sur tout le corpus.
    formule: str | None = None


class Reponse(BaseModel):
    reponse: str
    intention: str | None = None
    confiance: float | None = None
    chemin: str
    formule: str | None = None
    sources: list[Source] = []
    etat: Etat | None = None
    latence_ms: Latences


# ---------------------------------------------------------------------------

@app.post("/chat", response_model=Reponse)
def chat(requete: Requete) -> dict:
    routeur = SERVICE["routeur"]
    if routeur is None:
        raise HTTPException(status_code=503, detail="service en cours de démarrage")

    etat = requete.etat.model_dump() if requete.etat else None
    try:
        with VERROU:
            sortie = routeur.repondre(requete.message, etat, requete.formule)
    except ValueError as erreur:
        # Formule inconnue : faute du client, pas du service.
        raise HTTPException(status_code=400, detail=str(erreur)) from erreur

    # Journalisation d'observation seulement. La journalisation
    # persistante, avec identifiant de session, appartient à la
    # passerelle Spring (étape 11).
    journal.info("chemin=%s intention=%s confiance=%s formule=%s latences=%s",
                 sortie["chemin"], sortie["intention"], sortie["confiance"],
                 sortie["formule"], sortie["latence_ms"])
    return sortie


@app.get("/health")
def health(reponse: Response) -> dict:
    """Vérifie ce qui peut réellement tomber, et expose la configuration
    servie : celle qui est mesurée dans docs/metrics.md doit être celle
    qui tourne."""
    routeur = SERVICE["routeur"]
    if routeur is None:
        raise HTTPException(status_code=503, detail="routeur non chargé")

    try:
        with psycopg.connect(DSN, connect_timeout=3) as conn:
            (chunks_en_base,) = conn.execute(
                "SELECT count(*) FROM chunks "
                "WHERE decoupage = %s AND avec_fil = %s",
                (routeur.recherche.decoupage,
                 routeur.recherche.avec_fil)).fetchone()
        base = "ok"
    except Exception as erreur:
        journal.warning("base injoignable : %s", erreur)
        chunks_en_base, base = None, "injoignable"

    if base != "ok":
        reponse.status_code = 503

    return {
        "statut": "ok" if base == "ok" else "degrade",
        "base": base,
        "chunks_en_base": chunks_en_base,
        "chunks_en_memoire": len(routeur.recherche.chunks),
        "intentions": len(routeur.classifieur.etiquettes),
        "flux": len(routeur.flux),
        "device": routeur.classifieur.device,
        "configuration": {
            "decoupage": routeur.recherche.decoupage,
            "avec_fil": routeur.recherche.avec_fil,
            "seuil_confiance": SEUIL_CONFIANCE,
            "cosinus_plancher": COSINUS_PLANCHER,
            "chunks_servis": CHUNKS_SERVIS,
            "formules": routeur.recherche.formules,
        },
    }