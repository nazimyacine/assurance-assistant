"""Génère le jeu de données de classification d'intentions.

Trois sorties :
  data/raw/intents_train.csv        entraînement
  data/raw/intents_val.csv          validation pendant l'entraînement
  data/eval/intents_test_a_relire.csv  test, à relire à la main

Principe central : les gabarits sont répartis entre train et test avant
toute génération. Un gabarit utilisé à l'entraînement ne sert jamais au
test. Sans cette séparation, le modèle est évalué sur des phrases
construites avec les mêmes moules que celles qu'il a apprises, et le score
mesure la mémorisation du moule, pas la compréhension.
"""

import csv
import random
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from intents import INTENTIONS, VALEURS, CAS_AMBIGUS

RACINE = Path(__file__).resolve().parents[2]
SORTIE_RAW = RACINE / "data" / "raw"
SORTIE_EVAL = RACINE / "data" / "eval"

GRAINE = 42
TOTAL_TRAIN = 5000
TOTAL_VAL = 800
TOTAL_TEST = 500
PART_GABARITS_TEST = 0.30

# ---------------------------------------------------------------------------
# Bruit
# ---------------------------------------------------------------------------

ABREVIATIONS = {
    "je": "j", "je voudrais": "jvoudré", "je veux": "jveu",
    "c'est": "c", "s'il vous plaît": "svp", "beaucoup": "bcp",
    "pour": "pr", "pourquoi": "pq", "est ce que": "esk",
    "vous": "vs", "rendez vous": "rdv", "s'il vous plait": "stp",
    "bonjour": "bjr", "d'accord": "dac", "quelque": "qq",
}

POLITESSES_AVANT = [
    "bonjour ", "bonjour, ", "bjr ", "salut ", "hello ",
    "bonjour madame, ", "bonsoir, ", "coucou ",
]

POLITESSES_APRES = [
    " merci", " svp", " s'il vous plaît", " merci d'avance",
    " cordialement", " merci beaucoup", " ?", " ??",
]

BAVARDAGES = [
    "alors voilà, ",
    "je vous explique ma situation, ",
    "je suis adhérent depuis trois ans et ",
    "désolé de vous déranger mais ",
    "j'ai déjà appelé la semaine dernière et personne ne m'a répondu, ",
    "je ne trouve pas l'information sur votre site, ",
    "ma femme m'a dit de vous contacter, ",
]


def enlever_accents(texte: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", texte)
        if unicodedata.category(c) != "Mn"
    )


def faute_de_frappe(texte: str, rng: random.Random) -> str:
    """Applique une faute de frappe plausible sur un mot d'au moins 4 lettres."""
    mots = texte.split()
    candidats = [i for i, m in enumerate(mots) if len(m) >= 4]
    if not candidats:
        return texte
    i = rng.choice(candidats)
    mot = list(mots[i])
    j = rng.randrange(len(mot) - 1)
    action = rng.choice(["inversion", "suppression", "doublon"])
    if action == "inversion":
        mot[j], mot[j + 1] = mot[j + 1], mot[j]
    elif action == "suppression":
        del mot[j]
    else:
        mot.insert(j, mot[j])
    mots[i] = "".join(mot)
    return " ".join(mots)


def abreger(texte: str, rng: random.Random) -> str:
    for complet, court in ABREVIATIONS.items():
        if complet in texte and rng.random() < 0.5:
            texte = texte.replace(complet, court, 1)
    return texte


def bruiter(texte: str, rng: random.Random) -> tuple[str, str]:
    """Renvoie (texte bruité, niveau de difficulté)."""
    niveau = "facile"

    if rng.random() < 0.22:
        texte = rng.choice(BAVARDAGES) + texte
        niveau = "bruite"

    if rng.random() < 0.28:
        texte = rng.choice(POLITESSES_AVANT) + texte
        niveau = "bruite"

    if rng.random() < 0.25:
        texte = texte + rng.choice(POLITESSES_APRES)

    if rng.random() < 0.20:
        avant = texte
        texte = abreger(texte, rng)
        if texte != avant:
            niveau = "bruite"

    if rng.random() < 0.18:
        avant = texte
        texte = faute_de_frappe(texte, rng)
        if texte != avant:
            niveau = "bruite"

    if rng.random() < 0.12:
        avant = texte
        texte = enlever_accents(texte)
        if texte != avant:
            niveau = "bruite"

    if rng.random() < 0.08:
        avant = texte
        texte = texte.upper()
        if texte != avant:
            niveau = "bruite"

    return texte.strip(), niveau


# ---------------------------------------------------------------------------
# Remplissage des gabarits
# ---------------------------------------------------------------------------

ELISIONS = [
    ("de le ", "du "),
    ("de les ", "des "),
    ("de l'", "de l'"),
    ("à le ", "au "),
    ("à les ", "aux "),
]


def corriger_elisions(texte: str) -> str:
    """Répare les contractions cassées par le remplissage des gabarits.

    « le remboursement de {acte_court} » avec « les lunettes » produit
    « de les lunettes ». Aucun francophone n'écrit cela : c'est un défaut de
    génération, pas du bruit réaliste.
    """
    for faux, correct in ELISIONS:
        texte = texte.replace(faux, correct)
    return texte


def remplir(gabarit: str, rng: random.Random) -> str:
    texte = gabarit
    for cle, valeurs in VALEURS.items():
        marqueur = "{" + cle + "}"
        while marqueur in texte:
            texte = texte.replace(marqueur, rng.choice(valeurs), 1)
    return corriger_elisions(texte)


def repartir_gabarits(rng: random.Random) -> tuple[dict, dict]:
    """Sépare les gabarits de chaque intention en deux ensembles disjoints."""
    train, test = {}, {}
    for nom, config in INTENTIONS.items():
        gabarits = list(config["gabarits"])
        rng.shuffle(gabarits)
        n_test = max(2, round(len(gabarits) * PART_GABARITS_TEST))
        test[nom] = gabarits[:n_test]
        train[nom] = gabarits[n_test:]
    return train, test


def generer(pool: dict, total: int, rng: random.Random,
            deja_vus: set[str] | None = None) -> list[dict]:
    """Génère `total` phrases uniques, réparties selon les poids.

    On tire jusqu'à atteindre le quota de chaque intention plutôt qu'un
    nombre fixe de tirages : sans cela, la déduplication vide les petites
    classes et casse le déséquilibre voulu.
    """
    vus = set() if deja_vus is None else set(deja_vus)
    lignes = []

    for nom, config in INTENTIONS.items():
        quota = max(1, round(total * config["poids"] / 100))
        obtenus, essais = 0, 0
        limite = quota * 60

        while obtenus < quota and essais < limite:
            essais += 1
            gabarit = rng.choice(pool[nom])
            texte, niveau = bruiter(remplir(gabarit, rng), rng)
            cle = texte.lower().strip()
            if cle in vus:
                continue
            vus.add(cle)
            lignes.append({
                "texte": texte,
                "intention": nom,
                "difficulte": niveau,
                "origine": "gabarit",
            })
            obtenus += 1

        if obtenus < quota:
            print(f"  attention : {nom} plafonne à {obtenus}/{quota} "
                  f"phrases uniques (trop peu de gabarits)")

    rng.shuffle(lignes)
    return lignes


def dedupliquer(lignes: list[dict], deja_vus: set[str] | None = None) -> list[dict]:
    """Supprime les phrases en double.

    Sur un jeu de test, un doublon fausse la mesure : une formulation
    présente dix fois pèse dix fois dans le F1, et le score reflète autant
    le tirage aléatoire que la qualité du modèle.
    """
    vus = set() if deja_vus is None else set(deja_vus)
    uniques = []
    for ligne in lignes:
        cle = ligne["texte"].lower().strip()
        if cle in vus:
            continue
        vus.add(cle)
        uniques.append(ligne)
    return uniques


def ecrire(chemin: Path, lignes: list[dict]) -> None:
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with chemin.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["texte", "intention", "difficulte", "origine"]
        )
        writer.writeheader()
        writer.writerows(lignes)


def main() -> int:
    rng = random.Random(GRAINE)
    pool_train, pool_test = repartir_gabarits(rng)

    train = generer(pool_train, TOTAL_TRAIN, rng)
    textes_train = {l["texte"].lower().strip() for l in train}

    val = generer(pool_train, TOTAL_VAL, rng, textes_train)

    # Le test ne doit contenir aucune phrase déjà vue ailleurs : le bruit
    # aléatoire peut faire converger deux gabarits vers la même phrase.
    deja = textes_train | {l["texte"].lower().strip() for l in val}
    test = generer(pool_test, TOTAL_TEST, rng, deja)

    # Les cas ambigus sont réservés au test : ils mesurent la limite du
    # modèle, ils ne servent pas à l'entraîner.
    for phrase, intention, _note in CAS_AMBIGUS:
        for variante in range(3):
            texte, _ = bruiter(phrase, rng) if variante else (phrase, "facile")
            test.append({
                "texte": texte,
                "intention": intention,
                "difficulte": "ambigu",
                "origine": "cas_ambigu",
            })
    test = dedupliquer(test)
    rng.shuffle(test)

    ecrire(SORTIE_RAW / "intents_train.csv", train)
    ecrire(SORTIE_RAW / "intents_val.csv", val)
    ecrire(SORTIE_EVAL / "intents_test_a_relire.csv", test)

    # Note explicative des cas ambigus, pour l'analyse d'erreurs
    notes = SORTIE_EVAL / "cas_ambigus.md"
    with notes.open("w", encoding="utf-8") as f:
        f.write("# Cas volontairement ambigus\n\n")
        f.write("Étiquette retenue et raison de l'ambiguïté.\n\n")
        for phrase, intention, note in CAS_AMBIGUS:
            f.write(f"- « {phrase} » : **{intention}**. {note}\n")

    print(f"train : {len(train)} lignes")
    print(f"val   : {len(val)} lignes")
    print(f"test  : {len(test)} lignes")
    print("\nGabarits réservés au test, par intention :")
    for nom in INTENTIONS:
        print(f"  {nom:26} {len(pool_test[nom])} test / "
              f"{len(pool_train[nom])} train")
    return 0


if __name__ == "__main__":
    sys.exit(main())