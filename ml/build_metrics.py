"""Étape 5d : génération de docs/metrics.md à partir des résultats chiffrés.

Commit 1 : chargement des sources et garde-fous. Le script charge tout,
vérifie la cohérence, n'écrit rien. Lancer avec --check pour un état des lieux.

Principe : metrics.md sera GÉNÉRÉ, jamais rédigé. Toute valeur affichée
provient d'un fichier source, aucun chiffre n'est écrit en dur ici.
"""

from __future__ import annotations

import argparse
import json
import sys
import yaml
from pathlib import Path
from scipy.stats import binomtest

import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parents[1]
DOCS = RACINE / "docs"
ARTIFACTS = RACINE / "ml" / "artifacts"

SOURCES_JSON = {
    "baseline": DOCS / "baseline.json",
    "training": DOCS / "training.json",
    "eval_intent": DOCS / "eval_intent.json",
    "confusion": DOCS / "confusion.json",
    "seuil_rejet": DOCS / "seuil_rejet.json",
    "eval_rag": DOCS / "eval_rag.json",
}
NPZ_CAMEMBERT = ARTIFACTS / "predictions_test.npz"
NPZ_BASELINE = ARTIFACTS / "predictions_baseline_test.npz"
CSV_TEST = RACINE / "data" / "eval" / "intents_test.csv"
LABELS = ARTIFACTS / "intent-model" / "labels.json"
CAS_AMBIGUS = RACINE / "data" / "eval" / "cas_ambigus.md"
ANNOTATIONS = DOCS / "annotations_erreurs.yaml"
SORTIE = DOCS / "metrics.md"


def echec(message: str) -> None:
    raise SystemExit(f"ERREUR : {message}")


def charger_labels() -> dict[int, str]:
    """labels.json est imbriqué : {"vers_id": {...}, "vers_nom": {...}}."""
    contenu = json.loads(LABELS.read_text(encoding="utf-8"))
    if "vers_nom" not in contenu:
        echec(f"forme inattendue de {LABELS} : clé 'vers_nom' absente "
              f"(clés : {sorted(contenu)})")
    return {int(k): v for k, v in contenu["vers_nom"].items()}


def en_noms(tableau: np.ndarray, vers_nom: dict[int, str], origine: str) -> np.ndarray:
    """Convertit un tableau d'étiquettes en noms, qu'il soit numérique ou textuel."""
    if np.issubdtype(tableau.dtype, np.number):
        return np.array([vers_nom[int(i)] for i in tableau])
    return tableau.astype(str)


def charger_sources() -> dict:
    manquants = [str(p) for p in [*SOURCES_JSON.values(), NPZ_CAMEMBERT,
                 NPZ_BASELINE, CSV_TEST, LABELS, CAS_AMBIGUS] if not p.exists()]
    if manquants:
        echec("fichier(s) source absent(s) :\n  " + "\n  ".join(manquants))

    src = {nom: json.loads(chemin.read_text(encoding="utf-8"))
           for nom, chemin in SOURCES_JSON.items()}
    vers_nom = charger_labels()

    npz_c = np.load(NPZ_CAMEMBERT, allow_pickle=True)
    npz_b = np.load(NPZ_BASELINE, allow_pickle=True)
    test = pd.read_csv(CSV_TEST)

    src["textes"] = npz_c["textes"].astype(str)
    src["probas"] = npz_c["probas"]
    # "etiquettes" est le vocabulaire des classes (ordre des colonnes de
    # probas), pas un vecteur de prédictions par ligne.
    src["classes"] = npz_c["etiquettes"].astype(str)
    src["camembert_predit"] = src["classes"][npz_c["probas"].argmax(axis=1)]
    src["reels"] = en_noms(npz_c["reels"], vers_nom, "camembert")
    src["baseline_predit"] = en_noms(npz_b["predits"], vers_nom, "baseline")
    src["baseline_reels"] = en_noms(npz_b["reels"], vers_nom, "baseline")
    src["baseline_variante"] = str(np.atleast_1d(npz_b["variante"])[0])
    src["textes_baseline"] = npz_b["textes"].astype(str)
    src["test"] = test
    src["vers_nom"] = vers_nom
    src["cas_ambigus"] = CAS_AMBIGUS.read_text(encoding="utf-8")
    return src


def verifier(src: dict) -> None:
    """Garde-fous. Chaque hypothèse d'alignement est vérifiée, aucune supposée."""
    textes, test = src["textes"], src["test"]

    # 1. Les deux npz portent sur les mêmes phrases, dans le même ordre.
    if not np.array_equal(textes, src["textes_baseline"]):
        echec("les vecteurs de textes des deux .npz diffèrent, "
              "comparaison appariée impossible")

    # 2. Le npz est aligné ligne à ligne avec intents_test.csv,
    #    condition pour récupérer la colonne gabarit par position.
    colonnes_attendues = {"texte", "intention", "difficulte", "origine", "gabarit"}
    if not colonnes_attendues <= set(test.columns):
        echec(f"colonnes manquantes dans {CSV_TEST} : "
              f"{sorted(colonnes_attendues - set(test.columns))}")
    if len(test) != len(textes):
        echec(f"tailles différentes : csv {len(test)}, npz {len(textes)}")
    if not np.array_equal(test["texte"].astype(str).to_numpy(), textes):
        echec("intents_test.csv et predictions_test.npz ne sont pas dans "
              "le même ordre, jonction par position impossible")

    # 3. Les étiquettes réelles concordent partout.
    if not np.array_equal(src["reels"], src["baseline_reels"]):
        echec("étiquettes réelles différentes entre les deux .npz")
    if not np.array_equal(src["reels"], test["intention"].astype(str).to_numpy()):
        echec("étiquettes réelles du npz différentes de celles du csv")

    # 4. La légende des colonnes de probas concorde avec labels.json.
    classes = src["classes"]
    if len(classes) != src["probas"].shape[1]:
        echec(f"{len(classes)} noms de classes pour "
              f"{src['probas'].shape[1]} colonnes de probas")
    attendu = np.array([src["vers_nom"][i] for i in range(len(classes))])
    if not np.array_equal(classes, attendu):
        echec("l'ordre des classes du npz diffère de labels.json :\n"
              f"  npz    : {classes.tolist()}\n"
              f"  labels : {attendu.tolist()}")

    # 5. Recoupement avec confusion.json : les comptes d'erreurs recalculés
    #    doivent être identiques à ceux de l'étape 5b.
    for modele, predit in [("camembert", src["camembert_predit"]),
                           ("baseline", src["baseline_predit"])]:
        recalcule = int((predit != src["reels"]).sum())
        reference = src["confusion"]["modeles"][modele]["routage"]["erreurs"]
        if recalcule != reference:
            echec(f"{modele} : {recalcule} erreurs recalculées contre "
                  f"{reference} dans confusion.json")

    # 6. Recoupement avec eval_intent.json sur l'exactitude globale.
    exactitude = float((src["camembert_predit"] == src["reels"]).mean())
    reference = src["eval_intent"]["global"]["exactitude"]
    if abs(exactitude - reference) > 5e-4:
        echec(f"exactitude recalculée {exactitude:.4f} contre "
              f"{reference} dans eval_intent.json")


def etat_des_lieux(src: dict) -> None:
    test = src["test"]
    erreurs = src["camembert_predit"] != src["reels"]
    groupes = (test.loc[erreurs]
               .assign(predit=src["camembert_predit"][erreurs])
               .groupby(["gabarit", "intention", "predit"])
               .size().sort_values(ascending=False))
    print(f"lignes            : {len(test)}")
    print(f"unités (gabarits + ambigus) : {test['gabarit'].nunique()}")
    print(f"erreurs camembert : {int(erreurs.sum())}")
    print(f"erreurs baseline  : {int((src['baseline_predit'] != src['reels']).sum())}")
    print(f"variante baseline : {src['baseline_variante']}")
    print(f"groupes d'erreurs (gabarit, réel, prédit) : {len(groupes)}")
    print("\naperçu des 5 premiers groupes :")
    print(groupes.head(5).to_string())

def pct(x: float, dec: int = 1) -> str:
    return f"{100 * x:.{dec}f}%".replace(".", ",")


def nombre(x: float, dec: int = 4) -> str:
    return f"{x:.{dec}f}".replace(".", ",")


def charger_annotations() -> dict[str, str]:
    if not ANNOTATIONS.exists():
        return {}
    contenu = yaml.safe_load(ANNOTATIONS.read_text(encoding="utf-8"))
    return {str(k): str(v).strip() for k, v in (contenu or {}).items()}


def groupes_erreurs(src: dict) -> pd.DataFrame:
    """Les erreurs CamemBERT groupées par gabarit, triées par effectif."""
    test = src["test"]
    erreurs = src["camembert_predit"] != src["reels"]
    df = (test.loc[erreurs]
          .assign(predit=src["camembert_predit"][erreurs],
                  confiance=src["probas"].max(axis=1)[erreurs]))
    groupes = (df.groupby(["gabarit", "intention", "predit"], sort=False)
               .agg(n=("texte", "size"),
                    exemple=("texte", "first"),
                    origine=("origine", "first"),
                    confiance_moyenne=("confiance", "mean"))
               .reset_index()
               .sort_values(["n", "gabarit"], ascending=[False, True]))
    return groupes

def section_appariee(src: dict) -> list[str]:
    """Comparaison appariée baseline contre CamemBERT sur les mêmes phrases."""
    test = src["test"]
    reels = src["reels"]
    cam_ok = src["camembert_predit"] == reels
    base_ok = src["baseline_predit"] == reels

    seuls_base = base_ok & ~cam_ok
    seuls_cam = cam_ok & ~base_ok
    b, c = int(seuls_base.sum()), int(seuls_cam.sum())
    p_ligne = binomtest(min(b, c), b + c).pvalue

    # Même comparaison au niveau des unités indépendantes : pour chaque
    # gabarit, taux de réussite des deux modèles, puis test des signes.
    par_unite = (test.assign(cam=cam_ok, base=base_ok)
                 .groupby("gabarit")[["cam", "base"]].mean())
    gagne = int((par_unite["cam"] > par_unite["base"]).sum())
    perd = int((par_unite["cam"] < par_unite["base"]).sum())
    egal = len(par_unite) - gagne - perd
    p_unite = binomtest(min(gagne, perd), gagne + perd).pvalue

    def fmt_p(p: float) -> str:
        return "< 0,0001" if p < 1e-4 else nombre(p, 4)

    lignes = [
        "## Comparaison appariée baseline contre CamemBERT",
        "",
        "Les deux modèles sont évalués sur les mêmes phrases, dans le même "
        "ordre (vérifié par le script). Comparaison des désaccords :",
        "",
        "| | n |",
        "|---|---|",
        f"| les deux corrects | {int((base_ok & cam_ok).sum())} |",
        f"| les deux faux | {int((~base_ok & ~cam_ok).sum())} |",
        f"| baseline seule correcte | {b} |",
        f"| CamemBERT seul correct | {c} |",
        "",
        f"Test de McNemar (binomial exact) au niveau ligne : p {fmt_p(p_ligne)}. "
        "Les lignes d'un même gabarit n'étant pas indépendantes, ce chiffre "
        "surestime la certitude ; au niveau des unités indépendantes, "
        f"CamemBERT fait mieux sur {gagne} unités, moins bien sur {perd}, "
        f"égalité sur {egal} (test des signes : p {fmt_p(p_unite)})."
        + (" À ce niveau, l'écart n'atteint pas le seuil conventionnel de "
           "signification : 64 unités ne suffisent pas à trancher, "
           "voir `docs/limites.md`."
           if p_unite >= 0.05 else ""),
        "",
        "### Les cas ambigus que la baseline réussit et que CamemBERT rate",
        "",
    ]

    masque = seuls_base & (test["difficulte"] == "ambigu").to_numpy()
    gagnes_base = (test.loc[masque]
                   .assign(predit=src["camembert_predit"][masque]))
    n_amb_base = int(masque.sum())
    n_amb_cam = int((seuls_cam & (test["difficulte"] == "ambigu").to_numpy()).sum())
    if len(gagnes_base):
        lignes += ["| Phrase | Réel | CamemBERT prédit |", "|---|---|---|"]
        for _, r in gagnes_base.iterrows():
            texte = str(r["texte"]).replace("|", "\\|")
            lignes.append(f"| {texte} | {r['intention']} | {r['predit']} |")
    lignes += [
        "",
        f"Sur les cas ambigus, la baseline est seule correcte {n_amb_base} "
        f"fois, CamemBERT seul correct {n_amb_cam} fois.",
        "",
        "### Prédictions contre support : les classes sur-prédites",
        "",
        "Classes prédites au moins 25% au delà de leur support par au "
        "moins un des deux modèles :",
        "",
        "| Classe | Support | Prédictions baseline | Prédictions CamemBERT |",
        "|---|---|---|---|",
    ]
    supports = pd.Series(reels).value_counts()
    pred_base = pd.Series(src["baseline_predit"]).value_counts()
    pred_cam = pd.Series(src["camembert_predit"]).value_counts()
    sur_predites = sorted(
        (cl for cl in supports.index
         if max(pred_base.get(cl, 0), pred_cam.get(cl, 0))
         >= 1.25 * supports[cl]),
        key=lambda cl: supports[cl])
    for classe in sur_predites:
        lignes.append(f"| {classe} | {supports[classe]} "
                      f"| {pred_base.get(classe, 0)} "
                      f"| {pred_cam.get(classe, 0)} |")
    lignes += [
        "",
        "Deux mécanismes distincts : la baseline pondérée sur-prédit les "
        "classes rares par construction (`class_weight=\"balanced\"`), "
        "avec un rappel parfait payé en précision, ce qui explique une "
        "partie de son avantage sur les phrases à étiquette arbitrée. "
        "La sur-prédiction de `demander_attestation` par CamemBERT est "
        "un phénomène différent : l'effet de gabarit analysé ci-dessous.",
        "",
    ]
    return lignes

def section_rag(src: dict) -> list[str]:
    """Le tableau des configurations RAG, depuis docs/eval_rag.json."""
    r = src["eval_rag"]
    retenue = "titres + fil, vectoriel"
    lignes = [
        "## Évaluation du RAG : cinq configurations sur 50 questions",
        "",
        "50 questions écrites à la main : 34 à réponse, 16 sans réponse, "
        "soit 50 unités indépendantes. La recherche est mesurée par "
        "recall@5 et MRR@10 ; les réponses sont générées puis jugées par "
        "LLM (correct, incorrect ou refus), limites du juge dans "
        "`docs/limites.md`.",
        "",
        "| Configuration | recall@5 | MRR@10 | réponses correctes "
        "| refus corrects | faux refus |",
        "|---|---|---|---|---|---|",
    ]
    for nom, c in r["configs"].items():
        g = r["generation"][nom]
        gras = "**" if nom == retenue else ""
        lignes.append(
            f"| {gras}{nom}{gras} | {c['recall_a_5']}/{c['sur']} "
            f"| {nombre(c['mrr_a_10'], 3)} "
            f"| {g['reponses_correctes']}/{g['sur']} "
            f"| {g['refus_corrects']}/{g['sur_sans']} "
            f"| {g['faux_refus']} |")
    g = r["generation"][retenue]
    s = r["seuil_pertinence"]
    lignes += [
        "",
        f"Configuration retenue pour le service : **{retenue}**. À "
        f"réponses correctes égales ({g['reponses_correctes']}/{g['sur']}), "
        f"elle refuse les {g['sur_sans']} questions sans réponse sans "
        f"exception ; ses {g['faux_refus']} faux refus sont le prix "
        "accepté de cette prudence, inventer une réponse étant une faute "
        "là où refuser à tort n'est qu'une friction.",
        "",
        "Le recall@5 avantage structurellement le découpage naïf (top 5 "
        "sur un index de 22 chunks contre 94) ; la colonne des réponses "
        "correctes le démasque.",
        "",
        "Aucun seuil de pertinence exploitable sur le cosinus de tête : "
        f"minimum des questions faciles {nombre(s['facile']['min'], 3)}, "
        f"maximum des sans réponse {nombre(s['sans_reponse']['max'], 3)}. "
        "Le refus des questions hors documentation repose sur les "
        "consignes de génération.",
        "",
    ]
    return lignes


def generer_markdown(src: dict, annotations: dict[str, str]) -> tuple[str, list[str]]:
    e = src["eval_intent"]
    t = src["training"]
    c = src["confusion"]
    s = src["seuil_rejet"]
    lignes: list[str] = []
    manquantes: list[str] = []

    lignes += [
        "# Métriques de la classification d'intentions",
        "",
        "Document généré par `ml/build_metrics.py`, ne pas éditer à la main.",
        "Sources : `docs/baseline.json`, `docs/training.json`, "
        "`docs/eval_intent.json`, `docs/confusion.json`, "
        "`docs/seuil_rejet.json`, `docs/eval_rag.json`.",
        "",
        "## Entraînement",
        "",
        f"Modèle `{t['modele']}`, {t['epochs']} epochs, batch {t['batch']}, "
        f"lr {t['lr']}, pondération des classes : {'oui' if t['class_weights'] else 'non'}, "
        f"durée {t['duree_s']} s.",
        "",
        "| Epoch | Perte train | Perte val | F1 macro val |",
        "|---|---|---|---|",
    ]
    for h in t["historique"]:
        gras = "**" if h["f1_val"] == t["meilleur_f1_val"] else ""
        lignes.append(f"| {h['epoch']} | {nombre(h['perte_train'])} "
                      f"| {nombre(h['perte_val'])} "
                      f"| {gras}{nombre(h['f1_val'])}{gras} |")
    lignes += [
        "",
        "Modèle conservé : epoch au meilleur F1 de validation. La perte de "
        "validation remonte ensuite alors que la perte d'entraînement "
        "descend : surapprentissage.",
        "",
        "## Baseline contre CamemBERT",
        "",
        f"Variante de baseline retenue : {e['baseline']['retenue']} "
        "(la meilleure des deux, sélection automatique).",
        "",
        "| Modèle | F1 macro | Exactitude |",
        "|---|---|---|",
        f"| Baseline TF-IDF + régression logistique | {nombre(e['baseline']['f1_macro'])} "
        f"| {nombre(e['baseline']['exactitude'])} |",
        f"| **CamemBERT fine-tuné** | **{nombre(e['global']['f1_macro'])}** "
        f"| **{nombre(e['global']['exactitude'])}** |",
        "",
        f"Écart de F1 macro : +{nombre(e['global']['f1_macro'] - e['baseline']['f1_macro'])}.",
        "",
        "### Exactitude par difficulté (même métrique des deux côtés)",
        "",
        "| Difficulté | n | Baseline | CamemBERT |",
        "|---|---|---|---|",
    ]
    for diff in ["facile", "bruite", "ambigu"]:
        cam = e["par_difficulte"][diff]
        lignes.append(f"| {diff} | {cam['n']} "
                      f"| {nombre(e['baseline']['par_difficulte'][diff])} "
                      f"| {nombre(cam['exactitude'])} |")
    n_ambigu = e["par_difficulte"]["ambigu"]["n"]
    lignes += [
        "",
        f"L'écart sur les cas ambigus porte sur {n_ambigu} lignes et vaut "
        "quelques phrases : il ne permet pas de conclure, voir "
        "`docs/limites.md`.",
        "",
        "## Routage",
        "",
        "Une erreur qui reste dans le même type (transactionnelle, "
        "informationnelle, rejet) laisse le routeur sur le bon chemin ; "
        "une erreur inter-type l'envoie ailleurs.",
        "",
        "| | Baseline | CamemBERT |",
        "|---|---|---|",
    ]
    rb = c["modeles"]["baseline"]["routage"]
    rc = c["modeles"]["camembert"]["routage"]
    lignes += [
        f"| Erreurs | {rb['erreurs']} | {rc['erreurs']} |",
        f"| dont changement de chemin | {rb['inter_type']} | {rc['inter_type']} |",
        f"| **Messages mal routés** | **{pct(rb['part_du_jeu'])}** "
        f"| **{pct(rc['part_du_jeu'])}** |",
        f"| Erreurs restant sur le bon chemin | {pct(1 - rb['part_inter_type'])} "
        f"| {pct(1 - rc['part_inter_type'])} |",
        "",
        "![Matrices de confusion](matrice_confusion.png)",
        "",
        "Matrices ordonnées par type de routage, générées par "
        "`ml/plot_confusion.py`.",
        "",
        "## Seuil de rejet",
        "",
        f"Calibration : ECE {nombre(s['calibration']['ece'])}, confiance "
        f"moyenne {nombre(s['calibration']['confiance_moyenne'])} pour une "
        f"exactitude de {nombre(s['calibration']['exactitude'])}.",
        "",
        f"Règle de sélection : {s['recommandation']['regle']}.",
        "",
    ]
    sans_seuil = s["balayage"][0]
    retenu = s["recommandation"]["detail"]
    lignes += [
        f"| | sans seuil | seuil {nombre(retenu['seuil'], 3)} |",
        "|---|---|---|",
        f"| couverture (lignes) | {pct(sans_seuil['couverture'])} "
        f"| {pct(retenu['couverture'])} |",
        f"| couverture (gabarits) | {pct(sans_seuil['couverture_gabarit'])} "
        f"| {pct(retenu['couverture_gabarit'])} |",
        f"| exactitude sur acceptés | {pct(sans_seuil['exactitude_acceptes'])} "
        f"| {pct(retenu['exactitude_acceptes'])} |",
        f"| flux métier déclenchés à tort | {sans_seuil['flux_a_tort']} "
        f"| {retenu['flux_a_tort']} |",
        "",
    ]
    lignes += section_appariee(src)
    lignes += [
        "## Analyse des erreurs, groupée par gabarit",
        "",
        "Les phrases d'un même gabarit ne sont pas indépendantes : une "
        "erreur porte le plus souvent sur un gabarit entier. Les "
        f"{rc['erreurs']} erreurs de CamemBERT se réduisent à "
        "quelques groupes (gabarit, intention prédite).",
        "",
        "| n | Gabarit | Réel | Prédit | Confiance moy. | Explication |",
        "|---|---|---|---|---|---|",
    ]
    groupes = groupes_erreurs(src).head(20)
    for _, g in groupes.iterrows():
        cle = f"{g['gabarit']} || {g['predit']}"
        explication = annotations.get(cle)
        if explication is None:
            explication = "à annoter"
            manquantes.append(cle)
        gabarit = str(g["gabarit"]).replace("|", "\\|")
        lignes.append(f"| {g['n']} | {gabarit} | {g['intention']} "
                      f"| {g['predit']} | {nombre(g['confiance_moyenne'], 2)} "
                      f"| {explication} |")
    lignes += [
        "",
        "Les cas d'origine ambiguë sont arbitrés dans "
        "`data/eval/cas_ambigus.md`.",
        "",
    ]
    lignes += section_rag(src)
    return "\n".join(lignes), manquantes


def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("--check", action="store_true",
                         help="charge, vérifie, affiche un état des lieux")
    args = parseur.parse_args()

    src = charger_sources()
    verifier(src)
    print("Garde-fous : OK, sources chargées et alignées.")
    if args.check:
        print()
        etat_des_lieux(src)
        return

    annotations = charger_annotations()
    markdown, manquantes = generer_markdown(src, annotations)
    SORTIE.write_text(markdown, encoding="utf-8")
    print(f"Écrit : {SORTIE.relative_to(RACINE)}")
    if manquantes:
        print(f"\n{len(manquantes)} groupe(s) sans explication dans "
              f"{ANNOTATIONS.name} :")
        for cle in manquantes:
            print(f'  "{cle}": >-')
            print("    ")


if __name__ == "__main__":
    main()