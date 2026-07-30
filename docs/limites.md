# Limites connues

Ce document dit ce que les chiffres de `docs/metrics.md` ne démontrent pas, et
ce que le système ne fait pas. C'est le seul fichier de résultats du dépôt qui
soit rédigé à la main : `metrics.md` est généré par `ml/build_metrics.py` et ne
peut pas mentir sans qu'un script s'en aperçoive, celui-ci le peut. Une limite
qui cesse d'être vraie doit donc être retirée, jamais atténuée.

## Les données

**Le jeu de test est généré, pas annoté.** Les étiquettes sont exactes par
construction. La relecture manuelle a servi à détecter des artefacts de
génération (élisions, accords, doublons, fuites train/test), corrigés dans le
générateur plutôt que ligne par ligne. Sur un vrai corpus annoté, les scores
seraient plus bas et l'accord inter-annotateurs deviendrait une métrique à part
entière.

**Tout est synthétique.** La compagnie, les onze documents du corpus, les 201
gabarits d'intentions et les 50 questions d'évaluation ont été écrits pour ce
projet. Les formulations d'utilisateurs sont donc imaginées, y compris les
variantes bruitées : personne n'a mesuré leur distance à ce qu'écrivent de vrais
assurés. Les chiffres décrivent le comportement du système sur ce jeu, pas sur
du trafic réel.

**Le corpus est plus régulier qu'une documentation d'assurance réelle.** Onze
documents, environ 6 000 mots, des sections courtes et autosuffisantes, aucun
tableau, aucun renvoi contradictoire, aucune version périmée qui traîne. Une
part du bon score de la recherche vectorielle vient de cette régularité : des
sections thématiques et séparées se distinguent bien dans l'espace des
plongements. Un corpus réel dégraderait la recherche avant de dégrader le
modèle.

**Les flux métier sont simulés.** Résilier ne résilie rien, une demande de prise
en charge n'ouvre aucun dossier : les sept dialogues collectent des informations
et confirment, sans système de gestion derrière. Ce qui est démontré est le
dialogue à état et son articulation avec la classification, pas l'exécution.

## La classification

**La validation partageait initialement ses gabarits avec l'entraînement.** Son
F1 saturait à 1,0 dès la première epoch et la sélection du meilleur modèle
conservait un modèle sous-entraîné. Les gabarits sont désormais répartis en
trois pools strictement disjoints, et la perte de validation est suivie en plus
du F1.

**Le F1 de test dépasse celui de validation** (0,8425 contre 0,8087). C'est
inhabituel et ce n'est pas un bon signe : les trois pools de gabarits étant
disjoints, ils n'ont aucune raison d'avoir la même difficulté, et rien n'a été
fait pour l'équilibrer. Aucun des deux chiffres ne doit être lu comme une
estimation de la performance en production.

**La taille effective du jeu de test est de 64 unités indépendantes**, pas 536
lignes : les phrases d'un même gabarit se ressemblent trop pour compter chacune
pour une observation. Conséquence directe :

- au niveau des 64 unités indépendantes, l'avantage de CamemBERT
  (16 unités gagnées contre 8) n'atteint pas la signification
  statistique (test des signes, p 0,15). Le jeu de test peut
  constater l'écart, pas le démontrer.
- le seuil de rejet filtre la confusion, pas l'ambiguïté : les trois
  seuls groupes d'erreurs à confiance supérieure à 0,80 sont des cas
  volontairement ambigus, où le modèle choisit une lecture défendable
  avec aplomb.

**La calibration est mauvaise** (ECE 0,1058). Il n'y a pas de surconfiance
globale, la confiance moyenne (0,8537) étant proche de l'exactitude (0,8787),
mais la confiance ne se lit pas comme une probabilité. Le seuil de 0,80 est
heuristique : il vient d'une règle de couverture puis de rendement marginal, pas
d'une garantie probabiliste. À noter aussi que le modèle ne descend jamais sous
0,30 sur l'ensemble du jeu de test : il n'a pas de « je ne sais pas », seulement
des préférences plus ou moins marquées.

**Le taux de messages mal routés (4,9%) est un plancher, pas une prévision.** Il
est mesuré sur des phrases issues des mêmes gabarits que l'entraînement, dans
une taxonomie de douze intentions qui recouvre exactement le périmètre simulé.
Un utilisateur réel écrit des messages qui n'appartiennent à aucune classe, en
enchaîne deux dans la même phrase, ou change d'avis en cours de route.

**La taxonomie ne comporte aucune intention de politesse ou de clôture.** Un
message purement civil est donc hors des douze classes par construction, et le
classifieur le range dans la classe la plus proche avec une confiance élevée,
au-dessus du seuil de rejet : "c bon merci" sort à 0,876 sur
suivre_remboursement. Un pré-filtre lexical déterministe intercepte une liste
fermée de formules courantes avant la classification. Il ne corrige pas le fond
du problème et ne modifie aucune métrique, le jeu de test ne contenant aucune
formule de politesse. La solution correcte serait une treizième intention, au
prix d'un réentraînement et de la régénération de tous les chiffres publiés.

## La recherche et la génération

- le recall@5 avantage structurellement les gros chunks : l'index naïf
  compte 22 chunks (le top 5 en couvre 23%) contre 94 pour les sections
  (5%). La comparaison honnête entre granularités est le MRR, puis le
  taux de réponses correctes.
- la recherche hybride a été corrigée après mesure (retrait des mots
  vides et des scores BM25 nuls de la fusion RRF, correction standard
  mais décidée en voyant les chiffres) : MRR titres 0,766 vers 0,773,
  titres + fil 0,831 vers 0,767. Même corrigée, elle reste sous le
  vectoriel seul (0,855) : sur un corpus de 6 000 mots aux sections
  auto-suffisantes et des questions reformulées, BM25 n'apporte pas de
  signal complémentaire en moyenne, malgré ses réussites ponctuelles
  sur les montants exacts. Aucune itération supplémentaire n'a été
  faite contre les 50 questions pour ne pas surapprendre le jeu
  d'évaluation.
- il n'existe pas de seuil de pertinence exploitable sur le score
  cosinus de tête : le minimum des questions faciles (0,814) est sous
  le maximum des questions sans réponse (0,872). Le refus des questions
  hors corpus repose donc ENTIÈREMENT sur les consignes de génération.
  `COSINUS_PLANCHER` vaut `None` dans le routeur et `/health` le publie
  tel quel : il n'y a aucun filet en dessous, pas même pour les
  extrêmes. Si le modèle de génération changeait, le refus parfait
  (16/16) serait à remesurer avant toute mise en service.
- les réponses du RAG sont jugées par le même modèle que celui qui les
  génère (mistral-small, température 0). Un juge indépendant (autre
  modèle, ou relecture humaine des 250 verdicts) serait plus solide ;
  les verdicts de la configuration retenue ont été relus à la main,
  pas ceux des quatre autres.
- les 3 faux refus de la configuration retenue sont le prix accepté du
  refus parfait sur les questions sans réponse : pour un assistant
  d'assurance, inventer une réponse est une faute, refuser à tort une
  friction. Même arbitrage que le seuil de rejet de la classification.
- deux des trois faux refus de la configuration retenue surviennent
  alors que le bon extrait est dans le prompt et cité dans le refus
  même (q15, q29) : l'excès de prudence du générateur est le revers
  des consignes strictes qui donnent le refus parfait. Le troisième
  (q25) suit un raté de recherche ; refuser était la bonne dégradation.
- l'unique réponse incorrecte (q23, tarif famille) vient d'un contexte
  partiel : un seul des deux extraits nécessaires remonté, et le modèle
  a improvisé un calcul au lieu de refuser. Le contexte partiel est
  plus dangereux que l'absence de contexte.

**Le RAG n'a jamais été évalué en multi-tours.** Les 50 questions sont posées
isolément, sans historique. Les flux métier, eux, sont multi-tours depuis
l'étape 9, mais aucune question documentaire ne dépend de la précédente : une
question de suivi comme « et en Premium ? » n'est pas résolue, faute de
réécriture de requête à partir de l'historique.

## La mise en service

**Le contexte client ne renseigne pas le classifieur.** La formule injectée par
la passerelle filtre la recherche, elle n'entre pas dans le message classé.
« quel est le remboursement pour des verres progressifs » sort à 0,502 et tombe
sous le seuil, alors que la même question suivie de « en Confort » sort à 0,806.
Un utilisateur connecté n'a donc aucun avantage sur la compréhension de sa
question, seulement sur la recherche qui suit.

**Le filtre par formule libère des places dans le top 5.** En écartant les
garanties des deux autres formules, il laisse remonter des passages plus
faibles (présentation de l'offre, autres postes de garantie) qui n'auraient pas
été retenus sans lui. Le contexte envoyé au générateur est mieux ciblé sur sa
tête, plus bruité sur sa queue.

**Les flux métier dupliquent la procédure décrite dans le corpus.** Les étapes
d'une résiliation existent à la fois dans `data/reference/flux_metier.yaml` et
dans les conditions générales indexées. La duplication est volontairement
bornée : aucun montant de garantie n'est écrit dans le YAML, ils restent
l'exclusivité du RAG, et les délais sont résolus depuis `offre.yaml` au
démarrage. Elle reste une duplication : une procédure modifiée doit l'être aux
deux endroits.

**La chaîne des délais d'attente a été inversée pendant tout le
développement.** L'appel au modèle de génération attendait 60 secondes quand la
passerelle abandonnait à 15 : la couche supérieure renonçait toujours en
premier, et la dégradation soignée du routeur ne s'exécutait pour personne, un
503 brut prenant sa place. Découvert par une panne DNS réelle, corrigé en
délais croissants du plus interne vers le plus externe (5/10 s pour le modèle,
15 s pour la passerelle, 20 s pour le front). Aucune mesure publiée n'en dépend,
la dégradation n'étant sur aucun chemin évalué.

**Le générateur produit du Markdown.** Ses consignes ne l'interdisent pas et il
met des montants en gras. La conversion est faite côté front (`gras.ts`, gras
seul, échappement d'abord), et non dans les consignes : celles-ci ont produit
les chiffres de l'étape 8, les retoucher après coup invaliderait la mesure pour
un problème d'affichage. Le jour où le front ne serait plus le seul client, la
question se reposerait à l'endroit correct.