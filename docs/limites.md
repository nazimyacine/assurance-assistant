Le jeu de test est généré, pas annoté. Les étiquettes sont exactes par construction. La relecture manuelle a servi à détecter des artefacts de génération (élisions, accords, doublons, fuites train/test), corrigés dans le générateur plutôt que ligne par ligne. Sur un vrai corpus annoté, les scores seraient plus bas et l'accord inter-annotateurs deviendrait une métrique à part entière.

La validation partageait initialement ses gabarits avec l'entraînement. Son F1 saturait à 1,0 dès la première epoch et la sélection du meilleur modèle conservait un modèle sous-entraîné. Les gabarits sont désormais répartis en trois pools strictement disjoints, et la perte de validation est suivie en plus du F1.

- au niveau des 64 unités indépendantes, l'avantage de CamemBERT
  (16 unités gagnées contre 8) n'atteint pas la signification
  statistique (test des signes, p 0,15). Le jeu de test peut
  constater l'écart, pas le démontrer.
- le seuil de rejet filtre la confusion, pas l'ambiguïté : les trois
  seuls groupes d'erreurs à confiance supérieure à 0,80 sont des cas
  volontairement ambigus, où le modèle choisit une lecture défendable
  avec aplomb.

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
  hors corpus repose principalement sur les consignes de génération ;
  le SEUIL_PERTINENCE du routeur n'est qu'un filet pour les extrêmes.