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

  La taxonomie ne comporte aucune intention de politesse ou de clôture. Un message purement civil est donc hors des douze classes par construction, et le classifieur le range dans la classe la plus proche avec une confiance élevée, au-dessus du seuil de rejet : "c bon merci" sort à 0,876 sur suivre_remboursement. Un pré-filtre lexical déterministe intercepte une liste fermée de formules courantes avant la classification. Il ne corrige pas le fond du problème et ne modifie aucune métrique, le jeu de test ne contenant aucune formule de politesse. La solution correcte serait une treizième intention, au prix d'un réentraînement et de la régénération de tous les chiffres publiés.