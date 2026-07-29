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