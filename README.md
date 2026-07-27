# Assistant assurance : classification d'intentions + RAG

Assistant conversationnel pour une mutuelle santé fictive. Il classe l'intention
du message avec un CamemBERT fine-tuné, puis route soit vers un flux métier
guidé, soit vers une recherche documentaire (RAG) dans les conditions générales.

Projet personnel. Compagnie, documents et données d'entraînement entièrement
synthétiques. Aucune donnée, aucun code et aucun document d'entreprise.

## Avancement

- [x] Étape 0 : socle Docker et dépôt
- [ ] Étape 1 : corpus documentaire
- [ ] Étape 2 : jeu d'intentions
- [ ] Étape 3 : baseline TF-IDF
- [ ] Étape 4 : fine-tuning CamemBERT
- [ ] Étape 5 : évaluation de la classification
- [ ] Étape 6 : indexation vectorielle
- [ ] Étape 7 : recherche hybride et génération
- [ ] Étape 8 : évaluation du RAG
- [ ] Étape 9 : routeur
- [ ] Étape 10 : API FastAPI
- [ ] Étape 11 : passerelle Spring Boot
- [ ] Étape 12 : front Angular

## Démarrage

    copy .env.example .env
    docker compose up -d