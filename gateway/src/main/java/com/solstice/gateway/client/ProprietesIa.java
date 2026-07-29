package com.solstice.gateway.client;

import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * Le bloc {@code ia:} de application.yml, typé. Une faute de frappe dans
 * le YAML devient une erreur de démarrage, pas une valeur par défaut
 * silencieuse.
 *
 * @param baseUrl          racine du service IA, sans barre finale
 * @param connectTimeoutMs délai d'établissement de la connexion
 * @param readTimeoutMs    délai de réponse complète ; couvre la
 *                         génération LLM mesurée (jusqu'à 1600 ms) et
 *                         les réessais du palier gratuit Mistral
 */
@ConfigurationProperties(prefix = "ia")
public record ProprietesIa(String baseUrl,
                           int connectTimeoutMs,
                           int readTimeoutMs) {
}