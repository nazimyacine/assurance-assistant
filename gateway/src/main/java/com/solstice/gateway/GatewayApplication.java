package com.solstice.gateway;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;
import org.springframework.data.jpa.repository.config.EnableJpaRepositories;

/**
 * Passerelle de l'assistant Mutuelle Solstice (étape 11).
 *
 * <p>Ce que la passerelle porte, et que le service IA ne doit pas porter :
 * la session de conversation en base (le service Python reste sans état),
 * le contexte client (la formule souscrite, injectée dans la recherche
 * documentaire), la journalisation des échanges, et la dégradation propre
 * en 503 quand le service IA est indisponible.</p>
 *
 * <p>Le front n'envoie plus ni l'état du dialogue ni la formule : il
 * envoie un identifiant de session et un identifiant de client, et la
 * passerelle reconstitue le reste.</p>
 */
@SpringBootApplication
@ConfigurationPropertiesScan
// Les dépôts sont des interfaces imbriquées dans Depots, un simple espace
// de noms. L'analyse de Spring Data les ignore par défaut ; sans cette
// option, il trouve 0 dépôt et le contrôleur ne peut pas être construit.
@EnableJpaRepositories(considerNestedRepositories = true)
public class GatewayApplication {

    public static void main(String[] args) {
        SpringApplication.run(GatewayApplication.class, args);
    }
}