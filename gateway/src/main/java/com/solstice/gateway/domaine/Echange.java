package com.solstice.gateway.domaine;

import java.time.Instant;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

/**
 * Une ligne du journal par message traité : l'exigence de journalisation
 * du cadrage (chemin, intention, confiance, latences), requêtable en SQL
 * dans la console H2 pendant la démonstration.
 *
 * <p>La réponse de l'assistant n'est volontairement pas journalisée :
 * ce qui intéresse l'observation, c'est le routage et les latences,
 * pas le texte.</p>
 */
@Entity
@Table(name = "echanges")
public class Echange {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String sessionId;

    private Instant recuLe;

    private String message;

    private String chemin;

    private String intention;

    private Double confiance;

    private String formule;

    private Integer latenceClassificationMs;

    private Integer latenceRechercheMs;

    private Integer latenceGenerationMs;

    private Integer latenceTotaleMs;

    protected Echange() {
        // Requis par JPA.
    }

    public Echange(String sessionId, String message) {
        this.sessionId = sessionId;
        this.message = message;
        this.recuLe = Instant.now();
    }

    public String getSessionId() {
        return sessionId;
    }

    public void renseigner(String chemin, String intention, Double confiance,
                           String formule, Integer classificationMs,
                           Integer rechercheMs, Integer generationMs,
                           Integer totaleMs) {
        this.chemin = chemin;
        this.intention = intention;
        this.confiance = confiance;
        this.formule = formule;
        this.latenceClassificationMs = classificationMs;
        this.latenceRechercheMs = rechercheMs;
        this.latenceGenerationMs = generationMs;
        this.latenceTotaleMs = totaleMs;
    }
}