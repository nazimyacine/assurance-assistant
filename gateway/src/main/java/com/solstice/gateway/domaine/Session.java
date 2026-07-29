package com.solstice.gateway.domaine;

import java.time.Instant;
import java.util.UUID;

import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.Lob;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Table;

/**
 * Une session = une conversation. C'est ici que vit l'état du dialogue
 * entre deux messages : le JSON que FastAPI rend est conservé tel quel
 * dans {@code etatJson} et renvoyé tel quel au message suivant.
 *
 * <p>Choix délibéré : la passerelle ne DÉSÉRIALISE PAS cet état. Elle le
 * transporte comme une chaîne opaque. Le format appartient au service
 * IA ; le jour où il ajoute un champ, la passerelle n'a rien à changer,
 * et elle ne peut pas corrompre ce qu'elle ne lit pas.</p>
 */
@Entity
@Table(name = "sessions")
public class Session {

    @Id
    private String id;

    /** Nul en mode visiteur : aucun contexte de formule. */
    @ManyToOne(fetch = FetchType.EAGER)
    @JoinColumn(name = "client_id")
    private Client client;

    @Lob
    private String etatJson;

    private Instant creeeLe;

    private Instant activeLe;

    protected Session() {
        // Requis par JPA.
    }

    public Session(Client client) {
        this.id = UUID.randomUUID().toString();
        this.client = client;
        this.creeeLe = Instant.now();
        this.activeLe = this.creeeLe;
    }

    public void enregistrerEtat(String etatJson) {
        this.etatJson = etatJson;
        this.activeLe = Instant.now();
    }

    public String getId() {
        return id;
    }

    public Client getClient() {
        return client;
    }

    public String getEtatJson() {
        return etatJson;
    }

    public Instant getCreeeLe() {
        return creeeLe;
    }

    public Instant getActiveLe() {
        return activeLe;
    }
}