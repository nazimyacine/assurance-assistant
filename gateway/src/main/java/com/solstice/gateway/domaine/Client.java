package com.solstice.gateway.domaine;

import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;

/**
 * Client fictif de la démonstration, semé par data.sql. La formule est
 * le contexte injecté dans la recherche documentaire du service IA.
 *
 * <p>Entité en lecture seule de fait : la passerelle ne crée ni ne
 * modifie jamais de client.</p>
 */
@Entity
@Table(name = "clients")
public class Client {

    @Id
    private String id;

    private String nom;

    private String formule;

    protected Client() {
        // Requis par JPA.
    }

    public String getId() {
        return id;
    }

    public String getNom() {
        return nom;
    }

    public String getFormule() {
        return formule;
    }
}