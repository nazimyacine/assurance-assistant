package com.solstice.gateway.domaine;

import org.springframework.data.jpa.repository.JpaRepository;

/**
 * Dépôts Spring Data du domaine. Les requêtes dérivées suffisent :
 * aucune méthode à écrire, le nommage fait tout.
 */
public final class Depots {

    private Depots() {
        // Espace de noms, pas d'instance.
    }

    public interface Clients extends JpaRepository<Client, String> {
    }

    public interface Sessions extends JpaRepository<Session, String> {
    }

    public interface Echanges extends JpaRepository<Echange, Long> {
    }
}