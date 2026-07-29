package com.solstice.gateway.web;

import java.util.List;
import java.util.Map;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.fasterxml.jackson.databind.JsonNode;
import com.solstice.gateway.client.ClientIa;
import com.solstice.gateway.domaine.Depots;

/**
 * Deux routes de support du front : la liste des clients fictifs pour le
 * sélecteur « vous êtes », et la santé en cascade.
 */
@RestController
@RequestMapping("/api")
@CrossOrigin(origins = "http://localhost:4200")
public class ClientsControleur {

    private final Depots.Clients clients;
    private final ClientIa ia;

    public ClientsControleur(Depots.Clients clients, ClientIa ia) {
        this.clients = clients;
        this.ia = ia;
    }

    /** Le sélecteur du front. Jamais de données sensibles : des fictifs. */
    @GetMapping("/clients")
    public List<Map<String, String>> clients() {
        return clients.findAll().stream()
                .map(c -> Map.of("id", c.getId(),
                                 "nom", c.getNom(),
                                 "formule", c.getFormule()))
                .toList();
    }

    /**
     * Santé en cascade : celle de la passerelle est triviale (si elle
     * répond, elle va bien), l'information utile est l'état du service
     * IA vu depuis la passerelle, avec sa configuration servie. 200 si
     * l'amont répond, 503 sinon : un orchestrateur n'a pas à lire le
     * corps pour décider.
     */
    @GetMapping("/health")
    public ResponseEntity<Map<String, Object>> health() {
        JsonNode amont = ia.health();
        boolean disponible = amont != null;
        Map<String, Object> corps = Map.of(
                "passerelle", "ok",
                "service_ia", disponible ? "ok" : "injoignable",
                "detail_service_ia", disponible ? amont : "aucune reponse");
        return ResponseEntity
                .status(disponible ? HttpStatus.OK
                                   : HttpStatus.SERVICE_UNAVAILABLE)
                .body(corps);
    }
}