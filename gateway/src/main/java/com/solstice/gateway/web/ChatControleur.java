package com.solstice.gateway.web;

import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.solstice.gateway.client.ClientIa;
import com.solstice.gateway.domaine.Client;
import com.solstice.gateway.domaine.Depots;
import com.solstice.gateway.domaine.Echange;
import com.solstice.gateway.domaine.Session;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

/**
 * Le point d'entrée du front. Contrat d'entrée :
 * {@code {message, session_id?, client_id?}}.
 *
 * <p>Ce que la passerelle ajoute par rapport à un appel direct au
 * service IA : le front n'envoie plus NI l'état du dialogue NI la
 * formule. La session vit en base côté passerelle, le contexte client
 * est déduit de client_id, et chaque message laisse une ligne de
 * journal.</p>
 *
 * <p>La réponse rendue au front est celle du service IA, moins le champ
 * {@code etat} (retenu en session) et plus {@code session_id} et
 * {@code client}.</p>
 */
@RestController
@RequestMapping("/api")
@CrossOrigin(origins = "http://localhost:4200")
public class ChatControleur {

    private static final Logger journal =
            LoggerFactory.getLogger(ChatControleur.class);

    private final ClientIa ia;
    private final Depots.Clients clients;
    private final Depots.Sessions sessions;
    private final Depots.Echanges echanges;

    public ChatControleur(ClientIa ia, Depots.Clients clients,
                          Depots.Sessions sessions, Depots.Echanges echanges) {
        this.ia = ia;
        this.clients = clients;
        this.sessions = sessions;
        this.echanges = echanges;
    }

    /**
     * Corps d'entrée. session_id absent = nouvelle conversation ;
     * client_id absent = mode visiteur, sans contexte de formule.
     */
    public record Requete(
            @NotBlank @Size(max = 1000) String message,
            String session_id,
            String client_id) {
    }

    @PostMapping("/chat")
    @Transactional
    public JsonNode chat(@Valid @RequestBody Requete requete) {
        long debut = System.currentTimeMillis();

        Session session = retrouverOuCreer(requete);
        Client client = session.getClient();
        String formule = client == null ? null : client.getFormule();

        Echange echange = new Echange(session.getId(), requete.message());

        JsonNode reponse = ia.chat(requete.message(),
                session.getEtatJson(), formule);

        // L'état rendu par le service est retenu en session, jamais
        // renvoyé au front : c'est le coeur de l'étape 11.
        JsonNode etat = reponse.path("etat");
        session.enregistrerEtat(etat.isNull() || etat.isMissingNode()
                ? null : etat.toString());
        sessions.save(session);

        int totale = (int) (System.currentTimeMillis() - debut);
        journaliser(echange, reponse, totale);

        ObjectNode sortie = ((ObjectNode) reponse).deepCopy();
        sortie.remove("etat");
        sortie.put("session_id", session.getId());
        if (client != null) {
            sortie.putObject("client")
                    .put("id", client.getId())
                    .put("nom", client.getNom())
                    .put("formule", client.getFormule());
        }
        sortie.put("latence_totale_ms", totale);
        return sortie;
    }

    private Session retrouverOuCreer(Requete requete) {
        if (requete.session_id() != null && !requete.session_id().isBlank()) {
            Session existante = sessions.findById(requete.session_id())
                    .orElseThrow(() -> new ResponseStatusException(
                            HttpStatus.NOT_FOUND,
                            "session inconnue : " + requete.session_id()
                            + " (la base est en memoire, un redemarrage"
                            + " de la passerelle efface les sessions)"));
            // Le client est fixé à la création de la session : changer
            // d'identité en cours de conversation serait un état
            // incohérent, le front doit ouvrir une nouvelle session.
            return existante;
        }
        Client client = null;
        if (requete.client_id() != null && !requete.client_id().isBlank()) {
            client = clients.findById(requete.client_id())
                    .orElseThrow(() -> new ResponseStatusException(
                            HttpStatus.NOT_FOUND,
                            "client inconnu : " + requete.client_id()));
        }
        return sessions.save(new Session(client));
    }

    private void journaliser(Echange echange, JsonNode reponse, int totale) {
        JsonNode latences = reponse.path("latence_ms");
        String chemin = texteOuNul(reponse, "chemin");
        String intention = texteOuNul(reponse, "intention");
        Double confiance = reponse.path("confiance").isNumber()
                ? reponse.path("confiance").asDouble() : null;
        String formule = texteOuNul(reponse, "formule");
        echange.renseigner(chemin, intention, confiance, formule,
                latences.path("classification").asInt(0),
                latences.path("recherche").asInt(0),
                latences.path("generation").asInt(0),
                totale);
        echanges.save(echange);
        journal.info("session={} chemin={} intention={} confiance={} "
                        + "formule={} totale={}ms",
                echange.getSessionId(), chemin, intention, confiance,
                formule, totale);
    }

    private static String texteOuNul(JsonNode noeud, String champ) {
        JsonNode valeur = noeud.path(champ);
        return valeur.isTextual() ? valeur.asText() : null;
    }
}