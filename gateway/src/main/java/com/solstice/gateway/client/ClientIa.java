package com.solstice.gateway.client;

import java.time.Duration;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientRequestException;
import org.springframework.web.reactive.function.client.WebClientResponseException;
import org.springframework.web.server.ResponseStatusException;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import io.netty.channel.ChannelOption;
import reactor.netty.http.client.HttpClient;

/**
 * Appelle le service IA FastAPI. C'est ici, et nulle part ailleurs, que
 * vivent les délais d'attente et la traduction des pannes.
 *
 * <p>Contrat d'erreur, aligné sur le cadrage :</p>
 * <ul>
 *   <li>service injoignable ou trop lent : 503, le front affiche un
 *       mode dégradé</li>
 *   <li>400 du service (formule inconnue) : relayé en 400, c'est une
 *       faute du client</li>
 *   <li>toute autre erreur du service : 502, la passerelle est saine
 *       mais l'amont a un problème</li>
 * </ul>
 *
 * <p>La réponse est manipulée en {@link JsonNode}, jamais projetée sur
 * des classes Java : même principe que l'état opaque de {@code Session},
 * le contrat appartient au service IA et la passerelle ne filtre pas ce
 * qu'elle ne connaît pas.</p>
 */
@Component
public class ClientIa {

    private static final Logger journal = LoggerFactory.getLogger(ClientIa.class);

    private final WebClient web;
    private final ObjectMapper json;

    public ClientIa(ProprietesIa proprietes, ObjectMapper json) {
        this.json = json;
        HttpClient http = HttpClient.create()
                .option(ChannelOption.CONNECT_TIMEOUT_MILLIS,
                        proprietes.connectTimeoutMs())
                .responseTimeout(Duration.ofMillis(proprietes.readTimeoutMs()));
        this.web = WebClient.builder()
                .baseUrl(proprietes.baseUrl())
                .clientConnector(
                        new org.springframework.http.client.reactive
                                .ReactorClientHttpConnector(http))
                .build();
    }

    /**
     * POST /chat du service IA.
     *
     * @param message  le message de l'utilisateur
     * @param etatJson l'état du dialogue conservé en session, ou null
     * @param formule  la formule du client, ou null en mode visiteur
     * @return le corps de la réponse du service, tel quel
     */
    public JsonNode chat(String message, String etatJson, String formule) {
        ObjectNode corps = json.createObjectNode();
        corps.put("message", message);
        corps.put("formule", formule);
        if (etatJson != null && !etatJson.isBlank()) {
            try {
                corps.set("etat", json.readTree(etatJson));
            } catch (Exception e) {
                // Un état illisible en base ne doit pas bloquer la
                // conversation : on repart d'un dialogue neuf.
                journal.warn("etat de session illisible, ignore : {}",
                        e.getMessage());
            }
        }
        return echanger(corps);
    }

    /** GET /health du service IA, pour la santé en cascade. */
    public JsonNode health() {
        try {
            return web.get().uri("/health").retrieve()
                    .bodyToMono(JsonNode.class).block();
        } catch (Exception e) {
            return null;
        }
    }

    private JsonNode echanger(ObjectNode corps) {
        try {
            return web.post().uri("/chat")
                    .bodyValue(corps)
                    .retrieve()
                    .bodyToMono(JsonNode.class)
                    .block();
        } catch (WebClientResponseException e) {
            if (e.getStatusCode().value() == 400) {
                throw new ResponseStatusException(HttpStatus.BAD_REQUEST,
                        e.getResponseBodyAsString(), e);
            }
            journal.error("service IA en erreur {} : {}",
                    e.getStatusCode().value(), e.getResponseBodyAsString());
            throw new ResponseStatusException(HttpStatus.BAD_GATEWAY,
                    "le service IA a repondu en erreur", e);
        } catch (WebClientRequestException e) {
            journal.error("service IA injoignable : {}", e.getMessage());
            throw new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE,
                    "le service IA est indisponible", e);
        } catch (Exception e) {
            // Inclut le dépassement du responseTimeout (TimeoutException
            // enveloppée par Reactor) : trop lent = indisponible.
            journal.error("appel au service IA en echec : {}", e.toString());
            throw new ResponseStatusException(HttpStatus.SERVICE_UNAVAILABLE,
                    "le service IA n'a pas repondu dans les delais", e);
        }
    }
}