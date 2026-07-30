/**
 * Le contrat HTTP de la passerelle, décrit une seule fois.
 *
 * Deux décisions de conception.
 *
 * 1. Les noms de champs sont ceux du réseau, en snake_case, sans couche
 *    de traduction vers du camelCase. Une telle couche donnerait
 *    l'illusion d'un typage sûr alors qu'elle ne fait que déplacer
 *    l'endroit où un champ manquant devient `undefined`. Ici, la forme
 *    attendue du JSON est écrite à un seul endroit, et c'est ce fichier.
 *
 * 2. `etat` n'apparaît NULLE PART. C'est le coeur de l'architecture : le
 *    service Python est sans état, la passerelle Spring garde l'état du
 *    dialogue en base et ne le rend jamais au navigateur. Le front ne
 *    transporte qu'un identifiant de session. Si un `etat` apparaissait
 *    ici un jour, ce serait le signe que la séparation a été rompue.
 */

/** Les quatre chemins que le routeur peut emprunter (service/router.py).
 *  Union fermée volontairement : si le routeur en ajoutait un, le
 *  panneau d'inspection cesserait de compiler, ce qui est exactement le
 *  rappel que l'on veut. */
export type Chemin = 'flux_metier' | 'rag' | 'reformulation' | 'recadrage';

/** Une réponse attendue à une question fermée, rendue en bouton.
 *  `valeur` est le message à renvoyer tel quel : le bouton produit ce
 *  qu'un utilisateur aurait tapé, le front n'a aucune route privilégiée. */
export interface Option {
  rang: number;
  cle: string;
  libelle: string;
  valeur: string;
}

/** Un passage du corpus remonté par la recherche vectorielle. */
export interface Source {
  document: string;
  section: string;
  score: number;
  extrait: string;
}

/** Décomposition de la latence du service IA, en millisecondes.
 *  Toujours les trois clés, à zéro quand l'étape n'a pas eu lieu. */
export interface Latences {
  classification: number;
  recherche: number;
  generation: number;
}

/** Un des trois clients fictifs semés par la passerelle. */
export interface Client {
  id: string;
  nom: string;
  formule: string;
}

/** Corps de POST /api/chat.
 *  Ni état ni formule : la session vit en base côté passerelle, la
 *  formule est déduite du client. Les deux champs sont absents au
 *  premier message et en mode visiteur. */
export interface RequeteChat {
  message: string;
  session_id?: string;
  client_id?: string;
}

/** Réponse de POST /api/chat : celle du service IA, moins `etat`,
 *  plus la session, le client et la latence de bout en bout. */
export interface ReponseChat {
  reponse: string;
  /** null quand aucune classification n'a eu lieu (pré-filtre de
   *  courtoisie) ; renseigné même sous le seuil de rejet, car le
   *  panneau d'inspection doit pouvoir montrer le rejet. */
  intention: string | null;
  confiance: number | null;
  chemin: Chemin;
  formule: string | null;
  sources: Source[];
  /** Jamais null : liste vide quand la question est ouverte. */
  options: Option[];
  latence_ms: Latences;
  session_id: string;
  /** Absent du JSON en mode visiteur : la passerelle omet le champ au
   *  lieu de le mettre à null. Une seule forme d'absence à tester. */
  client?: Client;
  latence_totale_ms: number;
}

/** Configuration réellement servie par le service IA. C'est elle qui doit
 *  correspondre à celle mesurée dans docs/metrics.md. */
export interface ConfigurationServie {
  decoupage: string;
  avec_fil: boolean;
  seuil_confiance: number;
  cosinus_plancher: number | null;
  chunks_servis: number;
  formules: string[];
}

export interface DetailServiceIa {
  statut: string;
  base: string;
  chunks_en_base: number | null;
  chunks_en_memoire: number;
  intentions: number;
  flux: number;
  device: string;
  configuration: ConfigurationServie;
}

/** Réponse de GET /api/health, santé en cascade.
 *  Le détail est optionnel : quand le service IA est injoignable, la
 *  passerelle répond quand même, avec `service_ia` en dégradé. */
export interface Sante {
  passerelle: string;
  service_ia: string;
  detail_service_ia?: DetailServiceIa;
}