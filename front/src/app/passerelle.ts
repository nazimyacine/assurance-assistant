import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Injectable, inject, signal } from '@angular/core';
import { Observable, firstValueFrom, timeout } from 'rxjs';

import { Client, ReponseChat, RequeteChat, Sante } from './contrat';

/**
 * Le seul point de contact du front avec le monde extérieur.
 *
 * Tout passe par la passerelle Spring, jamais directement par le service
 * Python : c'est elle qui tient la session, injecte la formule du client
 * et journalise. Le préfixe /api est réécrit vers le port 8080 par le
 * proxy du serveur de développement, donc aucune URL absolue n'apparaît
 * ici et rien n'est à changer si la passerelle déménage.
 */

/** Au-delà des 15 secondes d'attente de lecture de la passerelle, plus
 *  une marge pour ses réessais. Dernier maillon d'une chaîne de délais
 *  croissants : 10 s pour l'appel au modèle, 15 s pour la passerelle,
 *  20 s ici. */
const DELAI_MS = 20_000;

/** Erreur déjà traduite en message affichable. Le composant décide où
 *  l'afficher, il n'a plus à interpréter un code de statut. */
export class ErreurPasserelle extends Error {
  constructor(readonly statut: number, message: string) {
    super(message);
    this.name = 'ErreurPasserelle';
  }
}

@Injectable({ providedIn: 'root' })
export class Passerelle {
  private readonly http = inject(HttpClient);

  /** Renseignée par la première réponse, puis renvoyée à chaque message :
   *  c'est elle qui permet à la passerelle de retrouver l'état du
   *  dialogue en cours. Le front ne voit jamais l'état lui-même. */
  readonly sessionId = signal<string | null>(null);

  /** null = mode visiteur : aucun contexte client, la recherche porte
   *  sur tout le corpus. */
  readonly clientId = signal<string | null>(null);

  /** Dernière santé connue des deux services. Partagée plutôt que
   *  redemandée par chaque vue : le panneau d'inspection y lit le seuil
   *  de rejet servi, la pastille d'en-tête y lira l'état. */
  readonly etat = signal<Sante | null>(null);

  choisirClient(id: string | null): void {
    this.clientId.set(id);
    this.sessionId.set(null);
  }

  reinitialiser(): void {
    this.sessionId.set(null);
  }

  async envoyer(message: string): Promise<ReponseChat> {
    const corps: RequeteChat = { message };
    const session = this.sessionId();
    const client = this.clientId();
    if (session) { corps.session_id = session; }
    if (client) { corps.client_id = client; }

    const reponse = await this.appeler(
      this.http.post<ReponseChat>('/api/chat', corps));
    this.sessionId.set(reponse.session_id);
    return reponse;
  }

  clients(): Promise<Client[]> {
    return this.appeler(this.http.get<Client[]>('/api/clients'));
  }

  /**
   * Ne lève JAMAIS et met à jour le signal partagé. Un indicateur de
   * santé qui échoue en cascade n'indique plus rien : quand le service
   * IA est injoignable, la passerelle répond en 503 avec un corps
   * parfaitement exploitable, et c'est précisément l'état à montrer.
   */
  async rafraichirSante(): Promise<Sante> {
    let sante: Sante;
    try {
      sante = await firstValueFrom(
        this.http.get<Sante>('/api/health').pipe(timeout(DELAI_MS)));
    } catch (erreur) {
      sante = erreur instanceof HttpErrorResponse && erreur.error?.passerelle
        ? erreur.error as Sante
        : { passerelle: 'injoignable', service_ia: 'inconnu' };
    }
    this.etat.set(sante);
    return sante;
  }

  private async appeler<T>(appel: Observable<T>): Promise<T> {
    try {
      return await firstValueFrom(appel.pipe(timeout(DELAI_MS)));
    } catch (erreur) {
      throw this.traduire(erreur);
    }
  }

  /**
   * Traduit une panne en phrase destinée à un utilisateur. Le contrat
   * d'erreur de la passerelle est respecté à la lettre : 503 mode
   * dégradé, 400 faute du client, 502 panne en amont.
   */
  private traduire(erreur: unknown): ErreurPasserelle {
    if (!(erreur instanceof HttpErrorResponse)) {
      return new ErreurPasserelle(
        0, "Le service n'a pas répondu dans le temps imparti.");
    }
    if (erreur.status === 0) {
      return new ErreurPasserelle(
        0, 'La passerelle est injoignable. Est-elle démarrée sur le port 8080 ?');
    }
    const detail = typeof erreur.error?.detail === 'string'
      ? erreur.error.detail : null;
    switch (erreur.status) {
      case 400:
        return new ErreurPasserelle(400, detail ?? 'Demande invalide.');
      case 404:
        return new ErreurPasserelle(
          404, detail ?? 'Session inconnue. La passerelle a peut-être redémarré : '
             + 'ses sessions vivent en mémoire.');
      case 502:
      case 503:
        return new ErreurPasserelle(
          erreur.status,
          detail ?? "L'assistant est momentanément indisponible.");
      default:
        return new ErreurPasserelle(
          erreur.status, `Erreur inattendue (${erreur.status}).`);
    }
  }
}