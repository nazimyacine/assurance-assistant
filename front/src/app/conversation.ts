import { Injectable, computed, inject, signal } from '@angular/core';

import { Option, ReponseChat } from './contrat';
import { ErreurPasserelle, Passerelle } from './passerelle';

/**
 * L'état de la conversation, hors de tout composant.
 *
 * Le fil de discussion et le panneau d'inspection montrent deux facettes
 * de la même chose : le premier ce que l'assistant a répondu, le second
 * comment il y est arrivé. Loger cet état dans un service plutôt que
 * dans le composant de chat évite de le faire transiter d'un composant à
 * l'autre, et garde les deux vues d'accord par construction.
 */

export interface Tour {
  id: number;
  auteur: 'vous' | 'assistant';
  /** Le texte affiché. Pour l'assistant, c'est reponse.reponse. */
  texte: string;
  /** Absent sur les tours de l'utilisateur. */
  reponse?: ReponseChat;
}

@Injectable({ providedIn: 'root' })
export class Conversation {
  private readonly passerelle = inject(Passerelle);
  private compteur = 0;

  readonly tours = signal<Tour[]>([]);
  readonly enCours = signal(false);
  readonly erreur = signal<string | null>(null);

  /** La réponse que le panneau d'inspection décortique : la dernière de
   *  l'assistant, qui reste affichée pendant que l'on écrit la suite. */
  readonly derniere = computed<ReponseChat | null>(() => {
    const tours = this.tours();
    for (let i = tours.length - 1; i >= 0; i--) {
      const reponse = tours[i].reponse;
      if (reponse) { return reponse; }
    }
    return null;
  });

  /** Les options encore cliquables : celles du tout dernier tour, et
   *  seulement lui. Une question à laquelle on a déjà répondu ne doit
   *  plus offrir ses boutons, l'état du flux a avancé côté passerelle. */
  readonly options = computed<Option[]>(
    () => this.tours().at(-1)?.reponse?.options ?? []);

  async envoyer(message: string): Promise<void> {
    const propre = message.trim();
    if (!propre || this.enCours()) { return; }

    this.erreur.set(null);
    this.ajouter({ auteur: 'vous', texte: propre });
    this.enCours.set(true);
    try {
      const reponse = await this.passerelle.envoyer(propre);
      this.ajouter({ auteur: 'assistant', texte: reponse.reponse, reponse });
    } catch (erreur) {
      // Le message est déjà rédigé pour un lecteur humain par le client
      // de la passerelle. Le tour de l'utilisateur reste affiché : il
      // doit pouvoir voir ce qui n'est pas passé, et le renvoyer.
      this.erreur.set(erreur instanceof ErreurPasserelle
        ? erreur.message
        : "La demande n'a pas abouti.");
    } finally {
      this.enCours.set(false);
    }
  }

  /** Vide le fil et abandonne la session. La passerelle gardera la
   *  précédente en base sans que personne ne la réclame, ce qui est sans
   *  conséquence : ses sessions vivent en mémoire. */
  recommencer(): void {
    this.tours.set([]);
    this.erreur.set(null);
    this.passerelle.reinitialiser();
  }

  private ajouter(tour: Omit<Tour, 'id'>): void {
    this.tours.update(tours => [...tours, { ...tour, id: ++this.compteur }]);
  }
}