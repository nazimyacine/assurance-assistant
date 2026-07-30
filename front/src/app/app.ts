import { TitleCasePipe } from '@angular/common';
import { Component, computed, effect, inject, signal } from '@angular/core';
import { toSignal } from '@angular/core/rxjs-interop';
import {
  NavigationEnd, Router, RouterLink, RouterLinkActive, RouterOutlet,
} from '@angular/router';
import { filter, map } from 'rxjs';

import { Client } from './contrat';
import { Conversation } from './conversation';
import { Inspection } from './inspection';
import { Passerelle } from './passerelle';

/** Les trois états montrés par la pastille, plus l'attente initiale. */
type EtatSante = 'inconnu' | 'ok' | 'degrade' | 'injoignable';

const LIBELLES: Record<EtatSante, string> = {
  inconnu: 'état inconnu',
  ok: 'en ligne',
  degrade: 'service IA dégradé',
  injoignable: 'passerelle injoignable',
};

@Component({
  selector: 'app-root',
  imports: [
    RouterOutlet, RouterLink, RouterLinkActive, Inspection, TitleCasePipe,
  ],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  protected readonly conversation = inject(Conversation);
  protected readonly passerelle = inject(Passerelle);
  private readonly router = inject(Router);

  /** Sans quoi l'onglet Conversation resterait actif sur toutes les
   *  routes, `/` étant le préfixe de tout. Déclaré ici plutôt qu'en
   *  littéral dans le gabarit, qui en reconstruirait un à chaque cycle. */
  protected readonly exact = { exact: true } as const;

  /** L'URL courante, en signal. Le panneau d'inspection est placé HORS du
   *  router-outlet, parce qu'il commente la conversation et non la route :
   *  il doit donc être masqué explicitement quand la route n'est plus
   *  celle de la conversation. */
  private readonly url = toSignal(
    this.router.events.pipe(
      filter(evenement => evenement instanceof NavigationEnd),
      map(() => this.router.url)),
    { initialValue: this.router.url });

  protected readonly surChat = computed(() => !this.url().startsWith('/metriques'));

  /** Les clients fictifs semés par la passerelle. Liste vide si elle est
   *  injoignable : le sélecteur se réduit alors au mode visiteur, qui est
   *  exactement ce que l'on peut encore faire dans cet état. */
  protected readonly clients = signal<Client[]>([]);

  /**
   * L'état de santé exige la forme connue en BONNE SANTÉ, au lieu de
   * tenter de reconnaître les formes malades. La différence n'est pas
   * cosmétique : une forme inattendue devient alors une réserve visible,
   * jamais un feu vert par défaut. Première version de ce critère : elle
   * comparait les deux comptages de passages sans vérifier que c'étaient
   * des nombres, et le service Python coupé donnait `undefined` des deux
   * côtés, donc l'égalité, donc le vert. Un point vert allumé par deux
   * absences est le pire des cas.
   */
  protected readonly sante = computed<EtatSante>(() => {
    const etat = this.passerelle.etat();
    if (!etat) { return 'inconnu'; }

    // La passerelle ne répond pas, ou pas d'elle-même : notre propre
    // repli pose autre chose que 'ok' dans ce champ.
    if (etat.passerelle !== 'ok') { return 'injoignable'; }

    const detail = etat.detail_service_ia;
    if (etat.service_ia !== 'ok' || !detail) { return 'degrade'; }

    // La cascade est descendue jusqu'au bout : reste à vérifier que
    // l'index servi est bien celui qui a été mesuré.
    const enBase = detail.chunks_en_base;
    const enMemoire = detail.chunks_en_memoire;
    if (typeof enBase !== 'number' || typeof enMemoire !== 'number') {
      return 'degrade';
    }
    return enBase === enMemoire && enMemoire > 0 ? 'ok' : 'degrade';
  });

  protected readonly libelleSante = computed(() => LIBELLES[this.sante()]);

  /** Le détail en infobulle : la pastille reste discrète, mais ne cache
   *  rien à qui s'y attarde. */
  protected readonly detailSante = computed(() => {
    const etat = this.passerelle.etat();
    if (!etat) { return 'Santé des services non encore interrogée.'; }

    const lignes = [
      `Passerelle : ${etat.passerelle}`,
      `Service IA : ${etat.service_ia}`,
    ];
    const detail = etat.detail_service_ia;
    if (detail) {
      lignes.push(
        `Passages : ${detail.chunks_en_base ?? 'base injoignable'} en base, `
        + `${detail.chunks_en_memoire} en mémoire`);
    }
    lignes.push('Cliquer pour réinterroger.');
    return lignes.join('\n');
  });

  constructor() {
    void this.passerelle.rafraichirSante();
    void this.chargerClients();

    // Une panne se voit d'abord dans le fil. Plutôt qu'un sondage
    // périodique qui bavarde pour rien, on réinterroge la santé au moment
    // précis où l'utilisateur vient de constater un échec.
    effect(() => {
      if (this.conversation.erreur()) {
        void this.passerelle.rafraichirSante();
      }
    });
  }

  /**
   * Changer d'identité, c'est changer de formule injectée dans la
   * recherche. La passerelle fixe le client d'une session à sa CRÉATION :
   * il n'est pas modifiable en cours de route, il faut une session
   * neuve. Et une session neuve sous un fil ancien serait un mensonge à
   * l'écran, donc le fil part avec.
   */
  protected changerIdentite(valeur: string): void {
    const id = valeur === '' ? null : valeur;
    if (id === this.passerelle.clientId()) { return; }
    this.conversation.recommencer();
    this.passerelle.choisirClient(id);
  }

  protected interrogerSante(): void {
    void this.passerelle.rafraichirSante();
  }

  private async chargerClients(): Promise<void> {
    try {
      this.clients.set(await this.passerelle.clients());
    } catch {
      // Ne jamais empêcher l'application de s'afficher pour un décor :
      // l'échec est déjà dit par la pastille.
      this.clients.set([]);
    }
  }
}