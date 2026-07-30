import { Component, ElementRef, effect, inject, signal, viewChild } from '@angular/core';

import { Chemin, Option } from './contrat';
import { Conversation } from './conversation';

/** Annotation de marge : ce que le routeur a décidé, en un mot.
 *  Terse par choix, c'est de la marginalia ; le libellé complet est en
 *  infobulle et le détail dans le panneau d'inspection. */
const CODES: Record<Chemin, { code: string; titre: string }> = {
  flux_metier: { code: 'flux', titre: 'Démarche guidée, dialogue à étapes' },
  rag: { code: 'docs', titre: 'Réponse rédigée à partir de la documentation' },
  reformulation: { code: 'seuil', titre: 'Intention non reconnue avec assez de confiance' },
  recadrage: { code: 'cadre', titre: 'Demande hors du périmètre de la mutuelle' },
};

/**
 * Trois questions, trois comportements distincts du système. Ce ne sont
 * pas des exemples au hasard : chacune a été classée en amont et sa
 * confiance vérifiée, pour qu'aucune ne tombe sous le seuil de rejet.
 *
 * 1. garantie chiffrée, confiance 0,922 : part sur la recherche
 *    documentaire et cite ses passages. Le nom de la formule est dans la
 *    question à dessein : sans lui la même question tombe à 0,502 et
 *    n'est pas reconnue, le contexte client ne renseignant que la
 *    recherche et jamais le classifieur.
 * 2. résiliation : ouvre un dialogue à étapes, avec confirmation.
 * 3. carence en audiologie, confiance 0,913 : intention reconnue, mais
 *    aucune section du corpus n'y répond. La génération refuse. C'est
 *    la démonstration qu'un refus peut venir de la documentation et non
 *    d'un défaut de compréhension.
 */
const EXEMPLES = [
  'Que rembourse la formule Confort pour des verres progressifs ?',
  'Je veux résilier mon contrat',
  'Quel est le délai de carence en audiologie ?',
];

@Component({
  selector: 'app-chat',
  templateUrl: './chat.html',
  styleUrl: './chat.css',
})
export class Chat {
  protected readonly conversation = inject(Conversation);
  protected readonly saisie = signal('');
  protected readonly exemples = EXEMPLES;

  private readonly fil = viewChild<ElementRef<HTMLElement>>('fil');

  constructor() {
    effect(() => {
      // Les deux lectures sont la dépendance de l'effet : un tour ajouté
      // ou l'indicateur d'attente qui s'allume doivent tous deux ramener
      // le fil en bas.
      this.conversation.tours();
      this.conversation.enCours();
      const element = this.fil()?.nativeElement;
      if (!element) { return; }
      // Reporté d'un tour de boucle : au moment où l'effet s'exécute, le
      // gabarit n'a pas encore été rendu, la hauteur serait celle d'avant.
      setTimeout(() => element.scrollTo({ top: element.scrollHeight }));
    });
  }

  protected code(chemin: Chemin): string {
    return CODES[chemin].code;
  }

  protected titre(chemin: Chemin): string {
    return CODES[chemin].titre;
  }

  /**
   * Vrai quand chaque option renvoie simplement son propre numéro, ce
   * qui est le cas des champs à choix : la liste des libellés est alors
   * DÉJÀ dans le texte de la réponse, et la répéter en boutons ferait
   * doublon. On rend un pavé de touches numérotées. Les questions
   * fermées oui/non, elles, portent leur libellé et rien dans le texte.
   */
  protected numerote(options: Option[]): boolean {
    return options.every(option => option.valeur === String(option.rang));
  }

  protected envoyer(): void {
    const message = this.saisie();
    this.saisie.set('');
    void this.conversation.envoyer(message);
  }

  protected repondre(valeur: string): void {
    void this.conversation.envoyer(valeur);
  }
}