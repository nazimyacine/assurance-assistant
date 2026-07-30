import { Component, computed, inject, linkedSignal } from '@angular/core';

import { Chemin, ReponseChat } from './contrat';
import { Conversation } from './conversation';
import { Passerelle } from './passerelle';

/** Ce que chaque chemin signifie, en une phrase. Le panneau explique la
 *  décision, il ne se contente pas de la nommer. */
const CHEMINS: Record<Chemin, string> = {
  flux_metier: 'Intention transactionnelle : dialogue guidé, sans recherche documentaire.',
  rag: 'Intention informationnelle : recherche dans le corpus, puis rédaction citée.',
  reformulation: 'Confiance sous le seuil : reformulation demandée plutôt que routage deviné.',
  recadrage: 'Demande hors du périmètre de la mutuelle.',
};

const ETAPES = [
  { cle: 'classification', libelle: 'Classification' },
  { cle: 'recherche', libelle: 'Recherche' },
  { cle: 'generation', libelle: 'Génération' },
] as const;

@Component({
  selector: 'app-inspection',
  templateUrl: './inspection.html',
  styleUrl: './inspection.css',
})
export class Inspection {
  private readonly passerelle = inject(Passerelle);
  protected readonly conversation = inject(Conversation);

  /** Lecture seule du signal partagé. Ce panneau NE DEMANDE PLUS la
   *  santé : la coquille en est propriétaire, elle interroge à
   *  l'ouverture et après chaque erreur. Le panneau étant détruit et
   *  reconstruit à chaque passage par l'onglet des métriques, un appel
   *  dans son constructeur produisait une requête réseau par aller-retour,
   *  pour une donnée qui vit dans un service racine et lui survit. */
  protected readonly sante = this.passerelle.etat;

  /** Seuil réellement servi, lu depuis /api/health. Jamais codé en dur :
   *  une valeur figée dans le front pourrait diverger de celle qui
   *  décide, et le repère dessiné sur la barre mentirait. */
  protected readonly seuil = computed(
    () => this.sante()?.detail_service_ia?.configuration.seuil_confiance ?? null);

  protected readonly reponse = this.conversation.derniere;

  /**
   * Les passages dépliés, par index.
   *
   * `linkedSignal` et non `signal` : l'état se réinitialise quand sa
   * source change, ici à chaque nouvelle réponse. Sans cela, un passage
   * déplié le restait pour la réponse SUIVANTE, dont le passage de même
   * rang n'a rien à voir. Le composant, lui, n'est pas reconstruit entre
   * deux messages ; il ne l'est qu'au retour de l'onglet des métriques.
   * Déclaré APRÈS `reponse`, dont il dépend : les champs s'initialisent
   * dans l'ordre d'écriture.
   */
  protected readonly deplies = linkedSignal<ReponseChat | null, Set<number>>({
    source: this.reponse,
    computation: () => new Set<number>(),
  });

  /** Latences en parts du total, pour la barre empilée. La génération
   *  pèse environ 88% du chemin documentaire : c'est le fait le plus
   *  utile du panneau, il doit se voir sans lire un chiffre. */
  protected readonly segments = computed(() => {
    const latences = this.reponse()?.latence_ms;
    if (!latences) { return []; }
    const total = ETAPES.reduce((somme, e) => somme + latences[e.cle], 0);
    if (total === 0) { return []; }
    return ETAPES
      .filter(etape => latences[etape.cle] > 0)
      .map(etape => ({
        cle: etape.cle,
        libelle: etape.libelle,
        ms: latences[etape.cle],
        part: (latences[etape.cle] / total) * 100,
      }));
  });

  protected explication(chemin: Chemin): string {
    return CHEMINS[chemin];
  }

  /** Vrai quand la confiance passe le seuil servi. Nul si l'un des deux
   *  manque : on ne colore pas une comparaison qu'on ne peut pas faire. */
  protected retenue(): boolean | null {
    const confiance = this.reponse()?.confiance;
    const seuil = this.seuil();
    return confiance == null || seuil == null ? null : confiance >= seuil;
  }

  protected basculer(index: number): void {
    this.deplies.update(courant => {
      const suivant = new Set(courant);
      suivant.has(index) ? suivant.delete(index) : suivant.add(index);
      return suivant;
    });
  }

  protected pourcent(valeur: number): string {
    return `${(valeur * 100).toFixed(1)} %`;
  }
}