import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Component, ViewEncapsulation, inject, signal } from '@angular/core';
import { firstValueFrom, timeout } from 'rxjs';

import { convertirMarkdown } from './markdown';

/**
 * La page des métriques : `docs/metrics.md` rendu à l'écran.
 *
 * L'appel HTTP ne passe PAS par `Passerelle`, et c'est délibéré. Ce
 * service est le point de contact unique avec la PASSERELLE : il tient la
 * session, traduit un contrat d'erreur 400/502/503, porte le client
 * choisi. Rien de tout cela n'a de sens pour un fichier statique déposé à
 * côté de `index.html` par le script `copier:metriques`. Le faire
 * transiter par là diluerait la responsabilité du service au lieu de la
 * renforcer.
 *
 * Le document est chargé à chaque ouverture de la page plutôt que mis en
 * cache : il est régénéré par `ml/build_metrics.py` à chaque campagne
 * d'évaluation, et une page de métriques qui affiche les chiffres d'avant
 * serait pire que pas de page du tout.
 *
 * ENCAPSULATION RETIRÉE, et c'est le seul composant de l'application dans
 * ce cas. Angular encapsule les styles en apposant un attribut de portée
 * sur les éléments du gabarit, puis en l'ajoutant à chaque sélecteur. Le
 * HTML injecté par `[innerHTML]` naît après la compilation du gabarit :
 * il ne porte pas cet attribut, donc AUCUN style encapsulé ne peut le
 * viser. Les deux issues sont `::ng-deep`, déprécié, et celle-ci. La
 * portée est alors rétablie par convention : dans `metriques.css`, tout
 * sélecteur commence par `app-metriques`.
 */
@Component({
  selector: 'app-metriques',
  imports: [],
  templateUrl: './metriques.html',
  styleUrl: './metriques.css',
  encapsulation: ViewEncapsulation.None,
})
export class Metriques {
  private readonly http = inject(HttpClient);

  protected readonly html = signal<string | null>(null);
  protected readonly erreur = signal<string | null>(null);

  constructor() {
    void this.charger();
  }

  private async charger(): Promise<void> {
    try {
      // URL relative, donc résolue contre le <base href> de la page :
      // l'application reste déplaçable sous un sous-chemin. C'est aussi
      // ce qui fait fonctionner le src des figures produites par
      // convertirMarkdown.
      const source = await firstValueFrom(
        this.http.get('metrics.md', { responseType: 'text' })
          .pipe(timeout(10_000)));
      this.html.set(convertirMarkdown(source));
    } catch (erreur) {
      // Le 404 a une cause unique et une réparation connue : la dire,
      // plutôt que d'afficher une panne anonyme.
      this.erreur.set(
        erreur instanceof HttpErrorResponse && erreur.status === 404
          ? 'Le document n\'a pas encore été généré. Lancer '
            + 'python ml\\build_metrics.py à la racine du dépôt, '
            + 'puis relancer npm start.'
          : "Le document des métriques n'a pas pu être chargé.");
    }
  }
}