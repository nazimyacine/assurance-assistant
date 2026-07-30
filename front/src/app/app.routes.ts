import { Routes } from '@angular/router';

import { Chat } from './chat';
import { Metriques } from './metriques';

export const routes: Routes = [
  { path: '', component: Chat, title: 'Assistant Mutuelle Solstice' },
  // Chargée avec le reste plutôt qu'en différé : le composant fait une
  // centaine de lignes, un chargement à la demande coûterait un
  // aller-retour réseau pour une économie invisible.
  {
    path: 'metriques',
    component: Metriques,
    title: 'Métriques · Mutuelle Solstice',
  },
  { path: '**', redirectTo: '' },
];