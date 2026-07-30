import { Routes } from '@angular/router';

import { Chat } from './chat';

export const routes: Routes = [
  { path: '', component: Chat, title: 'Assistant Mutuelle Solstice' },
  // L'onglet des métriques viendra ici au commit 4.
  { path: '**', redirectTo: '' },
];