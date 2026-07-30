import { ApplicationConfig, provideBrowserGlobalErrorListeners } from '@angular/core';
import { provideHttpClient } from '@angular/common/http';
import { provideRouter } from '@angular/router';
import { routes } from './app.routes';

export const appConfig: ApplicationConfig = {
  providers: [
    provideBrowserGlobalErrorListeners(),
    // Sans ce fournisseur, toute injection de HttpClient échoue au
    // démarrage avec une erreur d'injection peu explicite. Il n'est pas
    // posé par défaut : Angular ne suppose pas qu'une application parle
    // à un serveur.
    provideHttpClient(),
    provideRouter(routes)
  ]
};