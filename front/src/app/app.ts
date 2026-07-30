import { Component, inject } from '@angular/core';
import { RouterOutlet } from '@angular/router';

import { Conversation } from './conversation';
import { Inspection } from './inspection';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, Inspection],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  protected readonly conversation = inject(Conversation);
}