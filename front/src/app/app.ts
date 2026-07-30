import { Component, inject } from '@angular/core';
import { RouterOutlet } from '@angular/router';

import { Conversation } from './conversation';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  protected readonly conversation = inject(Conversation);
}