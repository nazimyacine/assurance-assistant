import { Pipe, PipeTransform } from '@angular/core';

/**
 * Convertit le seul balisage Markdown que le générateur produise
 * réellement, le gras entre doubles astérisques.
 *
 * Trois décisions.
 *
 * 1. La correction est ici et non dans les consignes de génération.
 *    Ces consignes ont produit les chiffres publiés de l'étape 8 ; les
 *    retoucher pour un problème d'affichage invaliderait des mesures.
 *
 * 2. Aucune bibliothèque Markdown. Elle apporterait les tableaux, les
 *    liens, les images et leur surface d'attaque, pour deux astérisques.
 *
 * 3. Le texte est ÉCHAPPÉ AVANT toute conversion, et la chaîne est
 *    rendue telle quelle à [innerHTML], donc encore passée au filtre
 *    d'Angular. On ne désarme jamais ce filtre avec bypassSecurityTrust :
 *    la balise <strong> fait partie de ce qu'il autorise.
 *
 * Le tube est pur : Angular ne le recalcule que si le texte change.
 */

const ECHAPPEMENTS: Record<string, string> = {
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#39;',
};

@Pipe({ name: 'gras' })
export class Gras implements PipeTransform {
  transform(texte: string): string {
    const echappe = texte.replace(/[&<>"']/g, caractere => ECHAPPEMENTS[caractere]);
    // Non gourmand, et sur une seule ligne : un astérisque isolé en
    // début de phrase ne doit pas mettre en gras la moitié du message.
    return echappe.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
  }
}