/**
 * Conversion de `docs/metrics.md` en HTML.
 *
 * Taillée sur ce que ce fichier contient RÉELLEMENT, et rien d'autre :
 * titres de niveaux 1 à 3, paragraphes, tableaux avec ligne d'en-tête,
 * une image, du gras et du `code` entre accents graves. Le choix d'écrire
 * une trentaine de lignes plutôt que d'ajouter une bibliothèque de rendu
 * markdown tient à trois raisons, dans cet ordre :
 *
 * 1. la surface de code non lue dans un dépôt public, qu'un recruteur
 *    pourrait ouvrir, reste nulle ;
 * 2. une bibliothèque complète accepterait du HTML brut dans la source,
 *    ce que l'on ne veut justement pas ;
 * 3. la source est GÉNÉRÉE par `ml/build_metrics.py`, donc sa syntaxe est
 *    connue et bornée. Ce n'est pas du markdown écrit par un inconnu.
 *
 * Ce qui n'est pas géré (listes, citations, blocs de code, liens) est
 * rendu tel quel, en texte échappé : la page se dégrade, elle ne casse
 * pas. Si `build_metrics.py` se met un jour à produire des listes, elles
 * s'afficheront avec leur tiret, ce qui se voit et se corrige ici.
 *
 * Aucune classe CSS n'est émise, uniquement des balises standard : le
 * filtre de sécurité d'Angular n'est jamais désarmé (pas de
 * `bypassSecurityTrust*`), et la mise en forme se fait par sélecteurs de
 * balises depuis le composant. Un attribut retiré par le filtre ne peut
 * donc pas emporter la mise en page.
 */

const ECHAPPEMENTS: Record<string, string> = {
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#39;',
};

/** Les figures vivent à la racine de `public/`, déposées par le script
 *  `copier:metriques`. Tout ce qui n'est pas un simple nom de fichier PNG
 *  est refusé : ni chemin, ni protocole, ni remontée par `..`. */
const FIGURE_AUTORISEE = /^[\w-]+\.png$/;

function echapper(texte: string): string {
  return texte.replace(/[&<>"']/g, caractere => ECHAPPEMENTS[caractere]);
}

/**
 * Le contenu d'une ligne. L'échappement vient TOUJOURS en premier : à
 * partir de là, plus aucun `<` de la source ne peut devenir une balise,
 * et les règles suivantes ne produisent que les balises qu'elles
 * écrivent elles-mêmes. Même ordre que le tube `gras`, pour la même
 * raison.
 */
function enLigne(texte: string): string {
  return echapper(texte)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
}

/** Une ligne de tableau découpée en cellules : les barres de début et de
 *  fin encadrent, elles ne séparent pas. Une cellule vide est légitime,
 *  `metrics.md` en a plusieurs en tête de colonne. */
function cellules(ligne: string): string[] {
  return ligne.trim()
    .replace(/^\|/, '')
    .replace(/\|$/, '')
    .split('|')
    .map(cellule => cellule.trim());
}

/** La ligne de tirets qui suit l'en-tête, et qui signe un tableau. */
function estSeparateur(ligne: string | undefined): boolean {
  return ligne !== undefined && /^\|[\s:|-]+\|$/.test(ligne.trim());
}

function rendreTableau(lignes: string[]): string {
  const entete = cellules(lignes[0])
    .map(cellule => `<th>${enLigne(cellule)}</th>`)
    .join('');
  const corps = lignes.slice(2)
    .map(ligne => cellules(ligne)
      .map(cellule => `<td>${enLigne(cellule)}</td>`)
      .join(''))
    .map(rangee => `<tr>${rangee}</tr>`)
    .join('');
  // Enveloppe de défilement. Le tableau d'analyse des erreurs a six
  // colonnes dont une de phrases entières : il déborde de la colonne de
  // lecture. Le défilement se pose ICI et jamais sur le <table>, dont le
  // display doit rester `table` : en `block`, thead et tbody calculent
  // leurs largeurs séparément et les colonnes cessent d'être alignées.
  return `<div class="tableau"><table><thead><tr>${entete}</tr></thead>`
    + `<tbody>${corps}</tbody></table></div>`;
}

/** Chemin laissé RELATIF, donc résolu contre le `<base href>` de la
 *  page : la figure suivrait l'application si elle était un jour servie
 *  ailleurs qu'à la racine. */
function rendreImage(texte: string, source: string): string {
  if (!FIGURE_AUTORISEE.test(source)) {
    return `<p><em>Figure non affichée : ${enLigne(source)}</em></p>`;
  }
  return `<figure><img src="${source}" alt="${echapper(texte)}" /></figure>`;
}

export function convertirMarkdown(source: string): string {
  const lignes = source.replace(/\r\n/g, '\n').split('\n');
  const sortie: string[] = [];
  let paragraphe: string[] = [];

  // Un paragraphe se ferme sur une ligne vide ou sur un bloc d'un autre
  // genre. Les lignes repliées de la source sont rejointes par une
  // espace, comme le veut le markdown.
  const viderParagraphe = (): void => {
    if (paragraphe.length > 0) {
      sortie.push(`<p>${enLigne(paragraphe.join(' '))}</p>`);
      paragraphe = [];
    }
  };

  for (let i = 0; i < lignes.length; i++) {
    const nue = lignes[i].trim();

    if (nue === '') {
      viderParagraphe();
      continue;
    }

    const titre = /^(#{1,3})\s+(.*)$/.exec(nue);
    if (titre) {
      viderParagraphe();
      const niveau = titre[1].length;
      sortie.push(`<h${niveau}>${enLigne(titre[2])}</h${niveau}>`);
      continue;
    }

    const image = /^!\[([^\]]*)\]\(([^)]+)\)$/.exec(nue);
    if (image) {
      viderParagraphe();
      sortie.push(rendreImage(image[1], image[2]));
      continue;
    }

    // Un tableau se reconnaît à sa DEUXIÈME ligne : une ligne isolée
    // commençant par une barre reste un paragraphe ordinaire.
    if (nue.startsWith('|') && estSeparateur(lignes[i + 1])) {
      viderParagraphe();
      const debut = i;
      i += 2;
      while (i < lignes.length && lignes[i].trim().startsWith('|')) { i++; }
      sortie.push(rendreTableau(lignes.slice(debut, i)));
      i--;  // la boucle rendra la main sur la ligne qui a arrêté le tableau
      continue;
    }

    paragraphe.push(nue);
  }

  viderParagraphe();
  return sortie.join('\n');
}