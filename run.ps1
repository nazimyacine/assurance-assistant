<#
.SYNOPSIS
    Rejoue la chaîne du projet, de la génération des données au service.

.DESCRIPTION
    Un point d'entrée unique plutôt qu'une liste de commandes dans un
    README, pour trois raisons : les commandes se périment moins vite
    quand elles sont exécutées, les arguments non triviaux (la couverture
    minimale du seuil, la pondération des classes) sont écrits une fois
    pour toutes, et la reproduction ne dépend plus de la mémoire de
    celui qui relit.

.EXAMPLE
    .\run.ps1 eval
    .\run.ps1 eval -AvecGeneration
    .\run.ps1 up
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('aide', 'data', 'train', 'index', 'eval', 'up', 'all')]
    [string]$Cible = 'aide',

    # L'évaluation du RAG appelle un LLM distant, contrairement à tout le
    # reste de la chaîne. Elle ne part jamais par accident.
    [switch]$AvecGeneration,

    # Le script threshold_intent.py a 0.90 par défaut ; le seuil publié
    # (0,80, couverture 82,7%) vient de cette valeur-ci. La laisser
    # implicite ferait diverger la reproduction du chiffre publié.
    [double]$CouvertureMin = 0.80
)

$ErrorActionPreference = 'Stop'
$racine = $PSScriptRoot
Set-Location $racine

# --- garde-fous ------------------------------------------------------

<#
    Piège payé à l'étape 10 : fastapi et uvicorn étaient installés hors du
    venv, et la commande `uvicorn` résolvait vers le Python global alors
    que l'invite affichait bien (.venv). D'où deux règles tenues partout
    ici : `python -m` plutôt qu'un exécutable nu, et vérification que le
    python résolu est bien celui du venv avant la première commande.
#>
function Confirmer-Venv {
    if (-not $env:VIRTUAL_ENV) {
        throw "Le venv n'est pas actif. Lancer d'abord .\.venv\Scripts\Activate.ps1"
    }
    $python = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $python -or -not $python.StartsWith($env:VIRTUAL_ENV)) {
        throw "python resout vers '$python', hors du venv '$env:VIRTUAL_ENV'."
    }
}

function Confirmer-Docker {
    docker info *> $null
    if ($LASTEXITCODE -ne 0) {
        throw 'Docker ne repond pas. Demarrer Docker Desktop, puis relancer.'
    }
    docker compose up -d | Out-Host
}

function Etape([string]$titre) {
    Write-Host ''
    Write-Host "=== $titre" -ForegroundColor Cyan
}

# Toute commande qui echoue arrete la chaine : une etape rejouee sur les
# artefacts d'une etape ratee produirait des chiffres faux, pas une erreur.
function Executer([string[]]$commande) {
    Write-Host "> $($commande -join ' ')" -ForegroundColor DarkGray
    & $commande[0] @($commande[1..($commande.Length - 1)])
    if ($LASTEXITCODE -ne 0) {
        throw "Echec : $($commande -join ' ')"
    }
}

function Ouvrir-Fenetre([string]$titre, [string]$commande) {
    Start-Process -FilePath 'powershell.exe' -ArgumentList @(
        '-NoExit', '-Command',
        "`$host.UI.RawUI.WindowTitle = '$titre'; $commande"
    )
}

# --- cibles ----------------------------------------------------------

function Cible-Data {
    Confirmer-Venv
    Etape 'Donnees : gabarits d''intentions et controle du corpus'
    Executer @('python', 'data\generator\generate_intents.py')
    Executer @('python', 'data\generator\check_corpus.py')
}

function Cible-Train {
    Confirmer-Venv
    Etape 'Entrainement CamemBERT (environ 141 s sur RTX 2060 Super)'
    # --class-weights : la pondération des classes fait partie de la
    # configuration qui a produit les chiffres publiés, pas d'une
    # exploration. Elle est donc écrite ici et non laissée au hasard.
    Executer @('python', 'ml\train_intent.py', '--class-weights')
}

function Cible-Index {
    Confirmer-Venv
    Confirmer-Docker
    Etape 'Indexation du corpus (3 variantes de decoupage, 210 chunks)'
    Executer @('python', 'ml\index_corpus.py')
}

function Cible-Eval {
    Confirmer-Venv
    Etape 'Baseline TF-IDF'
    Executer @('python', 'ml\baseline.py')

    Etape 'Evaluation de la classification'
    Executer @('python', 'ml\evaluate_intent.py')

    Etape 'Matrices de confusion'
    Executer @('python', 'ml\plot_confusion.py')

    Etape 'Seuil de rejet'
    Executer @('python', 'ml\threshold_intent.py',
               '--couverture-min', "$CouvertureMin")

    if ($AvecGeneration) {
        Confirmer-Docker
        Etape 'Evaluation du RAG avec generation'
        <#
            COUT : 5 configurations x 50 questions = 250 reponses generees,
            plus un appel de juge par reponse, soit environ 500 appels a
            l'API Mistral. Palier gratuit : environ 1 requete par seconde,
            comptez une dizaine de minutes et des echecs transitoires
            possibles. Sans ce commutateur, docs/eval_rag.json est
            conserve tel quel et la section RAG de metrics.md reste celle
            de la derniere campagne.
        #>
        Executer @('python', 'ml\evaluate_rag.py', '--generation')
    }
    else {
        Write-Host ''
        Write-Host 'RAG non reevalue (environ 500 appels Mistral).' -ForegroundColor Yellow
        Write-Host 'Relancer avec -AvecGeneration pour refaire la campagne.' -ForegroundColor Yellow
    }

    Etape 'Regeneration de docs/metrics.md'
    Executer @('python', 'ml\build_metrics.py')
    Executer @('python', 'ml\build_metrics.py', '--check')
}

function Cible-Up {
    Confirmer-Venv
    Confirmer-Docker
    Etape 'Demarrage des trois services'

    # HF_HUB_OFFLINE : chargement du routeur en 4,1 s au lieu de 7,8 s.
    # Sans elle, sentence-transformers fait une vingtaine d'allers-retours
    # de verification de cache vers le Hub, tres couteux sur une connexion
    # lente. A RETIRER sur une machine neuve, ou les modeles ne sont pas
    # encore en cache : le chargement echouerait.
    Ouvrir-Fenetre 'service IA' (
        "Set-Location '$racine'; " +
        ".\.venv\Scripts\Activate.ps1; " +
        "`$env:HF_HUB_OFFLINE = '1'; " +
        "python -m uvicorn service.api:app --port 8000")

    Ouvrir-Fenetre 'passerelle' (
        "Set-Location '$racine\gateway'; mvn spring-boot:run")

    # npm start et non ng serve : le script prestart copie docs/metrics.md
    # et les figures dans front/public/ avant de servir.
    Ouvrir-Fenetre 'front' (
        "Set-Location '$racine\front'; npm start")

    Write-Host ''
    Write-Host 'Trois fenetres ouvertes. Comptez une trentaine de secondes.'
    Write-Host '  service IA  http://localhost:8000/docs'
    Write-Host '  passerelle  http://localhost:8080/api/health'
    Write-Host '  front       http://localhost:4200'
}

function Cible-Aide {
    Write-Host @'
Usage : .\run.ps1 <cible> [-AvecGeneration] [-CouvertureMin 0.80]

  data    gabarits d'intentions (5000/800/536) et controle du corpus
  train   fine-tuning CamemBERT, environ 141 s sur RTX 2060 Super
  index   decoupage et indexation du corpus dans Postgres (Docker requis)
  eval    baseline, evaluation, matrices, seuil, docs/metrics.md
          -AvecGeneration ajoute la campagne RAG : environ 500 appels Mistral
  up      ouvre les trois services dans trois fenetres
  all     data, train, index, eval

Prerequis : venv actif (.\.venv\Scripts\Activate.ps1) et, pour index,
eval -AvecGeneration et up, Docker Desktop demarre.
'@
}

# --- aiguillage ------------------------------------------------------

switch ($Cible) {
    'data'  { Cible-Data }
    'train' { Cible-Train }
    'index' { Cible-Index }
    'eval'  { Cible-Eval }
    'up'    { Cible-Up }
    'all'   { Cible-Data; Cible-Train; Cible-Index; Cible-Eval }
    default { Cible-Aide }
}