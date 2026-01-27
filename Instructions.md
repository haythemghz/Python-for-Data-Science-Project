# Module Python – Data Science 2  
## Projet Machine Learning Guidé (13 semaines)

Je suis responsable d’un Module Python pour Data Science 2, organisé 100% sous forme de projet, sur une durée de 13 semaines.

Les étudiants :

maîtrisent déjà le pipeline classique de Machine Learning

connaissent les algorithmes standards : KNN, Decision Trees, Random Forest, etc.

ont déjà pratiqué le prétraitement de données, l’entraînement et l’évaluation de modèles

Le module vise des notions plus avancées, notamment :

Ensembling

Boosting

Feature Selection

Structuration d’un projet ML complet, reproductible et déployable

Le projet est guidé en classe.
👉 À chaque étape, je fournis un tutoriel détaillé que les étudiants doivent suivre.

🎯 Objectif global

Organiser un dossier de travail complet et produire tous les supports pédagogiques, prêts à être utilisés :

Structure attendue du projet

📁 cours/
→ supports de cours (théorie, concepts, slides ou notes)

📁 code/
→ code Python structuré, GitHub-ready, exécutable et propre

📁 tutos/
→ tutoriels pédagogiques en LaTeX (.tex), prêts à compiler

📄 guide_projet.tex
→ guide global du projet, qui :

décrit les grandes lignes

détaille le déroulement sur 13 semaines

référence tous les tutoriels

précise le planning et les livrables

🧠 Projet Machine Learning – Cadre pédagogique
Objectif pédagogique

Travailler sur un dataset CSV réel, avec :

un objectif ML clairement défini dès le début

une montée progressive en complexité

une approche end-to-end (data → modèle → API → front → déploiement)

📊 Choix du dataset

Deux possibilités :

1️⃣ Dataset téléchargeable directement

Sources possibles :

Kaggle → https://www.kaggle.com/datasets

Papers With Code → https://paperswithcode.com/datasets

Hugging Face Datasets → https://huggingface.co/datasets

Google Dataset Search → https://datasetsearch.research.google.com

UCI ML Repository → https://archive.ics.uci.edu/ml

👉 Pour des raisons pédagogiques, privilégier des datasets classiques et robustes
(ex. Breast Cancer Wisconsin, Adult Dataset, etc.)

2️⃣ Web Scraping (optionnel)

Outils :

BeautifulSoup

Selenium

Si cette option est choisie, elle doit être encadrée pédagogiquement.

🧩 Déroulement du projet (par étapes)

⚠️ Important :
À chaque étape, je travaille sur un seul dataset cohérent,
et je génère le tutoriel LaTeX correspondant.

🔹 Étape 0 – Web Scraping (si applicable)

Définir l’objectif avant le scraping

Expliquer la méthodologie

Exemple concret sur un site réel

📄 Livrables :

tutos/scraping_tuto.tex

code/scraping.py

🔹 Étape 1 – Choix du dataset & Data Exploration

Compréhension du problème

Analyse exploratoire

Visualisation des données

📄 Livrables :

tutos/exploration_tuto.tex

code/data_exploration.py

🔹 Étape 2 – Pipeline Machine Learning & Suivi des expériences

Nettoyage des données

Feature engineering

Préprocessing

Feature selection

Modélisation

Évaluation

👉 Intégrer MLflow pour :

le suivi des expériences

la comparaison des modèles

📄 Livrables :

Tutoriel MLflow en LaTeX

Scripts ML complets et structurés

🔹 Étape 3 – Backend & API (FastAPI)

Installation et configuration

Chargement du modèle entraîné

Création d’endpoints

Bonnes pratiques de structuration

📄 Livrables :

Tutoriel FastAPI (.tex)

Scripts backend fonctionnels

🔹 Étape 4 – Frontend (React)

Interface simple pour consommer l’API

Communication front ↔ backend

Cas d’usage clair (prédiction)

📄 Livrables :

Tutoriel React orienté ML

Exemple de front minimal

🔹 Étape 5 – Déploiement

Dockerisation du projet

Docker / Docker Compose

Déploiement du backend (et front si pertinent)

📄 Livrables :

Tutoriel déploiement (.tex)

Dockerfile

docker-compose.yml

✅ Résultat attendu

À la fin :

Un projet ML complet, structuré et professionnel

Tous les supports pédagogiques prêts :

cours

code

tutoriels LaTeX

guide global

Un projet reproductible, déployable et pédagogique, aligné avec un module Data Science avancé

