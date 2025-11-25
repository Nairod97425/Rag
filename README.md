# 🩺 Assistant RAG Local (Diabète)

Ce projet est un **Assistant Conversationnel Médical** fonctionnant 100% en local sur votre machine.
Il utilise la technique **RAG (Retrieval-Augmented Generation)** pour répondre aux questions en se basant exclusivement sur des documents fiables (ex: Ameli, Fédération des Diabétiques), garantissant la confidentialité des données.

**Stack Technique :** Python, Ollama, LangChain, ChromaDB, Streamlit, Ragas.

---

## 📋 1. Prérequis

Avant de commencer, assurez-vous d'avoir installé :

1.  **Python 3.10+** : [Télécharger Python](https://www.python.org/downloads/)
2.  **Ollama** : Le moteur d'IA local. [Télécharger Ollama](https://ollama.com)

---

## ⚙️ 2. Installation

### A. Cloner ou préparer le dossier
Ouvrez votre terminal (Command Prompt, PowerShell ou Terminal) dans le dossier du projet.

### B. Créer un environnement virtuel (Recommandé)
Cela permet d'isoler les librairies du projet pour éviter les conflits.

**Windows :**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Mac / Linux :**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

# Installer les dépendances

Installez toutes les librairies Python nécessaires via pip :

```bash
pip install langchain langchain-community langchain-ollama langchain-chroma chromadb trafilatura streamlit ragas datasets pandas watchdog
```

# Configuration des Modèles(Ollama)

Le projet a besoin de 3 modèles spécifiques pour fonctionner. Lancez ces commandes dans votre terminal une par une :

**Le Cerveau (Chat)** : Modèle léger et rapide pour générer les réponses.

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
ollama pull mistral
```

# Utilisation

Suivez ces étapes dans l'ordre pour lancer le projet.

**Étape 1 : Récupération des Données (Scraping)**
Cette étape télécharge les articles depuis les URLs définies dans ***config.py*** et crée le fichier ***data/scraped_data.json***.

```bash
python main.py
```

**Étape 2 : Lancer l'Assistant (Interface Web)**
C'est la méthode recommandée pour utiliser l'assistant. Cela lance une interface graphique dans votre navigateur.*

```bash
streamlit run app.py
```

L'application sera accessible à l'adresse : http://localhost:8501

(Alternative : Pour tester en ligne de commande uniquement, lancez ***python main_local.py***)

# Évaluation (Optionnel)

Pour tester la qualité scientifique des réponses de votre IA (Fidélité et Pertinence) :

```bash
python eval_ragas.py
```

**Attention**: Ce script est gourmand en ressources. Il utilise le modèle **mistral** pour s'auto-évaluer. Le processus est configuré pour traiter les questions une par une afin d'éviter de surcharger votre ordinateur. Les résultats seront sauvegardés dans **resultats_evaluation.csv**.

# Structure du Projet

-> ***app.py*** : L'interface utilisateur (Frontend Streamlit).

-> ***rag_engine.py*** : Le cœur du système. Gère l'indexation ChromaDB et la génération de réponses.

-> ***scraper.py*** : Script de récupération des données Web.

-> ***main.py*** : Point d'entrée pour lancer le scraping.

-> ***main_local.py*** : Interface de chat en ligne de commande (CLI).

-> ***eval_ragas.py*** : Script d'audit de qualité (utilise Ragas).

-> ***config.py*** : Configuration globale (URLs, chemins de fichiers).

-> ***db_storage_local/*** : Dossier créé automatiquement contenant la base de données vectorielle.


# Dépannage

**Erreur "ChromaDB" ou modifications de données** : Si vous changez les données sources ou le modèle d'embedding, supprimez le dossier ***db_storage_local*** et relancez l'application. Elle reconstruira la base proprement.

**Lenteur** : C'est normal en local ("Inférence CPU"). La vitesse dépend de la puissance de votre processeur/RAM.

**Une erreur "Dimension mismatch" ou ChromaDB crash** : Cela arrive si vous changez de modèle d'embedding. Supprimez simplement le dossier ***db_storage_local/*** et relancez ***streamlit run app.py***. Le dossier sera recréé proprement.

**L'évaluation Ragas est trop lente** : C'est normal en local. Le script est configuré pour traiter 1 question à la fois (***max_workers=1***) pour éviter de faire planter votre ordinateur.

**L'IA répond en anglais.** : Le prompt système dans ***rag_engine.py*** force le français, mais les petits modèles (Llama 3.2) peuvent parfois dériver. Relancez la question.