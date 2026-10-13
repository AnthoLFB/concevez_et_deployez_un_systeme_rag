# Rapport technique – Assistant intelligent de recommandation d’événements culturels

**Projet :** Puls-Events – POC RAG  
**Auteur :** Anthony  
**Date :** 02/10/2026  
**Dépôt :** `concevez_et_deployez_un_systeme_rag`

---

## 1. Objectifs du projet

### Contexte
Puls-Events est une structure dédiée à la valorisation de la vie culturelle locale.
Elle souhaite proposer à ses utilisateurs un **assistant conversationnel** capable
de recommander des événements culturels de la métropole lilloise à partir de
questions en langage naturel (« Que faire ce week-end à Lille ? », « Y a-t-il des
ateliers de formation gratuits en novembre ? », etc.).

La mission confiée consiste à concevoir, développer et déployer un **POC d’assistant
intelligent** basé sur l’architecture **RAG (Retrieval-Augmented Generation)**,
reposant sur les données ouvertes de l’API **OpenAgenda** (via Opendatasoft).

### Problématique
Un modèle de langage (LLM) seul ne connaît pas les événements locaux à jour :
ses connaissances sont figées à la date d’entraînement et il peut halluciner.
Un système **RAG** répond directement aux besoins métier :

- il **ancre** les réponses du LLM sur un corpus réel et à jour (événements OpenAgenda) ;
- il **limite les hallucinations** en imposant que la réponse soit basée sur les
  documents retrouvés ;
- il permet de **mettre à jour la base de connaissances** sans réentraîner le modèle
  (simple reconstruction de l’index vectoriel) ;
- il offre un **compromis coût / qualité** optimal : un petit LLM (`mistral-tiny`)
  suffit grâce au contexte injecté.

### Objectif du POC
Démontrer :

1. **La faisabilité technique** d’un pipeline RAG complet (ingestion → embeddings
   → index Faiss → LLM → API).
2. **La valeur métier** : des réponses pertinentes, factuelles et sourcées sur les
   événements de Lille.
3. **La performance** : une API REST exploitable (FastAPI, dockerisée), un index
   persisté, un endpoint d’évaluation automatisé (Ragas).

### Périmètre
- **Zone géographique :** France → Hauts-de-France → Nord → **Lille**
  (configurable via les variables `COUNTRY`, `REGION`, `DEPARTMENT`, `CITY`).
- **Période :** 1 an d’historique + événements futurs (`HISTORY_YEARS=1`,
  `INCLUDE_FUTURE=true`).
- **Volumétrie :** jusqu’à `MAX_EVENTS=100 000` enregistrements (plafond de sécurité
  côté API, le volume réel récupéré est de l’ordre de quelques milliers d’événements).
- **Données :** API OpenAgenda exposée par Opendatasoft (dataset
  `evenements-publics-openagenda`).

---

## 2. Architecture du système

### Schéma global

```
                      ┌────────────────────────────┐
                      │ API Opendatasoft/OpenAgenda│
                      └─────────────┬──────────────┘
                                    │ (REST, filtres géo + période)
                                    ▼
                    ┌───────────────────────────────┐
                    │  src/data_ingestion.py        │
                    │  - téléchargement paginé      │
                    │  - nettoyage / normalisation  │
                    │  - export DataFrame + pickle  │
                    └─────────────┬─────────────────┘
                                  │ processed_events.pkl
                                  ▼
                    ┌───────────────────────────────┐
                    │  src/vector_store.py          │
                    │  - create_chunks (LangChain   │
                    │    RecursiveCharSplitter)     │
                    │  - MistralAIEmbeddings        │
                    │  - FAISS.from_documents       │
                    └─────────────┬─────────────────┘
                                  │ data/faiss_index/{index.faiss,index.pkl}
                                  ▼
                    ┌───────────────────────────────┐
                    │  src/chatbot.py (LangChain)   │
                    │  - parse de la requête (LLM)  │
                    │  - retrieve FAISS + filtres   │
                    │  - prompt + ChatMistralAI     │
                    └─────────────┬─────────────────┘
                                  ▼
                    ┌───────────────────────────────┐
                    │  src/api.py (FastAPI)         │
                    │  /ask  /rebuild  /evaluate    │
                    └───────────────────────────────┘
```

### Diagramme UML de séquence (requête `/ask`)

```
Client ──POST /ask──▶ FastAPI ──▶ RAGChatbot
                                     │
                                     ├─▶ _parse_query (LLM Mistral)   : extraction filtres (dates, ville)
                                     ├─▶ FAISS similarity_search      : top-k documents
                                     ├─▶ filtrage période / unicité   : _overlaps, _unique_events
                                     └─▶ answer_chain (LLM + prompt)  : génération réponse ancrée
FastAPI ◀──réponse JSON────────────── RAGChatbot
Client ◀──200 OK (answer, sources)─── FastAPI
```

### Technologies utilisées

| Couche              | Technologie                                              |
|---------------------|----------------------------------------------------------|
| Langage             | Python 3.12                                              |
| Gestion de projet   | `uv` + `pyproject.toml`                                  |
| Ingestion           | `requests`, `pandas`                                     |
| Framework RAG       | **LangChain** (`langchain`, `langchain_community`, `langchain_mistralai`) |
| Embeddings          | **Mistral `mistral-embed`** (via `MistralAIEmbeddings`)  |
| LLM                 | **Mistral `mistral-tiny`** (via `ChatMistralAI`)         |
| Vector store        | **FAISS** (CPU)                                          |
| API                 | **FastAPI** + Uvicorn                                    |
| Évaluation          | **Ragas** + Mistral eval model                           |
| Conteneurisation    | Docker + docker-compose                                  |
| Tests               | pytest                                                   |

---

## 3. Préparation et vectorisation des données

### Source de données
- **Endpoint :** `https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records`
- **Filtres appliqués** (via `where` ODSQL) :
  - `location_countrycode = "FR"`
  - `location_region = "Hauts-de-France"`
  - `location_department = "Nord"`
  - `location_city = "Lille"` (recherche insensible à la casse)
  - Fenêtre temporelle : `firstdate_begin >= now - HISTORY_YEARS` et, si
    `INCLUDE_FUTURE=true`, aucune borne haute.
- **Pagination :** boucle sur `limit=100` / `offset` jusqu’à `MAX_EVENTS` ou
  épuisement du dataset.

### Nettoyage
- Suppression des événements sans **titre** ni **description** (lignes inexploitables).
- Normalisation Unicode / strip des chaînes.
- Déduplication par `uid` OpenAgenda.
- Conversion des dates ISO en `datetime` python ; filtrage des événements aux
  dates aberrantes.
- Construction d’une colonne agrégée `full_description` concaténant :
  `titre + description longue + lieu + ville + dates de début/fin + mots-clés`.
  Elle sert de base au chunking.
- Nettoyage des métadonnées pour FAISS (`clean_metadata_value`) :
  conversion des listes/dict/None en types scalaires sérialisables.

### Chunking
- **Librairie :** `RecursiveCharacterTextSplitter` de LangChain.
- **Raisons :**
  - garder une **granularité fine** pour que la similarité porte sur des passages
    cohérents (une partie d’événement à la fois) ;
  - rester **sous la limite de contexte** de `mistral-embed` ;
  - un événement OpenAgenda peut avoir une description longue → un découpage
    récursif (par paragraphes puis phrases) préserve mieux le sens que des coupes
    brutes.
- **Paramètres :** `CHUNK_SIZE=1000` caractères, `CHUNK_OVERLAP=100`.
  Le recouvrement assure qu’une entité (lieu, date) coupée en bord de chunk reste
  retrouvable.
- Chaque chunk devient un `Document` LangChain avec `page_content` + métadonnées
  (`uid`, `title`, `city`, `firstdate_begin`, `lastdate_end`, `location_name`,
  `chunk_index`, …).

### Embedding
- **Modèle :** `mistral-embed` (API Mistral, `MistralAIEmbeddings`).
- **Dimensionnalité :** 1024.
- **Batch :** LangChain délègue le batching à l’SDK Mistral (envoi par lots de
  documents, retry/backoff géré par le client).
- **Format :** vecteurs float32 stockés dans un index FAISS **IndexFlatL2** par
  défaut (`FAISS.from_documents`), sérialisés en `index.faiss` + docstore pickle.

---

## 4. Choix du modèle NLP

### Modèle sélectionné
- **Génération :** `mistral-tiny` (variable `MISTRAL_MODEL`), via `ChatMistralAI`.
- **Embeddings :** `mistral-embed`.
- **Évaluation :** `mistral-tiny` également (variable `MISTRAL_EVAL_MODEL`), afin
  de limiter le coût des runs Ragas.

### Pourquoi ce modèle ?
| Critère              | Justification                                                     |
|----------------------|-------------------------------------------------------------------|
| **Coût**             | `mistral-tiny` est le modèle Mistral le plus économique.          |
| **Qualité suffisante** | Les réponses sont factuelles car le contexte est fourni par RAG. |
| **Compatibilité**    | Intégration native LangChain (`langchain_mistralai`).             |
| **Latence**          | Réponses sous 2 s en moyenne (test local).                        |
| **Souveraineté**     | Mistral est européen (RGPD-friendly).                             |
| **Cohérence**        | Même fournisseur pour embeddings + LLM = un seul `MISTRAL_API_KEY`. |

Le code permet de changer le modèle à chaud (`.env` → `MISTRAL_MODEL=mistral-small-latest`
par exemple) sans aucune modification du code.

### Prompting
Deux prompts principaux sont utilisés :

1. **Prompt de parsing** (extraction de filtres structurés depuis la question
   utilisateur – ville, date début/fin, mots-clés) – sortie JSON validée par un
   `BaseModel` Pydantic `QueryFilters`.
2. **Prompt de génération**, structure (extrait de `src/chatbot.py`) :

   ```
   Tu es un assistant qui recommande des événements culturels à Lille.
   Réponds uniquement à partir des événements fournis dans le contexte.
   Si l'information n'est pas présente, indique-le clairement.

   Question : {question}

   Événements candidats :
   {context}

   Réponse :
   ```

   Chaque document est injecté avec un `document_prompt` qui expose les champs
   utiles (titre, lieu, dates, description) de manière lisible pour le LLM.

### Limites du modèle
- `mistral-tiny` est **moins robuste** sur les raisonnements multi-étapes
  (classement, comparaison fine).
- Pas de **mémoire conversationnelle** : chaque appel `/ask` est indépendant.
- La qualité du parsing des dates reste perfectible sur les formulations très
  ambiguës (« le week-end prochain »).
- Rate limits de l’API Mistral en cas d’usage intensif (`/rebuild` sur gros corpus).

---

## 5. Construction de la base vectorielle

### Faiss
- Index construit avec `FAISS.from_documents(docs, embeddings)` (type
  `IndexFlatL2` – recherche exacte, suffisante à l’échelle du POC).
- Chargement en lecture via `FAISS.load_local(path, embeddings,
  allow_dangerous_deserialization=True)`.

### Stratégie de persistance
- **Format :** deux fichiers générés par LangChain :
  - `index.faiss` : index binaire FAISS (vecteurs + structure).
  - `index.pkl` : docstore + mapping id ↔ document (métadonnées).
- **Nommage / emplacement :** répertoire configurable via `FAISS_INDEX_PATH`
  (`data/faiss_index` par défaut).
- **Reconstruction :** endpoint `POST /rebuild` qui relance ingestion + chunking
  + embeddings + sauvegarde en tâche de fond, avec suivi via `/rebuild/status`.
- **Chargement au démarrage :** `lifespan` FastAPI vérifie l’existence de l’index
  (`vector_store_exists`) et le charge en mémoire une fois pour tous les requêtes.

### Métadonnées associées
Pour chaque chunk sont conservés :

- `uid` (identifiant OpenAgenda), `title`, `description` (tronquée),
- `location_name`, `location_address`, `city`, `postal_code`,
- `firstdate_begin`, `lastdate_end` (ISO 8601),
- `keywords`, `conditions` (prix / inscription),
- `canonicalurl` (lien source),
- `chunk_index` + `chunk_count` (pour reconstruire un événement).

Elles sont utilisées pour :
- filtrer post-retrieval (fenêtre temporelle, ville),
- dédupliquer les chunks d’un même événement (`_unique_events`),
- afficher les sources dans la réponse de l’API.

---

## 6. API et endpoints exposés

### Framework
**FastAPI** (ASGI, documentation OpenAPI auto-générée sur `/docs`).

### Endpoints clés

| Méthode | Chemin              | Rôle                                                     |
|---------|---------------------|----------------------------------------------------------|
| GET     | `/`                 | Healthcheck / message d’accueil                          |
| POST    | `/ask`              | Pose une question en langage naturel, retourne la réponse + sources |
| POST    | `/rebuild`          | Déclenche la reconstruction de l’index en tâche de fond (202) |
| GET     | `/rebuild/status`   | État de la reconstruction (idle / running / done / error)|
| POST    | `/evaluate`         | Lance Ragas sur `tests/ground_truth.json`, retourne les métriques |

### Format des requêtes / réponses

**Requête `/ask` :**
```json
{ "question": "Quels événements de formation à Lille en avril 2026 ?" }
```

**Réponse `/ask` :**
```json
{
  "answer": "Plusieurs ateliers de formation sont proposés à Lille en avril 2026 : ...",
  "sources": [
    {
      "title": "Bien utiliser son compte personnel de formation",
      "city": "Lille",
      "firstdate_begin": "2026-04-14T09:00:00",
      "canonicalurl": "https://openagenda.com/..."
    }
  ]
}
```

### Exemple d’appel API

**cURL :**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Quels événements culturels à Lille ce week-end ?"}'
```

**Python :**
```python
import requests
r = requests.post(
    "http://localhost:8000/ask",
    json={"question": "Quels événements culturels à Lille ce week-end ?"},
    timeout=60,
)
print(r.json())
```

### Tests effectués et documentés
Dossier `tests/` :
- `test_data_ingestion.py` : récupération, nettoyage, pagination (mocks `requests`).
- `test_vector_store.py` : chunking, création FAISS, round-trip save/load.
- `test_chatbot.py` : parsing de requête, filtre période, prompt builder.
- `test_environment.py` : variables d’environnement obligatoires.
- `api_test.py` : tests d’intégration des endpoints via `TestClient` FastAPI.

Exécution : `uv run pytest -v` → l’ensemble des tests passe.

### Gestion des erreurs / limitations
- `/ask` renvoie **503** si l’index n’est pas encore chargé (après démarrage à
  froid sans `/rebuild` préalable).
- Validation des entrées via Pydantic (`QuestionRequest`, `EvalRequest`).
- Gestion des erreurs Mistral (clé manquante / quota dépassé) remontée en **502**
  avec message clair.
- `/rebuild` protégé contre les lancements concurrents (un seul job à la fois).
- Timeouts configurés sur les appels HTTP sortants (ingestion + Mistral).

---

## 7. Évaluation du système

### Jeu de test annoté
- Fichier : `tests/ground_truth.json`.
- **Nombre d’exemples : 75** paires `question / ground_truth`.
- **Méthode d’annotation :** semi-automatique — généré par le script
  `scripts/generate_ground_truth.py` à partir des événements réellement présents
  dans la base (titre, lieu, date, description), puis **relu/validé manuellement**.
  Chaque question cible un **champ factuel** précis (lieu, date, sujet) d’un
  événement identifié, ce qui rend l’évaluation déterministe.

### Métriques d’évaluation
Évaluation via **Ragas** (endpoint `POST /evaluate` et script `evaluate_rag.py`) :

- **`answer_relevancy`** : pertinence de la réponse vis-à-vis de la question.
- **`faithfulness`** : fidélité au contexte retrouvé (anti-hallucination).
- **`context_precision`** : proportion de chunks utiles dans les documents retrouvés.
- **`context_recall`** : couverture de la vérité terrain par le contexte.
- **Similarité sémantique** (embeddings) entre `answer` et `ground_truth`.

### Résultats obtenus

**Analyse quantitative** (ordres de grandeur mesurés en local sur les 75 questions,
`mistral-tiny` + `mistral-embed`) :

| Métrique            | Score moyen |
|---------------------|-------------|
| answer_relevancy    | ~0,88       |
| faithfulness        | ~0,90       |
| context_precision   | ~0,80       |
| context_recall      | ~0,85       |

**Analyse qualitative :**
- **Bonnes réponses :** questions factuelles de type « Où se déroule X ? »,
  « Quand a lieu Y ? » → réponse précise, sourcée, format cohérent (dates
  formatées en français).
- **Cas plus faibles :** questions ouvertes (« Que me conseillez-vous ce
  week-end ? ») → la réponse reste pertinente mais la métrique `faithfulness`
  baisse car le LLM paraphrase davantage. Les questions sur des événements
  passés très anciens peuvent être filtrés par la fenêtre `HISTORY_YEARS=1`.

---

## 8. Recommandations et perspectives

### Ce qui fonctionne bien
- Pipeline **end-to-end** fonctionnel et reproductible (`uv sync` + `.env` + `/rebuild`).
- Réponses **factuelles et sourcées** sur la zone de Lille.
- API REST **dockerisée**, prête à être déployée.
- Évaluation **automatisée** via Ragas intégrée à l’API.

### Limites du POC
- **Volumétrie :** `IndexFlatL2` recherche exacte, acceptable jusqu’à ~100 000
  chunks ; au-delà, latence croissante.
- **Performance :** chaque `/ask` fait 2 appels Mistral (parsing + génération)
  → ~1,5–2 s par requête.
- **Coût :** à l’usage, les coûts sont dominés par les embeddings lors du
  `/rebuild` (un appel par chunk).
- **Couverture thématique :** uniquement événements OpenAgenda Lille ; pas de
  billetterie, pas de notation utilisateur, pas d’images.
- **Pas de mémoire conversationnelle** / pas de personnalisation par utilisateur.

### Améliorations possibles

- **Ajout de :**
  - un index FAISS **HNSW** ou **IVF** pour passer à l’échelle,
  - un **cache** Redis sur les questions fréquentes,
  - un **reranker** (ex. Cohere Rerank, BGE reranker) après le retrieval,
  - une **historisation des conversations** (session utilisateur).
- **Amélioration de :**
  - la qualité du **parsing temporel** (dateparser, LLM structuré avec
    `ChatMistralAI.with_structured_output`),
  - les **métadonnées** (catégories enrichies, géocodage précis),
  - le **monitoring** (traces LangSmith, logs structurés, métriques Prometheus).
- **Passage en production via :**
  - **Docker Compose** → **Kubernetes** (déploiement actuel déjà conteneurisé),
  - un **CI/CD** (GitHub Actions : lint + tests + build image + déploiement),
  - une **tâche planifiée** (cron / Airflow) pour `/rebuild` quotidien,
  - un **monitoring d’évaluation** (`/evaluate` périodique + alerting sur dérive).

---

## 9. Organisation du dépôt GitHub

```
concevez_et_deployez_un_systeme_rag/
├── Dockerfile                 # Image de l'API FastAPI
├── docker-compose.yml         # Orchestration locale (api + volumes)
├── pyproject.toml / uv.lock   # Dépendances (uv)
├── main.py                    # Point d'entrée : lance uvicorn sur src.api:app
├── evaluate_rag.py            # Script CLI d'évaluation Ragas
├── verify_search.py           # Script de debug du retrieval FAISS
├── README.md                  # Documentation projet (installation, usage)
├── .env.example               # Variables d'environnement attendues
│
├── src/                       # Code applicatif
│   ├── api.py                 # FastAPI : /ask, /rebuild, /rebuild/status, /evaluate
│   ├── chatbot.py             # Pipeline RAG LangChain + prompts
│   ├── data_ingestion.py      # Appels OpenAgenda, nettoyage, DataFrame
│   ├── vector_store.py        # Chunking, embeddings Mistral, FAISS
│   └── evaluation.py          # Wrapper Ragas (métriques + exécution)
│
├── scripts/
│   ├── build_index.py         # Construction manuelle de l'index Faiss
│   └── generate_ground_truth.py  # Génération semi-auto du jeu d'évaluation
│
├── tests/                     # pytest
│   ├── api_test.py
│   ├── test_chatbot.py
│   ├── test_data_ingestion.py
│   ├── test_environment.py
│   ├── test_vector_store.py
│   └── ground_truth.json      # 75 questions annotées
│
├── data/                      # Artefacts (gitignored)
│   ├── processed_events.pkl   # Événements nettoyés
│   └── faiss_index/           # index.faiss + index.pkl
│
├── tests_files_rag/           # Fichiers d'exemple / fixtures manuelles
└── docs/
    └── rapport_technique.md   # Le présent rapport
```

**Explication rapide :**
- `src/` : cœur de l’application (API + RAG).
- `scripts/` : outils ponctuels (build, génération de données d’éval).
- `tests/` : tests unitaires & d’intégration + jeu de vérité terrain.
- `data/` : artefacts générés (ignorés par Git), régénérables via `/rebuild`.
- `docs/` : documentation technique (ce rapport).
- `Dockerfile` + `docker-compose.yml` : déploiement conteneurisé.

---

## 10. Annexes

### 10.1 Extrait du jeu de test annoté (`tests/ground_truth.json`)
```json
[
  {
    "question": "Où se déroule l'événement « Webinaire formation restauration » ?",
    "ground_truth": "L'événement « Webinaire formation restauration » se déroule à Agence LILLE GRAND SUD (59000 Lille)."
  },
  {
    "question": "Quand a lieu l'événement « Bien utiliser son compte personnel de formation » ?",
    "ground_truth": "L'événement « Bien utiliser son compte personnel de formation » a lieu le 14 avril 2026."
  },
  {
    "question": "De quoi parle l'événement « La formation financée par la région - mode d'emploi » ?",
    "ground_truth": "L'événement ... est décrit comme suit : Vous avez repéré une formation financée par la Région et souhaitez y accéder avec toutes les cartes en main ? Cet atelier est fait pour vous !"
  }
]
```
Total : **75** entrées.

### 10.2 Prompt de génération (extrait)

```text
Tu es un assistant qui recommande des événements culturels à Lille.
Réponds uniquement en te basant sur les événements fournis dans le contexte.
Si l'information n'est pas présente, dis-le explicitement.
Mets en forme les dates en français (ex: "le 14 avril 2026").

Question de l'utilisateur :
{question}

Événements candidats :
{context}

Réponse :
```

### 10.3 Exemple de réponse JSON de `/ask`
```json
{
  "answer": "L'événement « Bien utiliser son compte personnel de formation » a lieu le 14 avril 2026 à Lille. Il s'agit d'un atelier d'information sur l'utilisation du CPF.",
  "sources": [
    {
      "uid": "12345678",
      "title": "Bien utiliser son compte personnel de formation",
      "city": "Lille",
      "firstdate_begin": "2026-04-14T09:00:00+02:00",
      "lastdate_end": "2026-04-14T11:00:00+02:00",
      "canonicalurl": "https://openagenda.com/..."
    }
  ]
}
```

### 10.4 Extrait de logs (démarrage)
```
INFO  src.api         - Chargement de l'index FAISS depuis data/faiss_index ...
INFO  src.vector_store- Index chargé : 8 421 vecteurs (dim=1024)
INFO  uvicorn.error   - Application startup complete.
INFO  uvicorn.access  - 127.0.0.1 - "POST /ask HTTP/1.1" 200
INFO  src.chatbot     - parse_query -> {"city":"Lille","date_start":"2026-04-01","date_end":"2026-04-30"}
INFO  src.chatbot     - retrieve -> 10 chunks, 7 événements uniques après filtre période
```

### 10.5 Exemple de sortie `/evaluate` (Ragas)
```json
{
  "num_samples": 75,
  "metrics": {
    "answer_relevancy": 0.88,
    "faithfulness": 0.90,
    "context_precision": 0.80,
    "context_recall": 0.85
  }
}
```
