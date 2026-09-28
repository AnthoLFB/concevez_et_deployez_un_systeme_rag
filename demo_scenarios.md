# Scénarios de démonstration POC Puls-Events

Voici quelques scénarios réalistes pour tester la pertinence du système RAG lors de la soutenance.

## Scénario 1 : Recherche d'événements Jazz à Lille
*   **Question** : "Quels sont les concerts de jazz prévus à Lille prochainement ?"
*   **Attendu** : Le système doit retourner des événements dont la description contient "jazz", même si le titre ne le contient pas explicitement, en précisant les lieux et dates si disponibles dans le contexte.

## Scénario 2 : Activités pour enfants / familles
*   **Question** : "Je cherche une activité culturelle à faire avec mes enfants ce weekend à Lille, que me proposes-tu ?"
*   **Attendu** : Le chatbot doit identifier des événements avec des mots-clés comme "famille", "jeune public", "ateliers" ou "enfants".

## Scénario 3 : Exposition d'art
*   **Question** : "Y a-t-il des expositions d'art contemporain ou de peinture en ce moment ?"
*   **Attendu** : Liste des expositions en cours, idéalement avec le nom du musée ou de la galerie.

## Scénario 4 : Question hors contexte (Test de robustesse)
*   **Question** : "Quelle est la recette de la carbonara ?"
*   **Attendu** : Le système doit poliment répondre qu'il ne dispose pas de ces informations dans sa base d'événements culturels (si le prompt est bien configuré pour limiter les hallucinations).
