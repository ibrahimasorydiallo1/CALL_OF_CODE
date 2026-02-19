# CALL OF CODE

#
#  Projet Pipeline d'Ingestion pour Maintenance Prédictive

#
## 🌟 1. Description du Projet

Ce projet consiste à concevoir, développer et mettre en production un **pipeline d'ingestion et de préparation de données (ETL)** robustes, automatisés et sécurisés pour la société **EnergiTech**.

Ce pipeline est la couche fondamentale destinée à alimenter un futur modèle d'**Intelligence Artificielle de maintenance prédictive** ciblant spécifiquement les turbines éoliennes du parc de l'entreprise. Il doit être capable de centraliser, nettoyer et structurer des données provenant de sources hétérogènes (internes et externes) afin de fournir un jeu de données unique, homogène et fiable pour les équipes de Data Science.

L'objectif principal est de passer d'une approche de maintenance réactive à une **approche proactive**, capable d'anticiper les dysfonctionnements grâce à l'analyse de signaux historiques et en temps réel.

## 🎯 2. Objectifs Clés du Pipeline

| Catégorie | Objectif | Bénéfice Associé |
| :--- | :--- | :--- |
| **Technique** | Concevoir un pipeline d'ingestion quotidien et automatisé. | Assurer la **disponibilité permanente** des données pour les modèles d'IA. |
| **Data Quality** | Transformer les flux hétérogènes en un jeu de données homogène et fiable. | Améliorer la disponibilité des actifs en détectant les signes précurseurs de panne. |
| **Sécurité** | Intégrer la traçabilité, le chiffrement (OAuth) et l'anonymisation des données. | Respecter les exigences légales strictes (notamment le **RGPD** et la norme **ISO 27001**). |
| **Évolutivité** | Créer une architecture réutilisable. | Extension possible à d'autres cas d’usage (prévision de production, optimisation de la consommation). |
| **Métier** | Charger les données dans une base relationnelle **PostgreSQL**. | Réduire les dépenses de maintenance en optimisant l'affectation des équipes. |

## 👥 3. Public Cible

| Segment | Rôle |
| :--- | :--- |
| **Data Scientists / Équipe R&D** | Utilisateurs principaux qui consommeront le jeu de données pour l'entraînement des modèles de maintenance prédictive. |
| **Ingénieurs Data** | Responsables de la maintenance, de l'extension et de l'audit technique du pipeline d'ingestion. |
| **Gestionnaires de la Conformité** | Pour la vérification des mécanismes de sécurité, de traçabilité et de respect du RGPD. |

## 💾 4. Sources et Destination des Données

Le pipeline doit gérer et intégrer des flux provenant de plusieurs origines :

* **Sources Hétérogènes à Ingérer :**
    * **Base de Données Interne :** Données brutes issues des capteurs (température, vent, consommation) sur les installations.
    * **Fichiers CSV :** Données d'historique ou de performance fournies par le service de production.
    * **API Météo Publiques :** Données environnementales externes nécessaires (vitesse du vent, etc.).

* **Destination Finale :**
    * Base de données relationnelle **PostgreSQL**.

## 🛠 5. Compétences Évaluées

Ce projet permet de valider les compétences techniques essentielles à un rôle d'Ingénieur Data ou de Développeur de pipeline ETL :

| Compétence | Description |
| :--- | :--- |
| **Acquisition de Données** | Définir les sources, recueillir les informations à partir de sources hétérogènes (internes et Open Data), et écrire des scripts d’importation automatisée et sécurisée. |
| **Qualité des Données (T)** | Analyser, nettoyer, trier et s’assurer de la qualité des données pour les rendre exploitables par la solution IA. |
| **Modélisation & Stockage (L)** | Construire la structure de stockage des données (modèle de données) qui répond au mieux au besoin d’analyse dans la base **PostgreSQL**. |
| **Sécurité & Accès** | Configurer les privilèges d’accès à la base de données relationnelle selon le **principe du moindre privilège**. |
| **Visualisation (Optionnel)** | Développer une interface utilisateur pour visualiser les données stockées et valider la bonne ingestion. |

## 📦 6. Productions (Livrables) Attendues

À l'issue de la mission, un ensemble de productions tangibles et documentées est attendu, organisé dans un répertoire unique nommé `Projet_Collecte_Données_IA` et livré sous forme d'archive compressée (`.zip`).

| Livrable | Description & Contenu | Public Cible |
| :--- | :--- | :--- |
| **Scripts d’Automatisation** | Le cœur fonctionnel du pipeline (Extraction, Transformation, Chargement). Doivent être écrits en **Python**. | Ingénieurs Data |
| **Schéma de la Base de Données** | Formalisation complète de la structure de stockage : **Modèle Conceptuel, Logique et Physique** des Données, ainsi que le script SQL de création de la base. | Ingénieurs Data |
| **Base de Données Peuplée** | Base de données opérationnelle, prête à l'interrogation. Fournie sous la forme d'un **dump SQL** (environ **12 000 enregistrements** pour un mois de mesures) accompagné d'un guide d'importation. | Data Scientists |
| **Tableau de Bord de Qualité** | Généré automatiquement à la fin de chaque ingestion, consultable dans un navigateur. Il doit synthétiser : **nombre de lignes extraites**, **taux de complétude**, **anomalies détectées** et un graphique de **distribution des valeurs critiques** (ex: vitesse du vent). | Responsables Conformité, Direction Technique |
| **Documentation Complète** | Ensemble des explications pour la compréhension, la maintenance et l'évolution. Comprend : **Rapport de Projet** (justification des choix, analyse des risques, limites, format PDF $\approx$ 10 pages) et une **Annexe Technique** (dépendances, exemples de requêtes SQL, procédure du job cron). | Toutes les équipes |

---

## ⚖️ 7. Critères d'Évaluation

L'évaluation de ce projet repose sur la combinaison de trois éléments principaux :

1.  **Qualité du Travail Réalisé :** Solidité, fiabilité et sécurité du pipeline d'ingestion.
2.  **Exhaustivité des Livrables :** Pertinence et documentation complète des productions demandées.
3.  **Soutenance Orale :** Capacité à présenter, justifier et valoriser le travail réalisé devant un client professionnel (public technique), démontrant la maîtrise technique et la communication efficace.

## Installation & Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/ibrahimasorydiallo1/CALL_OF_CODE.git
   cd CALL_OF_CODE
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Future direction

Il y'a toujours des axes améliorations comme le fine-tuning du model d'IA, comme l'analyse approfondie des données
à disposition mais nous sommes fiers de ce que nous avons accompli.

## LICENSE

Veuillez lire la [LICENSE](LICENSE) pour plus d'informations.

## Contact des collaborateurs

- Ibrahima Sory DIALLO. I am available on linkedin https://www.linkedin.com/in/ibrahima-sory-diallo-isd/
