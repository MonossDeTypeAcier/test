# Rapport de Projet : Machine Learning
Groupe: ER ROUASSE Ayoub, KONE Shyli, MOREL Robin et THIAW Alioune

---

## Sommaire

# Sommaire

1. Introduction  
   1.1 Historique du Machine Learning  
   1.2 Définition et enjeux  

2. Apprentissage supervisé  
   2.1 Kppv (k plus proches voisins)  
   2.2 MLP (Perceptron multicouche)  
   2.3 Discrimination fonctionnelle  
   2.4 SVM (Machines à vecteurs de support)  

3. Apprentissage non supervisé  
   3.1 Kmeans  
   3.2 DBScan  
   3.3 Carte de Kohonen  

4. Conclusion  

--- 

1. Introduction

1.1 Historique du Machine Learning

Le Machine Learning (ML) s'est développé à partir des avancées en intelligence artificielle, débutant avec les travaux d'Alan Turing et son test de Turing en 1950. L’apparition des premières méthodes d’apprentissage automatique, comme le Perceptron de Rosenblatt dans les années 1950, a marqué le début d’une approche computationnelle de la modélisation cognitive. L’émergence de l’apprentissage profond et des grandes bases de données au cours des années 2000 a révolutionné le domaine, favorisant son application dans des secteurs variés, notamment la bio-informatique, la finance et la reconnaissance d’images.

1.2 Définition et enjeux

Le Machine Learning est une sous-discipline de l'intelligence artificielle visant à développer des modèles capables de généraliser des structures sous-jacentes dans les données, sans instructions explicites. Les défis actuels incluent la complexité algorithmique, la généralisation sur des données hors distribution et la robustesse face aux perturbations adversariales.

2. Apprentissage supervisé

L’apprentissage supervisé repose sur l'utilisation d’un ensemble de données étiquetées pour entraîner un modèle, en minimisant une fonction de coût via des méthodes d'optimisation comme la descente de gradient.

2.1 Kppv (k plus proches voisins)

Le k-NN est une méthode d’apprentissage non paramétrique basée sur une fonction de similarité, généralement une distance métrique (euclidienne, Manhattan, Minkowski). La classification repose sur le vote majoritaire parmi les k voisins les plus proches, nécessitant une recherche efficace comme les KD-trees ou les ball trees pour améliorer la complexité computationnelle.

2.2 MLP (Perceptron multicouche)

Le MLP est un réseau de neurones feed-forward composé de plusieurs couches de neurones interconnectés. L’apprentissage repose sur l’optimisation des poids par rétropropagation de l’erreur et une descente de gradient stochastique. L’utilisation de fonctions d’activation non linéaires (ReLU, tanh, sigmoïde) permet d’accroître la capacité du réseau à modéliser des relations complexes.

2.3 Discrimination fonctionnelle

La discrimination fonctionnelle repose sur l’optimisation d’une fonction de séparation linéaire ou quadratique, exploitant la maximisation du critère de Fisher. Elle est couramment employée pour la classification de données gaussiennes multi-variées, avec une analyse des vecteurs propres pour maximiser la séparabilité inter-classes.

2.4 SVM (Machines à vecteurs de support)

Les SVM optimisent une frontière de décision en maximisant la marge entre les classes, en utilisant des noyaux (linéaires, RBF, polynomiaux) pour projeter les données dans des espaces de dimension supérieure. La résolution repose sur des méthodes d'optimisation convexe comme les multiplicateurs de Lagrange et la programmation quadratique.

3. Apprentissage non supervisé

L’apprentissage non supervisé repose sur l’analyse exploratoire des structures sous-jacentes dans des données non étiquetées.

3.1 Kmeans

L’algorithme Kmeans partitionne les données en k clusters en minimisant la variance intra-cluster par rapport aux centroïdes. Il est sensible aux conditions initiales, nécessitant des stratégies comme Kmeans++ pour une meilleure initialisation.

3.2 DBScan

DBScan segmente les données en fonction de leur densité locale, permettant la détection de clusters de formes arbitraires et l’identification de points aberrants. Il repose sur la définition d’un rayon de voisinage (ε) et d’un seuil minimal de points requis pour former un cluster.

3.3 Carte de Kohonen

Les cartes de Kohonen sont des réseaux de neurones non supervisés appliqués à la réduction dimensionnelle et au clustering. L’auto-organisation repose sur une mise à jour compétitive des poids des neurones à travers une fonction de voisinage gaussienne qui décroît au fil des itérations.

4. Conclusion

Le Machine Learning regroupe une diversité d’approches adaptées à des problématiques variées, de la classification supervisée à l’exploration de données non structurées. L’amélioration des algorithmes et des architectures computationnelles reste un enjeu majeur pour garantir la robustesse et la scalabilité des modèles appliqués à des volumes de données toujours plus importants.

