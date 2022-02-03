# Generative Pseudo-labels Refinement for Unsupervised Domain Adaptation 


## Description du projet <a class="anchor" id="description"></a>  

Dans le cadre d'un cours sur la perception pour les systèmes autonomes, j’ai choisi d’implémenter une première approche d’adaptation de domaine non-supervisée (UDA). L’objectif de ce type d’approche est de développer un classifieur d’image performant pour un dataset ne disposant pas de labels.
L’adaptation de domaine consiste à mettre à profit un dataset source qui dispose pour sa part de données labellisées permettant l’apprentissage d’un classifieur. Ce modèle utilisé en prédiction sur les données cibles verra sa performance s’effondrer en comparaison de celle du domaine source.
Les approches d’adaptation de domaine visent donc à régler cette problématique et à développer un classifieur performant en prédiction sur un domaine cible ne disposant pas de données labellisées.
L’approche que j’ai choisie d’implémenter est tirée du papier [1], Générative Pseudo-Label Refinement.

## Description de l'approche <a class="anchor" id="apprche"></a>  
 
L’idée de ce papier est d’utiliser un classifieur pré-entrainé sur un domaine source pour inférer des pseudo-labels dans le domaine cible. Un nombre important de données seront mal classées par le classifieur. Les Gan conditionnels « cGan » ont des propriétés de robustesse sur des données dont les labels sont bruités. Cette propriété des cGAn est exploitée au travers d’une procédure itérative dans laquelle le cGAN et le classifieur apprennent de manière jointe.
J'ai implémenté cette approche en prenant comme domaine source le dataset « SVHN » et comme domaine cible le domaine « MNIST ». L’adaptation de domaine dans ce sens permet de réduire les temps de calcul des réseaux profonds (moins de couches).
Il y a deux grandes phases successives à réaliser pour implémenter cette méthode:

1. Une phase de pré-entrainement d'un classifieur C0 réalisé sur le dataset source (SVHN), ainsi que d'un cGAN GO et DO. 
2. Une phase d'apprentissage itérative qui alterne l'apprentissage du classifieur C sur un dataset généré par le générateur G et l'apprentissage du cGAN G et D su un dataset bruité par des pseudos-labels inférés pas le classifieur C.

## Phase 1 : pré-entrainement des modèles C0, G0 et D0 <a class="anchor" id="init"></a>  

### Pré-entrainement, performance et choix du modèle C0 <a class="anchor" id="C0"></a> 

Afin de choisir un réseau de neurones convolutif pertinent pour cette étude, j’ai comparé la performance en classification de plusieurs modèles sur le dataset SVHN préalablement converti en Grayscale.
• Le détail des réseaux CNN envisagés sont documentés dans le script « CNN.py »,
• L’apprentissage des modèles se fait via le script « Train-CNN_SVHN.py »
• La prédiction des modèles via le script « Predict.py »

Le modèle dont la performance a été la meilleure est celui du ResNet, c’est donc celui-ci qui a été retenu comme classifieur C0. Le modèle pré-entrainé est sauvegardé dans le dossier « Outputs sous le nom « Model_Trained_SVHN_C1.pth ». Les performances du classifieur C0 sont de :
• 95% sur le dataset SVHN converti en grayscale,
• 67% sur le dataset MNIST transformé en dimension 32x32.
Ces résultats démontrent bien la perte de performance en prédiction du classifieur sur le domaine cible.

### Pré-entrainement, performance et choix du modèle cGAN G0 et D0 <a class="anchor" id="G0D0"></a>

De la même manière il a fallu construire et développer un cGan sur le dataset MNIST transformé en 32 x 32 pixels. Lors du développement de ce cGan j’ai rencontré plusieurs problématiques notamment des problèmes de divergence des « loss » très marquées et de « collapse ». Afin de résoudre ces problématiques j’ai dû :
• Implémenter une approche de « label smoothing »,
• Introduire un bruit gaussien dans les images réelles et fausses en entrée du discriminateur D avec un decay,
• Implémenter un "flip" de 5% des labels positifs et négatifs en entrée du discriminateur,
• Implémenter un cDCGAN et complexifier les modèles G et D en ajoutant des convolutions et des « Batch Normalisation ». L’idée était de développer un générateur suffisamment fin pour et avec un contraste suffisamment qualitatif pour générer des images de qualités suffisantes.

Ces approches ont permis de stabiliser l'apprentissage du cGan et d'éviter l'effondrement de la perte du discriminant D et la dérive de la perte du générateur G. j’ai constaté de manière empirique que la performance du générateur était acceptable entre 15 et 25 epochs.
A noter que pour le cGAN le label est encodé de la même manière pour le générateur et le discriminateur en ajoutant par concaténation un canal « label » aux channels existants.
• Le détail des implémentations du « label smoothing », de l’introduction du bruit gaussien et du flip des labels est disponible via le script « Train_cGAN_MNIST.py »
• La structure du cGAn retenu est disponible via le script « cGan.py »
• L'apprentissage des pré-modèles a eu lieu via le script "PseudoLabelPretrain.py"
• Les images générées par le modèle G0 sont disponibles dans le répertoire /outputs/Résultats G0
• Les modèles G0 et D0 pré-entrainés sont sauvegardés dans le dossier « Outputs sous le nom « G0_MNIST.pth » et « D0_MNIST.pth ».

Images générées par C0 après 1 epoch :

<p align="center">
    <img src='outputs/Résultats G0/gan_generated_image_epoch_1.png'>
</p> 

Images générées par C0 après 25 epoch :

<p align="center">
    <img src='outputs/Résultats G0/gan_generated_image_epoch_25.png'>
</p> 

## Phase 2 : Apprentissage itératif de C, G et D <a class="anchor" id="iteration"></a> 

Durant cette phase, les modèles C, G et D font l'objet d'étapes d'apprentissage itératives. Lors de la pre-mière itération les modèles C, G et D correspondent aux modèles C0, G0 et D0. • L'étape 2.1 consiste à générer un dataset MNIST « cible » via le générateur conditionnel "G". Les labels des images sont tirés suivant une loi uniforme parmi les 10 classes disponibles. Les images conditionnelles et ces labels constitue un dataset « cible » qui permet d’affiner l'apprentissage du classifieur C.
• L’étape 2.2 consiste à affiner les modèles cGan G et D. Le dataset utilisé pour l'apprentissage est un échantillon aléatoire d'images du dataset MNIST dont les labels sont inférés par le classifieur C. J’ai décidé de conserver 25 epochs pour l’apprentissage du cGAN dans cette phase. Mon code me permet de générer des images à chaque epoch et ainsi de vérifier visuellement que les images générées sont acceptables.
En alternant ses deux étapes, l'asymétrie du bruit initial diminue et la performance du modèle C sur le dataset MNIST, et ce, alors que l'apprentissage se déroule de manière non supervisée.
J'ai mis en oeuvre cette approche sur 10 itérations. A chaque étape, 7680 échantillons sont tirés de ma-nière aléatoire pour les étapes d’apprentissage du classifieur C (2.1) et du conditionnel Gan G (2.2).
• Le code utilise pour implémenter ces itérations est retranscrit dans le script « PseudoLabelmain.py »..

Ci-dessous se trouvent les images générées par le générateur G à la dixième itération.

<p align="center">
    <img src='outputs/Résultats G apres 10 epochs/gan_generated_image_epoch_14.png'>
</p> 

## 3. Performance finale du classifier C sur le dataset MNIST <a class="anchor" id="resultats"></a>  

A l’issue des 10 itérations, la performance en termes de précision du classifieur « C » est de 93,5% sur le dataset MNIST. On constate donc que la performance du modèle a évolué de 67% à 93,5% avec l’approche « Pseudo-Label Refinement ». On peut donc considérer cette approche comme performante. Néanmoins, comme précisé dans le papier [3], Light - weight – calibrator, cette approche souffre d’une perte de performance du modèle sur le dataset source d’origine. En effet, j’ai recalculé la performance en précision du modèle final sur le dataset source « SVHN » et j’ai obtenu une performance de 25%, bien en deca des 95% d’origine.


## 4. Auteurs <a class="anchor" id="auteurs"></a>  
**Arnaud Sangouard** _alias_ [@Arnauyd](https://github.com/Arnauyd)

## Références <a class="anchor" id="réf"></a>

[1] GENERATIVE PSEUDO-LABEL REFINEMENT FOR UNSUPERVISED DOMAIN ADAPTATION, PIETRO MORERIO, RICCARDO VOLPI, RUGGERO RAGONESI, VITTORIO MURINO.
