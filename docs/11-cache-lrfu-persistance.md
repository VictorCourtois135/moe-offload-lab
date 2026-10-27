# 11 — Les caches : LRFU, épinglage, anti-oscillation, historique (étape 11)

Il y a deux caches (`TierCache`), un en VRAM et un en RAM, chacun avec un budget en
octets. Ils sont **partagés entre toutes les couches** (comme FreeToken) : une
couche très sollicitée peut occuper plus de place qu'une autre.

## Le score LRFU (le "LFRU" de Colibrì)

LRU (le plus récemment utilisé reste) et LFU (le plus fréquemment utilisé reste)
sont deux extrêmes. LRFU les réunit avec un seul compteur qui décroît avec le temps :

```
à chaque utilisation au temps t :   f <- f * gamma^(t - t_dernier) + 1
score maintenant :                   f * gamma^(maintenant - t_dernier)
gamma = 0.5 ^ (1 / demi_vie)
```

Exemple avec une demi-vie de 32 tokens : un expert utilisé 4 fois il y a 64 tokens
a un score de 4 x 0,25 = 1 ; un expert utilisé une seule fois au token précédent a
un score proche de 1. Les deux se valent. Demi-vie courte → comportement LRU,
demi-vie longue → comportement LFU.

Quand il faut de la place, on évince l'entrée de plus petit score, jamais une
entrée épinglée, jamais un expert en cours d'utilisation dans la couche.

## Une entrée par précision (le LHU de HOBBIT)

La clé d'une entrée est `(couche, expert, précision)`. La version fp16 et la version
int4 d'un même expert sont deux entrées, avec leurs propres scores : on suit donc
séparément l'usage en haute précision, comme le LHU de HOBBIT.

Une version plus précise peut servir une demande moins précise (un fp16 en cache
sert une demande int4, gain de qualité gratuit). Sauf en mode `lo`, qui doit
reproduire exactement une quantification statique.

## Épinglage et anti-oscillation (Colibrì)

Une entrée **épinglée** n'est jamais évincée par le chargement à la demande. Au
plus la moitié du budget est épinglable (le reste sert au cache dynamique).

Entre deux requêtes (point sûr), `repin()` épingle les entrées chaudes non
épinglées à la place d'entrées épinglées plus froides, avec deux garde-fous :

- **hystérésis de 25%** : le candidat doit être au moins 25% plus chaud que la victime ;
- **4 échanges maximum** par appel.

Sans ça, deux experts de popularité proche s'échangeraient la place à chaque
requête, et chaque échange coûte un transfert.

## Historique persistant (`usage.json`)

Le `.coli_usage` de Colibrì : chaque utilisation d'un expert est comptée, et
enregistrée à la fin de chaque requête dans `<store>\usage.json`. Au lancement
suivant, `warm_start` précharge et épingle les experts historiquement les plus
utilisés, en RAM puis en VRAM. Le système devient plus rapide à mesure qu'on
l'utilise sur le même type de tâches. L'historique est lié au store (`store_id`) :
un historique d'un autre modèle est ignoré.

Le benchmark, lui, démarre chaque réglage avec un historique vierge : sinon l'ordre
des réglages biaiserait la comparaison.

## Préchargement du pool basse précision

Si tout le pool int4 tient dans 60% du budget RAM (3,2 Gio pour OLMoE), il est
chargé et épinglé au démarrage : le disque ne sert plus qu'aux experts fp16. C'est
le "pool hôte" de FreeToken. Option `--preload-lo auto|yes|no`.

## La mémoire derrière le cache RAM : un pool de slots (`hostpool.py`)

Pour que les copies RAM -> VRAM soient asynchrones (et à pleine vitesse, 12,5 Go/s
mesurés au lieu de ~6), les tampons RAM doivent être en **mémoire épinglée**
(page-locked : Windows ne peut pas la déplacer, le GPU la lit directement).

Première version : un `torch.empty(n, pin_memory=True)` par lecture disque. Sur la
vraie machine, ça a planté (`CUDA_ERROR_OUT_OF_MEMORY` dans `cuMemHostAlloc`) avec un
budget de 5 Gio, parce que l'allocateur de PyTorch pour la mémoire épinglée :

- arrondit chaque demande à la puissance de 2 supérieure : 12 Mio -> 16 Mio (+33%),
  3,19 Mio -> 4 Mio (+25%) ;
- garde la mémoire libérée pour lui au lieu de la rendre, et ne la réutilise que
  pour la même taille arrondie : les tampons fp16 et int4 s'accumulent séparément.

La mémoire épinglée réelle dépassait donc largement le budget du cache.

Le pool fait comme les slots de Colibrì : des plaques de taille puissance de 2
(donc allouées sans perte), découpées en slots de la taille exacte d'un record,
réutilisés indéfiniment. Le total ne dépasse jamais le budget RAM + une marge
pour les lectures en cours. Si le pool est plein, ou si Windows refuse d'épingler
davantage, le runtime reçoit un tampon ordinaire : plus lent à copier, jamais
d'erreur.

Un slot libéré n'est pas réutilisable tout de suite : une copie vers la VRAM peut
encore le lire. Il passe en quarantaine jusqu'à ce qu'un événement CUDA enregistré
après ces copies soit passé, et au moins jusqu'à la couche suivante. `/stats` dans
le chat affiche l'état du pool.
