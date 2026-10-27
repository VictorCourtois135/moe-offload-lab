# 14 — Le benchmark A/B (étape 13)

## La question

À mémoire égale, sur la même machine, laquelle de ces politiques donne le meilleur
compromis qualité / vitesse ?

| Réglage | Ce que c'est |
|---|---|
| `hi` | tout en fp16 : le modèle d'origine, **référence de qualité et de vitesse** |
| `lo` | tout en int4, statique : ce que fait Colibrì aujourd'hui |
| `hobbit` | fp16 ou int4 selon l'importance de chaque expert pour chaque token |
| `hobbit_edges` | hobbit + couches 0, 1 et dernière toujours en fp16 (pattern en U, doc 07) |
| `hobbit_skip` | (optionnel) hobbit + saut des experts `s > 0.9` |

C'est exactement la question de l'issue Colibrì #249 (doc 16).

## Protocole

1. **Référence de qualité** : le texte `bench/eval_text.txt` passe en teacher
   forcing dans le modèle en mode `hi`.
2. Pour chaque réglage, dans un ordre alterné si `--repeats 2` (A B C D puis D C B A,
   pour neutraliser la dérive, comme le "reverse-order gate" de `coli tune`) :
   - caches vidés, historique vierge, même préchargement ;
   - **vitesse** : génération gloutonne de 64 tokens sur les 8 prompts de
     `bench/prompts.json` (TTFT et tokens/s, médiane sur les prompts) ;
   - **qualité** : le même texte en teacher forcing, comparé à la référence.
3. Verdict de chaque réglage contre `hi`.

## Règle d'acceptation

Adaptée de `coli tune` (Colibrì) :

- génération au moins **3% plus rapide** que la référence ;
- TTFT au plus **20% plus lent** ;
- qualité : **accord top-1 >= 99%** et **perplexité +1% au plus**.

Colibrì exige aussi que le taux de hit ne bouge pas de plus de 0,5 point. Ici les
réglages changent volontairement la précision, donc les taux de hit changent
forcément : on les rapporte sans en faire un critère.

## Lire le tableau

| Colonne | Sens |
|---|---|
| tok/s, TTFT | vitesse de génération, temps jusqu'au premier token |
| accord top-1 | % de positions où le token le plus probable est le même que le modèle d'origine |
| écart perplexité | dégradation de la prédiction du vrai texte, en % |
| KL | écart moyen entre les distributions (0 = identique) |
| VRAM hit | part des experts déjà en VRAM quand on en a besoin |
| fp16 demandé | part des experts que HOBBIT juge importants (à calculer en fp16) |
| fp16 utilisé | part réellement calculée en fp16 : plus haut que « demandé », car un fp16 déjà en cache sert aussi une demande int4 (gratuit) |
| CPU | part des experts calculés sur CPU (branche q*) |
| disque/token | lectures disque par token généré |
| prefetch | part des experts utilisés qui avaient été prédits |

## Lancer

```powershell
python scripts\benchmark.py --model D:\models\olmoe-hf --store C:\moe-store\olmoe
python scripts\benchmark.py --model D:\models\olmoe-hf --store C:\moe-store\olmoe --repeats 2
```

Durée indicative sur ta machine : 5 à 20 minutes par répétition selon le disque
(le mode `hi` lit beaucoup d'experts fp16). Les résultats sont écrits dans
`results\benchmark_<modèle>_<date>.md` (tableau) et `.json` (données brutes,
prompt par prompt, avec la configuration de la machine).

## Ce qu'on peut raisonnablement attendre

Ces hypothèses sont à vérifier, pas des résultats :

- `lo` est probablement le plus rapide : tout le pool int4 (3,2 Gio) tient en RAM,
  et presque en VRAM. Sa qualité est le point à mesurer.
- `hi` souffre du disque : 12 Mio par expert fp16 manquant, lus sur le HDD.
- `hobbit` devrait se placer entre les deux. S'il garde une qualité proche de `hi`
  en lisant beaucoup moins d'octets, c'est la réponse à l'issue #249.

Un résultat négatif (hobbit ni plus rapide que hi, ni meilleur que lo) est un vrai
résultat, à publier tel quel.

## Balayer le seuil t1

Un seuil propre à un réglage s'écrit avec `@` :

```powershell
python scripts\benchmark.py --model D:\models\olmoe-hf --store C:\moe-store\olmoe --configs hi,lo,hobbit@0.2,hobbit@0.4,hobbit@0.6,hobbit@0.8 --repeats 2
```

Premiers résultats et leur analyse : [doc 17](17-resultats-olmoe.md).
