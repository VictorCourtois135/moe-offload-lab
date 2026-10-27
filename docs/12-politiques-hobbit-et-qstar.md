# 12 — Les deux décisions : HOBBIT et q* (étapes 9 et 10)

## 1. En quelle précision ? (HOBBIT, `PrecisionPolicy`)

Pour chaque token, les k poids du routeur sont **normalisés** (somme = 1), puis
chaque expert reçoit son score d'"unimportance" `s` = somme des poids des experts
mieux classés (docs/06).

| Condition | Décision |
|---|---|
| `s <= t1` (0,6 par défaut) | haute précision (fp16) |
| `s > t1` | basse précision (int4) |
| `s > t2` (désactivé par défaut) | expert sauté |

Le premier expert (`s = 0`) est toujours en haute précision. Sauter un expert change
les décisions du routeur : ce n'est fait que si on le demande (`--t2 0.9`).

**Le vote par expert (ajout de ce projet).** Un expert n'est chargé qu'une fois par
couche, pour tous les tokens qui l'ont choisi. En génération (1 token), c'est
exactement HOBBIT. En prefill (des centaines de tokens), il suffirait qu'un seul
token juge l'expert important pour que tous le reçoivent en fp16 : presque tout
finirait en fp16. Règle retenue : haute précision si au moins 50% de ses tokens la
demandent (`--hi-vote`). Ce problème a été trouvé par un test, pas anticipé.

**Couches protégées.** `--protect 0,1,15` (ou `edges`) : ces couches restent en fp16.
C'est la version simple de DyMoE, appliquée au pattern en U mesuré (docs/07).

**Modes.** `hi` = modèle d'origine ; `lo` = tout en int4, sans substitution (ce que
fait Colibrì) ; `hobbit` = décision par expert.

## 2. Où calculer un expert absent de la VRAM ? (q* de FreeToken)

Soit `m` experts manquants dans une couche. On en transfère `q*` en VRAM (ils y
restent pour la suite), et on calcule les `m - q*` autres directement sur CPU,
**en même temps** que les transferts.

```
q* = m · B_PCIe / B_Host        (arrondi, au moins 1 ; si B_Host <= B_PCIe : q* = m)
```

D'où ça vient : le transfert PCIe et le calcul CPU lisent tous les deux la RAM.
Quand le PCIe est saturé, il reste au CPU `B_Host - B_PCIe`. On égalise la durée des
deux branches :

```
q · S / B_PCIe  =  (m - q) · S / (B_Host - B_PCIe)    donc    q* = m · B_PCIe / B_Host
```

Exemple du papier (figure 2) : `B_PCIe : B_Host = 1 : 4` et 4 experts manquants →
1 transféré, 3 calculés sur CPU. C'est un test unitaire.

**Les valeurs sont mesurées, pas supposées** : `scripts\measure_bandwidth.py`
mesure B_PCIe (copie épinglée RAM → VRAM) et B_Host (octets d'expert traités par
seconde quand le CPU calcule un expert pour un token), et les enregistre dans
`bandwidth.json`.

**À quoi s'attendre sur ta machine.** Une RTX 2060 en PCIe 3.0 x16 copie ~12 Go/s.
Un CPU grand public qui calcule un expert fp16 traite probablement moins que ça
(conversion fp16 → fp32 comprise). Si B_Host <= B_PCIe, q* = m : tout est
transféré et la branche CPU ne sert pas. C'est exactement ce que dit FreeToken
("q* approche m quand B_Host approche B_PCIe"). La branche CPU prend tout son
sens sur les machines à mémoire rapide (DDR5, mémoire unifiée), ou quand
la partie dense est sur CPU.

**Qui est transféré en priorité ?** Les experts utilisés par le plus de tokens, puis
les plus utilisés historiquement.

**Simplification assumée.** FreeToken raisonne en nombre d'experts de même taille.
Ici les manquants peuvent mélanger fp16 et int4 ; on applique la formule telle
quelle, avec B_Host mesuré en fp16.
