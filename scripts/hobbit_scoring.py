"""
Scoring d'importance des experts, inspiré de HOBBIT (Tang et al., 2024, arXiv:2411.01433).

Principe : l'expert e_i contribue à la sortie via G(x)_ei * E_ei(x). Calculer la
sortie complète E_ei(x) est coûteux -- HOBBIT montre que la sortie du routeur seule
(||G(x)_ei||) est un proxy suffisant (corrélation de Pearson mesurée à 0.99 dans le
papier), sans avoir à exécuter l'expert.

Score d'importance (équation 2 du papier) : pour l'expert classé i-ème (0-indexed,
en ordre décroissant de score de routeur), le score d'"unimportance" est la somme
cumulée des scores des experts mieux classés que lui. Un score proche de 0 = expert
important (garder haute précision). Un score élevé = candidat à la basse précision
en cas de cache miss.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_hobbit_importance(router_scores: torch.Tensor) -> torch.Tensor:
    """
    router_scores: tensor de shape (top_k,), scores normalisés d'UN SEUL token,
                   DÉJÀ TRIÉS par ordre décroissant (c'est ce que retourne
                   torch.topk, utilisé en interne par le routeur MoE).

    Retourne : tensor de shape (top_k,), le score d'"unimportance" de chaque
               expert sélectionné. index 0 (l'expert le mieux classé) vaut
               toujours 0 -- HOBBIT le traite comme systématiquement important.
    """
    cumulative = torch.cumsum(router_scores, dim=0)
    unimportance = cumulative - router_scores
    unimportance[0] = 0.0
    return unimportance


def capture_router_output(model, layer_idx: int, inputs: dict) -> dict:
    """
    Exécute un passage complet du modèle et capture, via un forward hook, les
    scores et indices réels du routeur MoE à la couche `layer_idx`.

    Important : il faut passer par un vrai passage complet du modèle (embedding
    -> attention -> normalisation -> routeur), pas reconstruire le calcul à la
    main -- un test initial en sautant l'attention a produit une distribution
    quasi uniforme sur les experts, symptomatique d'un signal non enrichi par
    le contexte de la séquence (voir CHANGELOG.md, entrée du 2026-08-21).

    Retourne un dict {'scores': ..., 'indices': ...} pour le premier token
    de la séquence.
    """
    captured = {}

    def hook_fn(module, input, output):
        _, router_scores, router_indices = output
        captured['scores'] = router_scores
        captured['indices'] = router_indices

    gate = model.model.layers[layer_idx].mlp.gate
    handle = gate.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()

    return captured


if __name__ == "__main__":
    MODEL_PATH = "D:/models/olmoe-hf"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float16, device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    inputs = tokenizer("Le chat mange une souris", return_tensors="pt")
    captured = capture_router_output(model, layer_idx=0, inputs=inputs)

    scores = captured['scores'][0]
    indices = captured['indices'][0]
    importance = compute_hobbit_importance(scores)

    print("Experts sélectionnés :", indices)
    print("Scores du routeur    :", scores)
    print("Unimportance HOBBIT  :", importance)