"""
Agent local (05/10) : outils décrits depuis des fonctions Python, appels lus dans le texte
du modèle, boucle robuste (erreurs rendues au modèle), rendu du prompt avec et sans
outils natifs dans le modèle de chat, et boucle complète sur un modèle moelab miniature.
"""

from __future__ import annotations

import json

import torch

from moelab.agent import Agent, native_tools, parse_tool_calls, render, tool


@tool
def ajouter(a: int, b: int = 1) -> int:
    """Additionne deux entiers.

    Args:
        a: premier nombre
        b: second nombre
    """
    return a + b


@tool(name="echoue")
def _boom() -> str:
    """Lève toujours une erreur."""
    raise RuntimeError("panne")


def test_tool_schema():
    s = ajouter.schema()["function"]
    assert s["name"] == "ajouter" and s["description"] == "Additionne deux entiers."
    assert s["parameters"]["properties"]["a"] == {"type": "integer", "description": "premier nombre"}
    assert s["parameters"]["required"] == ["a"]
    assert _boom.name == "echoue"


def test_parse_tool_calls():
    text = ('Je regarde.\n<tool_call>\n{"name": "ajouter", "arguments": {"a": 2, "b": 3}}\n</tool_call>\n'
            '<tool_call>{"name": "etat", "arguments": "{}"}</tool_call>')
    content, calls = parse_tool_calls(text)
    assert content == "Je regarde."
    assert calls == [{"name": "ajouter", "arguments": {"a": 2, "b": 3}}, {"name": "etat", "arguments": {}}]
    # balise non fermée (génération coupée) et JSON dans un bloc de code
    _, calls = parse_tool_calls('<tool_call>\n```json\n{"name": "x", "arguments": {}}\n```')
    assert calls == [{"name": "x", "arguments": {}}]
    _, calls = parse_tool_calls("<tool_call>{pas du json}</tool_call>")
    assert calls[0]["name"] is None and "illisible" in calls[0]["error"]
    assert parse_tool_calls("Réponse simple.") == ("Réponse simple.", [])


def _scripted(replies):
    seen = []

    def fn(messages, tools):
        seen.append([dict(m) for m in messages])
        return replies[len(seen) - 1]

    return fn, seen


def test_agent_loop_and_errors():
    fn, seen = _scripted([
        '<tool_call>{"name": "ajouter", "arguments": {"a": 40, "b": 2}}</tool_call>'
        '<tool_call>{"name": "inconnu", "arguments": {}}</tool_call>',
        '<tool_call>{"name": "ajouter", "arguments": {"z": 1}}</tool_call><tool_call>{"name": "echoue", "arguments": {}}</tool_call>',
        "La réponse est 42.",
    ])
    agent = Agent([ajouter, _boom], fn)
    msgs = [{"role": "user", "content": "40 + 2 ?"}]
    assert agent.run(msgs) == "La réponse est 42."
    r1, r2 = agent.steps[0].results, agent.steps[1].results
    assert r1[0] == "42" and "outil inconnu" in r1[1]
    assert "Erreur d'arguments" in r2[0] and "panne" in r2[1]
    # le modèle a bien reçu les résultats, dans l'ordre
    roles = [m["role"] for m in msgs]
    assert roles == ["user", "assistant", "tool", "tool", "assistant", "tool", "tool", "assistant"]
    assert seen[1][2] == {"role": "tool", "name": "ajouter", "content": "42"}


def test_agent_max_steps():
    fn, _ = _scripted(['<tool_call>{"name": "ajouter", "arguments": {"a": 1}}</tool_call>'] * 5)
    agent = Agent([ajouter], fn, max_steps=3)
    assert "arrêt" in agent.run([{"role": "user", "content": "?"}]) and len(agent.steps) == 3


def test_render_fallback_and_native(tmp_path):
    from moelab.testing import make_tiny_tokenizer

    tok = make_tiny_tokenizer(tmp_path / "tok")
    assert not native_tools(tok)
    msgs = [{"role": "user", "content": "ab"}, {"role": "assistant", "content": "cd"},
            {"role": "tool", "name": "ajouter", "content": "42"}]
    text = tok.decode(render(tok, msgs, [ajouter.schema()])[0])
    assert "tool_call" in text and "ajouter" in text and "<tool_response>\n42\n</tool_response>" in text
    # modèle de chat qui sait décrire les outils (comme Qwen3) : c'est lui qui les rend
    tok.chat_template = ("{% if tools %}<|system|>\nOUTILS {{ tools | tojson }} tool_call\n{% endif %}"
                         + tok.chat_template)
    assert native_tools(tok)
    ids = render(tok, msgs[:1], [ajouter.schema()])
    assert "OUTILS" in tok.decode(ids[0])


def test_agent_on_tiny_moelab_model(olmoe_dir, olmoe_store, tmp_path):
    """Boucle complète sur un vrai modèle moelab (miniature, aléatoire) : le texte généré passe par le parseur."""
    from moelab import PrecisionPolicy, RuntimeConfig, load_offloaded_model
    from moelab.agent import moelab_generate_fn
    from moelab.testing import make_tiny_tokenizer

    tok = make_tiny_tokenizer(tmp_path / "tok")
    model, rt = load_offloaded_model(olmoe_dir, olmoe_store, RuntimeConfig(policy=PrecisionPolicy("lo"),
                                     host_budget_bytes=64 * 2**20, fast_budget_bytes=0), dense="cpu",
                                     dtype=torch.float32, log=lambda *a: None)
    timings = []
    gen = moelab_generate_fn(model, tok, max_new_tokens=6, timings=timings)
    agent = Agent([ajouter], gen, max_steps=2)
    out = agent.run([{"role": "user", "content": "salut"}])
    assert isinstance(out, str) and len(timings) == 1 and timings[0]["new_tokens"] <= 6
    json.dumps(agent.steps[0].__dict__, default=str)
    rt.close()


def test_prefix_length(tmp_path):
    from moelab.agent import prefix_length
    from moelab.testing import make_tiny_tokenizer

    tok = make_tiny_tokenizer(tmp_path / "tok")
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]
    ids = render(tok, msgs, [ajouter.schema()])
    n = prefix_length(tok, msgs, [ajouter.schema()], ids)
    pre = render(tok, msgs[:1], [ajouter.schema()], add_generation_prompt=False)
    assert n == pre.shape[1] and ids[0, :n].tolist() == pre[0].tolist()
    assert prefix_length(tok, msgs[1:], [], render(tok, msgs[1:], [])) is None


def test_future_dates_guard():
    """13/10 (chat) : « les trois dernières versions » écrites avec 2.15, « prévue le 28 octobre 2026 »."""
    import datetime as dt

    from moelab.app.factcheck import future_dates
    from moelab.app.tools import future_guard

    today = dt.date(2026, 10, 13)
    text = ("2.15 (prévu le 28 octobre 2026), 2.14 sorti le 2 septembre 2026, 2.13 : 2026-07-08, "
            "annonce October 28, 2026, réunion 01/12/2026, sans année : 30 octobre")
    assert future_dates(text, today) == ["28 octobre 2026", "October 28, 2026", "01/12/2026"]
    written = []
    write = future_guard(lambda **kw: written.append(kw) or "écrit", today)
    r = write(chemin="a.md", contenu=text)
    assert r.startswith("NON ÉCRIT") and "28 octobre 2026" in r and not written
    assert write(chemin="a.md", contenu=text) == "écrit"                       # réécrit tel quel : accepté
    assert write(chemin="b.md", contenu="2.14 : 2026-09-02") == "écrit"       # rien du futur : direct


def test_find_passages_never_empty_and_chat_archive():
    """13/10 : neuf tours perdus à deviner des titres absents ; résultats perdus au nettoyage du chat."""
    from moelab.agent import Tool
    from moelab.app.tools import archived, find_passages, rebuild_archive, relire_tool

    page = "PyTorch 2.14 Release Blog. We are excited to announce PyTorch 2.14 ... NVGEMM brings CUTLASS kernels. " * 3
    r = find_passages(page, "NVGEMM")
    assert r.startswith("3 passage(s)") or r.startswith("1 passage(s)")
    r = find_passages(page, "New CUTLASS Features")        # expression absente : ses mots un par un
    assert "n'apparaît pas tel quel" in r and "CUTLASS" in r
    r = find_passages(page, "Highlights")                   # rien : le début de la page
    assert "Voici son début" in r and "We are excited" in r

    archive: dict = {}
    big = Tool("lire_page", "lit", {"type": "object", "properties": {}}, lambda **kw: "x" * 9000 + " released 02 Sep")
    out = archived(big, archive).fn(url="u")
    assert out.startswith("[r1] ") and "relire('r1'" in out and len(out) < 4000
    assert "released 02 Sep" in relire_tool(archive).fn(ref="r1", chercher="released")
    saved = [{"role": "tool", "content": out, "archive": archive["r1"]}, {"role": "tool", "content": "[r2] court"}]
    again = rebuild_archive(saved)
    assert again == {"r1": archive["r1"], "r2": "court"}
    assert archived(big, again).fn(url="u").startswith("[r3] ")  # la numérotation continue


def test_page_dates_multi_terms_and_unwritten_block():
    """13/10 : année devinée (« 2023 ») faute de date complète ; un mot par appel ; fichier recopié dans la réponse."""
    from moelab.app.chat import unwritten_block
    from moelab.app.tools import find_passages, html_to_text

    html = ('<div>PyTorch 2.13.0 Release <relative-time datetime="2026-07-08T17:39:00Z" class="no-wrap">08 Jul'
            '</relative-time></div><p>FlexAttention on MPS</p><time datetime="2026-09-02">Sep 2</time>')
    t = html_to_text(html)
    assert "08 Jul (2026-07-08)" in t and "Sep 2 (2026-09-02)" in t     # le texte affiché et la date complète
    r = find_passages("a 2.14.0 b ... c 2.13.0 d ... e 2.12.0 f", "2.14.0 | 2.12.0", width=2)
    assert r.startswith("2 passage(s) contenant « 2.14.0 » ou « 2.12.0 »")

    req = r"Cherche ... et écris dans D:\Utilisateurs\Home\Documents\test\pytorch.md leur date"
    ans = "Voici le contenu :\n```markdown\n# PyTorch\n## 2.14.0\n```\nRemarque : ..."
    assert unwritten_block(req, ans, set()) == (r"D:\Utilisateurs\Home\Documents\test\pytorch.md", "# PyTorch\n## 2.14.0\n")
    assert unwritten_block(req, ans, {r"d:\utilisateurs\home\documents\test\pytorch.md"}) is None   # déjà écrit
    assert unwritten_block("Explique-moi ça", ans, set()) is None                                   # pas de fichier demandé
    assert unwritten_block(req, ans + "\n```py\nx\n```", set()) is None                             # deux blocs : ambigu
    assert unwritten_block("Lis https://pytorch.org/blog/a.md puis écris dans /home/v/x.md", ans, set())[0] == "/home/v/x.md"
