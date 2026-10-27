/* moelab : l'interface. Sans dépendance ni compilation : ce fichier est servi tel quel. */
"use strict";

// ------------------------------------------------------------------ outils
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (k === "html") e.innerHTML = v;
    else e.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) if (c !== null && c !== undefined && c !== false) e.append(c.nodeType ? c : String(c));
  return e;
}
async function api(method, path, body) {
  const r = await fetch(path, { method, headers: body ? { "Content-Type": "application/json" } : {},
                                body: body ? JSON.stringify(body) : undefined });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || `erreur ${r.status}`);
  return data;
}
function toast(msg, ms = 4000) {
  const t = $("#toast"); t.textContent = msg; t.hidden = false;
  clearTimeout(toast.h); toast.h = setTimeout(() => (t.hidden = true), ms);
}
const n1 = (x) => (x === null || x === undefined ? "–" : Number(x).toLocaleString("fr-BE", { maximumFractionDigits: 1 }));
const n2 = (x) => (x === null || x === undefined ? "–" : Number(x).toLocaleString("fr-BE", { maximumFractionDigits: 2 }));
const pct = (x) => (x === null || x === undefined ? "–" : Math.round(x * 100) + " %");
function visibleText(raw) { return (raw || "").replace(/<tool_call>[\s\S]*?(<\/tool_call>|$)/g, "").trim(); }
function follow(jobId, onEvent, after = 0) {
  const es = new EventSource(`/api/jobs/${jobId}/events?after=${after}`);
  es.onmessage = (m) => { const e = JSON.parse(m.data); onEvent(e); if (e.type === "done") es.close(); };
  es.onerror = () => { if (es.readyState === EventSource.CLOSED) return; };
  return es;
}

// ------------------------------------------------------------------ navigation
const loaders = {};
function show(name) {
  if (!$("#screen-" + name)) name = "chat";
  $$(".screen").forEach((s) => (s.hidden = s.id !== "screen-" + name));
  $$(".side a").forEach((a) => a.classList.toggle("active", a.dataset.screen === name));
  show.current = name;
  if (loaders[name]) loaders[name]();
}
window.addEventListener("hashchange", () => show(location.hash.slice(1)));

// ------------------------------------------------------------------ état (bande latérale)
let STATUS = null;
async function pollStatus() {
  try {
    STATUS = await api("GET", "/api/status");
    const st = STATUS;
    $("#strip-model").textContent = st.model ? `${st.model.name} · ${st.model.policy}` : "Aucun modèle chargé";
    const c = st.caches;
    const set = (k, frac, label) => { $("#bar-" + k).style.width = Math.min(100, (frac || 0) * 100) + "%"; $("#val-" + k).textContent = label; };
    if (c) {
      set("vram", c.vram.utilise_gio / (c.vram.budget_gio || 1), n2(c.vram.utilise_gio) + " Gio");
      set("ram", c.ram.utilise_gio / (c.ram.budget_gio || 1), n2(c.ram.utilise_gio) + " Gio");
      set("disk", (st.stats.disque_par_token || 0) / 200, n1(st.stats.disque_par_token) + "/tok");
    } else { set("vram", 0, "–"); set("ram", 0, "–"); set("disk", 0, "–"); }
    $("#strip-busy").textContent = st.busy ? `En cours : ${st.busy.title}` : (st.model ? "Libre" : "");
    const b = $("#queue-badge"); b.hidden = !st.queue; b.textContent = st.queue;
    if (show.current === "engine") renderEngineTop();
  } catch (e) { $("#strip-busy").textContent = "Serveur injoignable"; }
}
setInterval(pollStatus, 2000);

// ------------------------------------------------------------------ chat
const chat = { id: null, job: null, es: null };
async function loadConvs() {
  const list = await api("GET", "/api/conversations");
  const ul = $("#conv-list"); ul.replaceChildren();
  for (const c of list) {
    ul.append(el("li", { class: c.id === chat.id ? "active" : "" },
      el("button", { class: "open", title: c.titre, onclick: () => openConv(c.id) }, c.titre),
      el("button", { class: "del", title: "Supprimer", "aria-label": "Supprimer la conversation",
        onclick: async () => { if (!confirm(`Supprimer « ${c.titre} » ?`)) return;
          await api("DELETE", "/api/conversations/" + c.id); if (chat.id === c.id) chat.id = null; loadChat(); } }, "×")));
  }
  return list;
}
function statsLine(s) {
  if (!s) return null;
  return el("div", { class: "stats" },
    `${n2(s.tok_s)} tok/s · 1er token ${n2(s.premier_token_s)} s · ${s.tokens} tokens` +
    (s.repris_du_cache ? ` · ${s.repris_du_cache}/${s.prompt_tokens} tokens repris du cache` : "") +
    (s.tours > 1 ? ` · ${s.tours} tours` : ""));
}
function toolCard(name, args, result) {
  return el("details", { class: "toolcard" },
    el("summary", {}, `Outil ${name}(${args ? JSON.stringify(args) : ""})`),
    el("pre", {}, result || ""));
}
function renderThread(c) {
  const th = $("#thread"); th.replaceChildren();
  if (!c.messages.length) {
    th.append(el("p", { class: "empty" }, c.outils
      ? "Cette conversation a des outils : le modèle peut lire tes dossiers autorisés, consulter le moteur et calculer. Les permissions se règlent dans Outils et MCP."
      : "Pose ta question. Tout reste sur ce PC."));
  }
  let pendingCalls = [];
  for (const m of c.messages) {
    if (m.role === "user") th.append(el("div", { class: "msg user" }, m.content));
    else if (m.role === "assistant") {
      pendingCalls = [...(m.content || "").matchAll(/<tool_call>\s*([\s\S]*?)\s*(<\/tool_call>|$)/g)].map((x) => {
        try { return JSON.parse(x[1]); } catch { return { name: "?" }; } });
      const txt = m.visible ?? visibleText(m.content);
      if (txt || m.stats) th.append(el("div", { class: "msg assistant" }, el("div", { class: "body" }, txt), statsLine(m.stats)));
    } else if (m.role === "tool") {
      const call = pendingCalls.shift() || { name: m.name };
      th.append(toolCard(m.name || call.name, call.arguments, m.content));
    }
  }
  th.scrollTop = th.scrollHeight;
}
async function openConv(id) {
  chat.id = id;
  const c = await api("GET", "/api/conversations/" + id);
  $("#conv-title").textContent = c.titre;
  $("#conv-tools-wrap").hidden = false; $("#conv-tools").checked = !!c.outils;
  renderThread(c); loadConvs();
}
async function loadChat() {
  const list = await loadConvs();
  if (!chat.id && list.length) return openConv(list[0].id);
  if (!list.length) { $("#thread").replaceChildren(el("p", { class: "empty" }, STATUS && !STATUS.model
      ? "Aucun modèle n'est chargé. Va dans Mise en route pour en installer ou en charger un."
      : "Crée une conversation pour commencer.")); $("#conv-title").textContent = "Conversation"; }
}
loaders.chat = loadChat;
$("#new-conv").onclick = async () => {
  const c = await api("POST", "/api/conversations", { outils: $("#new-conv-tools").checked });
  await openConv(c.id); $("#msg").focus();
};
$("#conv-tools").onchange = async (e) => { if (chat.id) await api("PATCH", "/api/conversations/" + chat.id, { outils: e.target.checked }); };
$("#msg").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#composer").requestSubmit(); } });
$("#composer").onsubmit = async (e) => {
  e.preventDefault();
  const text = $("#msg").value.trim();
  if (!text || chat.job) return;
  if (!chat.id) { const c = await api("POST", "/api/conversations", { outils: $("#new-conv-tools").checked }); chat.id = c.id; }
  let job;
  try { job = await api("POST", `/api/conversations/${chat.id}/message`, { texte: text }); }
  catch (err) { toast(err.message); return; }
  $("#msg").value = "";
  const th = $("#thread"); $(".empty", th)?.remove();
  th.append(el("div", { class: "msg user" }, text));
  let raw = "", body = el("div", { class: "body cursor" }), box = el("div", { class: "msg assistant" }, body);
  const wait = el("div", { class: "stats" }, STATUS && STATUS.busy ? "En attente : le moteur termine « " + STATUS.busy.title + " »" : "");
  box.append(wait); th.append(box); th.scrollTop = th.scrollHeight;
  chat.job = job.id; $("#send").disabled = true; $("#stop").hidden = false;
  chat.es = follow(job.id, (ev) => {
    if (ev.type === "text") { wait.remove(); raw += ev.text; body.textContent = visibleText(raw); th.scrollTop = th.scrollHeight; }
    else if (ev.type === "tool") { body.classList.remove("cursor"); th.append(toolCard(ev.nom, ev.arguments, ev.resultat)); }
    else if (ev.type === "turn") { raw = ""; body = el("div", { class: "body cursor" }); box = el("div", { class: "msg assistant" }, body); th.append(box); }
    else if (ev.type === "approval") {
      const a = el("div", { class: "approval" },
        el("div", {}, `Le modèle veut utiliser ${ev.outil} avec ${JSON.stringify(ev.arguments)}`),
        el("div", { class: "actions" },
          el("button", { class: "primary", onclick: () => answer(true) }, "Autoriser"),
          el("button", { onclick: () => answer(false) }, "Refuser")));
      const answer = async (ok) => { await api("POST", `/api/jobs/${job.id}/approve`, { id: ev.id, accepte: ok }); a.remove(); };
      th.append(a); th.scrollTop = th.scrollHeight;
    } else if (ev.type === "done") {
      body.classList.remove("cursor"); chat.job = null; $("#send").disabled = false; $("#stop").hidden = true;
      if (ev.status === "erreur") box.append(el("div", { class: "stats ko" }, ev.error));
      else if (ev.result && ev.result.stats) box.append(statsLine(ev.result.stats));
      else if (ev.status === "annulé") box.append(el("div", { class: "stats" }, "Arrêté."));
      loadConvs().then(() => api("GET", "/api/conversations/" + chat.id)).then((c) => { $("#conv-title").textContent = c.titre; });
    }
  });
};
$("#stop").onclick = () => chat.job && api("POST", `/api/jobs/${chat.job}/cancel`);

// ------------------------------------------------------------------ agents
const ag = { job: null, es: null, tools: [], colors: {}, view: null };
const AGENT_HUES = 6;
function agentColor(name) {
  if (name === "orchestrateur") return "orch";
  if (!(name in ag.colors)) ag.colors[name] = Object.keys(ag.colors).length % AGENT_HUES;
  return "a" + ag.colors[name];
}
function toolChecks(box, names, checked) {
  box.replaceChildren(...names.map((t) => el("label", { class: "check", title: t.description || "" },
    el("input", { type: "checkbox", value: t.nom, checked: checked(t) }), t.nom,
    t.permission === "demander" ? el("span", { class: "tag" }, "demander") : null)));
}
const checkedTools = (box) => $$("input:checked", box).map((i) => i.value);
async function loadAgents() {
  const d = await api("GET", "/api/agents");
  ag.tools = d.outils;
  $("#ag-nofolder").hidden = d.dossiers.length > 0;
  if (!$("#ag-tools").children.length) toolChecks($("#ag-tools"), d.outils, (t) => t.nom !== "changer_reglage");
  if (!$("#ae-tools").children.length) toolChecks($("#ae-tools"), d.outils, () => false);
  const ul = $("#ag-library"); ul.replaceChildren();
  if (!d.agents.length) ul.append(el("li", { class: "empty" }, "Aucun pour l'instant : garde un agent après un run, ou crée-le ci-dessous."));
  for (const a of d.agents) {
    ul.append(el("li", {},
      el("button", { class: "open", title: "Modifier", onclick: () => editAgent(a) },
        el("b", {}, a.nom), el("span", {}, a.role)),
      el("button", { class: "del", "aria-label": `Supprimer ${a.nom}`, onclick: async () => {
        if (!confirm(`Supprimer l'agent « ${a.nom} » ?`)) return;
        await api("DELETE", "/api/agents/" + encodeURIComponent(a.nom)); loadAgents(); } }, "×")));
  }
  loadRuns(); loadLessons();
  if (!ag.job) {
    const jobs = await api("GET", "/api/jobs");
    const run = jobs.find((j) => j.kind === "agents" && ["en cours", "en attente"].includes(j.status));
    if (run) followRun(run.id, run.title.replace(/^Agents : /, ""));
  }
}
function editAgent(a) {
  $("#ag-edit").open = true; $("#ag-edit summary").textContent = `Modifier « ${a.nom} »`;
  $("#ae-name").value = a.nom; $("#ae-role").value = a.role;
  $$("input", $("#ae-tools")).forEach((i) => (i.checked = a.outils.includes(i.value)));
  $("#ae-name").focus();
}
$("#ae-save").onclick = async () => {
  try {
    const a = await api("POST", "/api/agents", { nom: $("#ae-name").value, role: $("#ae-role").value, outils: checkedTools($("#ae-tools")) });
    toast(`Agent « ${a.nom} » enregistré.`); $("#ae-name").value = ""; $("#ae-role").value = "";
    $("#ag-edit summary").textContent = "Créer un agent"; loadAgents();
  } catch (e) { toast(e.message); }
};
async function loadLessons() {
  const list = await api("GET", "/api/agents/lessons");
  const ul = $("#ag-lessons"); ul.replaceChildren();
  if (!list.length) ul.append(el("li", { class: "empty" }, "Aucune pour l'instant : « Corriger » sous une réponse en crée une."));
  for (const l of list) {
    ul.append(el("li", {},
      el("div", { class: "lesson" }, el("span", {}, l.texte), el("small", {}, `${l.source}, utilisée ${l.utilisations} fois`)),
      el("button", { class: "del", "aria-label": "Supprimer la leçon", onclick: async () => {
        await api("DELETE", "/api/agents/lessons/" + l.id); loadLessons(); } }, "×")));
  }
}
$("#al-save").onclick = async () => {
  try { await api("POST", "/api/agents/lessons", { texte: $("#al-text").value }); $("#al-text").value = ""; toast("Leçon enregistrée."); loadLessons(); }
  catch (e) { toast(e.message); }
};
async function loadRuns() {
  const runs = await api("GET", "/api/agents/runs");
  const ul = $("#ag-runs"); ul.replaceChildren();
  if (!runs.length) ul.append(el("li", { class: "empty" }, "Aucun run terminé."));
  for (const r of runs) {
    const when = r.date ? new Date(r.date * 1000).toLocaleString("fr-BE", { dateStyle: "short", timeStyle: "short" }) : "";
    ul.append(el("li", {}, el("button", { class: "open", onclick: () => openRun(r.id) },
      el("b", {}, r.objectif.split("\n")[0]), el("span", {}, `${when}, ${n1(r.minutes)} min, ${r.agents.length} agent${r.agents.length > 1 ? "s" : ""}`))));
  }
}

/* Le déroulé : une colonne pour l'orchestrateur ; chaque tâche confiée s'y branche, à la couleur de l'agent. */
function newView(objective, live) {
  const v = { root: el("div", { class: "run" }), timeline: el("ol", { class: "timeline" }), cur: null, task: {}, approvals: el("div") };
  v.head = el("div", { class: "run-head" }, el("h2", {}, objective), v.status = el("div", { class: "run-status" }, live ? "En attente du moteur…" : ""));
  v.final = el("section", { class: "final", hidden: true });
  v.kept = el("div", { class: "kept", hidden: true });
  v.root.append(v.head, v.approvals, v.final, v.kept, v.timeline);
  $("#ag-view").replaceChildren(v.root);
  v.orchItem = null;
  return v;
}
function orchTurn(v) {
  const body = el("div", { class: "think" });
  const li = el("li", { class: "turn orch" }, el("div", { class: "who" }, "Orchestrateur"), body);
  v.timeline.append(li);
  v.cur = { agent: "orchestrateur", body, raw: "", li };
  return v.cur;
}
function agentCreated(v, a) {
  const c = agentColor(a.agent);
  v.timeline.append(el("li", { class: `turn created ${c}` },
    el("div", { class: "who" }, `Nouvel agent : ${a.agent}`),
    el("p", { class: "role" }, a.role),
    el("div", { class: "tags" }, (a.outils.length ? a.outils : ["bloc-notes seulement"]).map((t) => el("span", { class: "tag" }, t)))));
}
function taskStart(v, a) {
  const c = agentColor(a.agent);
  const card = el("li", { class: `turn task ${c}` },
    el("div", { class: "who" }, `${a.agent} · tâche ${a.numero}`),
    el("p", { class: "ask" }, a.tache));
  const steps = el("div", { class: "steps" });
  card.append(steps);
  v.timeline.append(card);
  v.task[a.agent] = { card, steps };
  return v.task[a.agent];
}
function agentTurn(v, name) {
  const t = v.task[name] || taskStart(v, { agent: name, numero: "?", tache: "" });
  const body = el("div", { class: "think" });
  t.steps.append(body);
  v.cur = { agent: name, body, raw: "" };
  return v.cur;
}
function report(v, a) {
  const t = v.task[a.agent];
  if (t) t.card.append(el("div", { class: "report" }, el("div", { class: "lbl" }, "Compte rendu"), el("div", { class: "txt" }, a.compte_rendu)));
}
function container(v, agent) {
  if (agent === "orchestrateur") return v.cur && v.cur.agent === "orchestrateur" ? v.cur.li : orchTurn(v).li;
  return (v.task[agent] || taskStart(v, { agent, numero: "?", tache: "" })).steps;
}
function criteriaCard(v) {
  if (!v.crit) {
    v.crit = { body: el("div", { class: "think" }), list: el("ul", { class: "crit-list" }) };
    v.timeline.prepend(el("li", { class: "turn crit" }, el("div", { class: "who" }, "Critères de réussite"), v.crit.body, v.crit.list));
  }
  return v.crit;
}
function showCriteria(v, crit) {
  const c = criteriaCard(v);
  c.body.textContent = ""; c.list.replaceChildren(...crit.map((x) => el("li", {}, x)));
}
function askCriteria(v, ev, jobId) {
  const ta = el("textarea", { rows: Math.max(3, ev.criteres.length + 1) }, ev.criteres.map((x) => "- " + x).join("\n"));
  let left = Math.round(ev.delai_s);
  const timer = el("span", { class: "hint" }, "");
  const tick = () => { timer.textContent = `Sans réponse, le run continue dans ${left} s avec ces critères.`; left -= 1; };
  tick(); const h = setInterval(() => { if (left < 0) clearInterval(h); else tick(); }, 1000);
  const send = async (value) => { clearInterval(h); await api("POST", `/api/jobs/${jobId}/approve`, { id: ev.id, accepte: true, valeur: value }); box.remove(); };
  const box = el("div", { class: "approval crit-ask" },
    el("div", {}, el("b", {}, "Critères de réussite : "), "corrige-les si l'objectif est mal compris (un par ligne)."),
    ta, timer,
    el("div", { class: "actions" },
      el("button", { class: "primary", onclick: () => send(ta.value) }, "Valider ces critères"),
      el("button", { onclick: () => send(null) }, "Continuer sans changer")));
  v.approvals.append(box); v.status.textContent = "En attente de ta validation des critères";
}
function verifiedLine(v, ev, objective) {
  const rej = (ev.rejetes || []).length ? ` (${ev.rejetes.length} remarque(s) ignorée(s) : citation introuvable)` : "";
  if (ev.saute) { v.timeline.append(el("li", { class: "turn verdict" }, "Vérification finale sautée : limite de temps du run dépassée.")); return; }
  if (ev.ok) { v.timeline.append(el("li", { class: "turn verdict ok" }, "Vérificateur : tous les critères sont remplis." + rej)); return; }
  const keep = el("button", { onclick: async (e) => {
    await api("POST", "/api/agents/lessons", { texte: ev.problemes.slice(0, 600), objectif: objective, source: "verificateur" });
    e.target.disabled = true; e.target.textContent = "Gardé comme leçon"; loadLessons(); } }, "Garder comme leçon");
  v.timeline.append(el("li", { class: "turn verdict ko" }, el("div", { class: "who" }, "Vérificateur : problèmes trouvés" + rej),
    el("div", { class: "txt" }, ev.problemes), keep));
}
function showFinal(v, text) {
  v.final.hidden = false;
  v.final.replaceChildren(el("div", { class: "lbl" }, "Réponse"),
    el("div", { class: "txt" }, text),
    el("button", { onclick: () => navigator.clipboard.writeText(text).then(() => toast("Réponse copiée.")) }, "Copier la réponse"),
    v.runId ? correctBox(v.runId) : null);
}
function correctBox(runId) {
  const ta = el("textarea", { rows: 2, placeholder: "Ce qui n'allait pas. Ex. : il manque la version 2.14, la plus récente." });
  return el("details", { class: "correct" }, el("summary", {}, "Corriger (devient une leçon pour les prochains runs)"),
    ta, el("button", { onclick: async (e) => {
      try { await api("POST", `/api/agents/runs/${runId}/corriger`, { texte: ta.value }); e.target.disabled = true;
        e.target.textContent = "Leçon enregistrée"; loadLessons(); } catch (err) { toast(err.message); } } }, "Enregistrer la correction"));
}
function showKeep(v, runId, agents) {
  const list = Object.values(agents || {}).filter((a) => a.origine === "orchestrateur");
  if (!list.length) return;
  v.kept.hidden = false;
  v.kept.replaceChildren(el("span", {}, "Garder un agent pour les prochains runs :"),
    ...list.map((a) => el("button", { onclick: async (e) => {
      await api("POST", `/api/agents/runs/${runId}/garder`, { agent: a.nom });
      e.target.disabled = true; e.target.textContent = `${a.nom} gardé`; loadAgents(); } }, a.nom)));
}
function onRunEvent(v, ev, jobId) {
  if (ev.type === "speaker" && ev.agent === "criteres") {
    v.status.textContent = "Préparation des critères de réussite…";
    v.cur = { agent: "criteres", body: criteriaCard(v).body, raw: "" };
  } else if (ev.type === "speaker") {
    v.status.textContent = ev.agent === "orchestrateur" ? "L'orchestrateur réfléchit…" : ev.agent === "verificateur" ? "Vérification finale…" : `${ev.agent} travaille…`;
    if (ev.agent === "orchestrateur") orchTurn(v); else agentTurn(v, ev.agent);
  } else if (ev.type === "text" && v.cur) {
    v.cur.raw += ev.text; v.cur.body.textContent = visibleText(v.cur.raw);
  } else if (ev.type === "tool") {
    if (ev.nom === "creer_agent" && !String(ev.resultat).startsWith("Erreur")) return;
    if (["terminer", "rendre", "noter"].includes(ev.nom)) return;  // déjà montrés (réponse, compte rendu, note)
    container(v, ev.agent).append(toolCard(ev.nom, ev.arguments, ev.resultat));
  } else if (ev.type === "agent_created") agentCreated(v, ev);
  else if (ev.type === "agent_start") { taskStart(v, ev); v.cur = null; }
  else if (ev.type === "agent_done") { report(v, ev); v.cur = null; }
  else if (ev.type === "note") container(v, ev.agent).append(el("div", { class: "note" }, `Note : ${ev.texte}`));
  else if (ev.type === "verification") container(v, ev.agent).append(el("div", { class: "note warn" }, `Vérification (${ev.ou}) : non retrouvé dans ce qui a été lu : ${ev.elements.join(", ")}`));
  else if (ev.type === "criteria_ask") askCriteria(v, ev, jobId);
  else if (ev.type === "criteria_ask_done") { $$(".crit-ask", v.approvals).forEach((x) => x.remove()); }
  else if (ev.type === "criteria") showCriteria(v, ev.criteres);
  else if (ev.type === "verified") verifiedLine(v, ev, v.objective);
  else if (ev.type === "final") showFinal(v, ev.reponse);
  else if (ev.type === "log") v.timeline.append(el("li", { class: "turn log" }, ev.texte));
  else if (ev.type === "approval") {
    const box = el("div", { class: "approval" },
      el("div", {}, `${ev.outil} veut s'exécuter avec ${JSON.stringify(ev.arguments)}`),
      el("div", { class: "actions" },
        el("button", { class: "primary", onclick: () => answer(true) }, "Autoriser"),
        el("button", { onclick: () => answer(false) }, "Refuser")));
    const answer = async (ok) => { await api("POST", `/api/jobs/${jobId}/approve`, { id: ev.id, accepte: ok }); box.remove(); };
    v.approvals.append(box); v.status.textContent = "En attente de ton accord";
  } else if (ev.type === "approval_done") { v.approvals.replaceChildren(); }
  else if (ev.type === "done") {
    v.status.textContent = ev.status === "terminé" ? `Terminé en ${n1(ev.result && ev.result.minutes)} min${timeLine(ev.result && ev.result.temps)}${ctxLine(ev.result && ev.result.contexte)}`
      : ev.status === "annulé" ? "Arrêté" : `Erreur : ${ev.error}`;
    v.status.classList.toggle("ko", ev.status === "erreur");
    ag.job = null; $("#ag-go").disabled = false; $("#ag-go").textContent = "Lancer l'orchestrateur";
    $$(".think:empty", v.root).forEach((x) => x.remove());
    if (ev.status !== "erreur") api("GET", `/api/agents/runs/${jobId}`).then((r) => showKeep(v, jobId, r.agents)).catch(() => {});
    loadRuns();
  }
}
function timeLine(t) {
  if (!t) return "";
  const parts = [];
  if (t.criteres_s) parts.push(`critères ${n1(t.criteres_s)} s`);
  if (t.verification_s) parts.push(t.verification_s < 90 ? `vérification ${n1(t.verification_s)} s` : `vérification ${n1(t.verification_s / 60)} min`);
  return parts.length ? " · " + parts.join(", ") : "";
}
function ctxLine(c) {
  if (!c || !c.prompt_tokens) return "";
  const pctReused = Math.round(100 * c.repris / c.prompt_tokens);
  return ` · contexte : ${c.generations} générations, ${pctReused} % des tokens de prompt repris du cache` +
    (c.nettoyages ? `, ${c.nettoyages} nettoyage(s)` : "") + (c.archives ? `, ${c.archives} résultat(s) archivé(s)` : "");
}
function followRun(jobId, objective) {
  ag.job = jobId; ag.colors = {};
  const v = newView(objective, true); v.runId = jobId; v.objective = objective;
  v.status.append(" ", el("button", { class: "danger", onclick: () => api("POST", `/api/jobs/${jobId}/cancel`) }, "Arrêter"));
  $("#ag-go").disabled = true; $("#ag-go").textContent = "Un run est en cours";
  ag.es = follow(jobId, (ev) => {
    const near = innerHeight + scrollY >= document.body.scrollHeight - 80;
    onRunEvent(v, ev, jobId);
    if (near && ev.type !== "done") v.root.lastElementChild.scrollIntoView({ block: "end" });
  });
}
async function openRun(id) {
  const r = await api("GET", "/api/agents/runs/" + id);
  ag.colors = {};
  const v = newView(r.objectif, false); v.runId = id; v.objective = r.objectif;
  if (r.criteres && r.criteres.length) showCriteria(v, r.criteres);
  v.status.textContent = `${n1(r.minutes)} min · ${Object.keys(r.agents).length} agent(s)${timeLine(r.temps)}${ctxLine(r.contexte)}`;
  if (r.reponse) showFinal(v, r.reponse);
  const seen = {};
  for (const st of r.orchestrateur) {
    const t = orchTurn(v); t.body.textContent = st.texte;
    for (const c of st.appels) {
      if (c.nom === "creer_agent" && !c.resultat.startsWith("Erreur")) {
        const a = r.agents[(c.resultat.match(/« (.+?) »/) || [])[1]];
        if (a) agentCreated(v, { agent: a.nom, role: a.role, outils: a.outils });
      } else if (c.nom === "confier") {
        const name = Object.keys(r.agents).find((n) => n === String(c.arguments.agent || "").toLowerCase()) ||
          (c.resultat.match(/Compte rendu de (\S+) :/) || [])[1];
        const a = r.agents[name];
        const k = seen[name] = (seen[name] || 0) + 1;
        const task = a && a.taches[k - 1];
        if (!task) { v.timeline.append(toolCard(c.nom, c.arguments, c.resultat)); continue; }
        const tk = taskStart(v, { agent: name, numero: k, tache: task.tache });
        for (const s of task.etapes) {
          if (s.texte) tk.steps.append(el("div", { class: "think" }, s.texte));
          for (const x of s.appels) {
            if (x.nom === "noter") tk.steps.append(el("div", { class: "note" }, `Note : ${x.arguments.texte || ""}`));
            else if (x.nom !== "rendre") tk.steps.append(toolCard(x.nom, x.arguments, x.resultat));
          }
        }
        report(v, { agent: name, compte_rendu: task.compte_rendu || "(pas de compte rendu)" });
        v.cur = null;
      } else if (c.nom === "noter") t.li.append(el("div", { class: "note" }, `Note : ${c.arguments.texte || ""}`));
      else if (c.nom !== "terminer") t.li.append(toolCard(c.nom, c.arguments, c.resultat));
    }
  }
  if (r.verification) verifiedLine(v, r.verification, r.objectif);
  $$(".think:empty", v.root).forEach((x) => x.remove());
  showKeep(v, id, r.agents);
}
loaders.agents = loadAgents;
$("#ag-form").onsubmit = async (e) => {
  e.preventDefault();
  if (ag.job) return;
  const goal = $("#ag-goal").value.trim();
  if (!goal) return;
  try {
    const job = await api("POST", "/api/agents/run", {
      objectif: goal, outils: checkedTools($("#ag-tools")),
      max_agents: +$("#ag-max-agents").value, max_etapes: +$("#ag-max-steps").value,
      etapes_par_tache: +$("#ag-sub-steps").value, minutes: +$("#ag-minutes").value,
      criteres: $("#ag-criteria").checked, attente_criteres: +$("#ag-wait").value, verification: $("#ag-verify").value });
    followRun(job.id, goal);
  } catch (err) { toast(err.message, 6000); }
};

// ------------------------------------------------------------------ mise en route
const setup = { model: null, machine: null };
async function loadSetup() {
  const [m, cat] = await Promise.all([api("GET", "/api/setup/machine"), api("GET", "/api/setup/catalog")]);
  setup.machine = m;
  $("#machine").replaceChildren(
    el("div", {}, m.gpu ? `${m.gpu.nom} : ${n1(m.gpu.vram_totale_gio)} Go de VRAM (${n1(m.gpu.vram_libre_gio)} libres)` : "Pas de GPU NVIDIA détecté"),
    el("div", {}, `RAM : ${n1(m.ram_totale_gio)} Go (${n1(m.ram_dispo_gio)} disponibles), ${m.cpu_threads} fils CPU`),
    el("ul", { class: "plain" }, m.disques.map((d) => el("li", {}, `Disque ${d.chemin} : ${n1(d.libre_gio)} Go libres sur ${n1(d.total_gio)}`))));
  const box = $("#catalog"); box.replaceChildren();
  for (const e of cat) {
    box.append(el("div", { class: "model-card" + (setup.model === e.id ? " selected" : "") },
      el("h3", {}, e.nom),
      el("p", {}, `${e.params}. ${e.note}`),
      el("div", { class: "sizes" },
        el("span", {}, `Téléchargement ~${n1(e.telechargement_go)} Go`),
        el("span", {}, `Store int4 ${n1(e.stores_gio.int4)} Gio`),
        el("span", {}, `int8 + int4 ${n1(e.stores_gio["int8+int4"])} Gio`)),
      el("p", {}, el("button", { onclick: () => choose(e) }, "Choisir ce modèle"))));
  }
  loadLibrary();
}
loaders.setup = loadSetup;
function defaultDirs(e) {
  const disks = (setup.machine?.disques || []).slice().sort((a, b) => b.libre_gio - a.libre_gio);
  const root = (disks[0]?.chemin || "C:\\").replace(/[\\/]$/, "");
  const sep = root.includes("/") || root === "" ? "/" : "\\";
  return { dl: `${root}${sep}models${sep}${e.id}`, st: `${root}${sep}moe-store${sep}${e.id}` };
}
function choose(e) {
  setup.model = e.id;
  $$(".model-card").forEach((c) => c.classList.toggle("selected", c.querySelector("h3").textContent === e.nom));
  const d = defaultDirs(e);
  const p = $("#plan"); p.hidden = false;
  requestAnimationFrame(() => p.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" }));
  p.replaceChildren(el("h3", {}, e.nom),
    el("div", { class: "form" },
      el("label", {}, "Dossier du téléchargement", el("input", { id: "p-dl", value: d.dl })),
      el("label", {}, "Dossier du store", el("input", { id: "p-st", value: d.st })),
      el("label", { class: "check" }, el("input", { type: "checkbox", id: "p-keep" }), "garder le checkpoint après l'installation (sinon supprimé : le store est autonome)"),
      el("button", { class: "primary", onclick: checkPlan }, "Vérifier la place")),
    el("div", { id: "p-result" }));
}
async function checkPlan() {
  const body = { model_id: setup.model, download_dir: $("#p-dl").value, store_dir: $("#p-st").value, garder_checkpoint: $("#p-keep").checked };
  let p; try { p = await api("POST", "/api/setup/plan", body); } catch (e) { return toast(e.message); }
  const r = $("#p-result"); r.replaceChildren();
  r.append(el("p", {}, p.deja_telecharge ? "Déjà téléchargé." :
    `Téléchargement : ${n1(p.telechargement_gio)} Gio (libres : ${n1(p.libre_telechargement_gio)} Gio). Store : ${n1(p.libre_store_gio)} Gio libres` +
    (p.meme_disque ? " sur le même disque." : " sur l'autre disque.")));
  for (const w of p.avertissements) r.append(el("p", { class: "ko" }, w));
  const name = "prec";
  for (const c of p.choix) {
    r.append(el("label", { class: "choice" + (c.tient ? "" : " nofit") },
      el("input", { type: "radio", name, value: c.precisions, disabled: !c.tient, checked: c.precisions === p.recommande }),
      `${c.precisions} : ${n1(c.store_gio)} Gio`,
      el("span", { class: c.tient ? "ok" : "ko" }, c.tient ? "tient" : "pas assez de place"),
      c.precisions === p.recommande ? el("b", {}, "recommandé") : null));
  }
  r.append(el("p", { class: "hint" }, "int4 : le plus rapide (politique lo). int8 + int4 : meilleure qualité possible avec hobbit, 3 fois plus de place."));
  if (p.recommande) r.append(el("button", { class: "primary", onclick: () => install(body) }, "Installer"));
  else r.append(el("p", { class: "ko" }, "Aucune option ne tient : libère de la place ou choisis d'autres disques."));
}
async function install(body) {
  const prec = $("input[name=prec]:checked")?.value || "int4";
  let job; try { job = await api("POST", "/api/setup/install", { ...body, precisions: prec }); } catch (e) { return toast(e.message); }
  watchInstall(job.id);
}
function watchInstall(id) {
  const box = $("#install"); box.hidden = false;
  const step = el("div", { class: "strip-model" }, "Démarrage…"), bar = el("i"), log = el("div", { class: "log" });
  const cancel = el("button", { class: "danger", onclick: () => api("POST", `/api/jobs/${id}/cancel`) }, "Annuler");
  box.replaceChildren(el("h3", {}, "Installation"), step, el("div", { class: "progress" }, bar), log, cancel);
  follow(id, (ev) => {
    if (ev.type === "step") { step.textContent = ev.texte; bar.style.width = "0"; }
    else if (ev.type === "progress") {
      const f = ev.total ? ev.fait / ev.total : 0; bar.style.width = Math.round(f * 100) + "%";
      if (ev.etape === "telechargement") step.textContent = `Téléchargement : ${n1(ev.fait)} / ${n1(ev.total)} Go`;
      if (ev.etape === "store") step.textContent = `Construction du store : couche ${ev.fait}/${ev.total}, ~${Math.ceil(ev.reste_s / 60)} min restantes`;
    } else if (ev.type === "log") { log.textContent += ev.texte + "\n"; log.scrollTop = log.scrollHeight; }
    else if (ev.type === "done") {
      cancel.remove();
      step.textContent = ev.status === "terminé" ? "Installé. Charge-le depuis la bibliothèque." : `Arrêté : ${ev.error || ev.status}`;
      bar.style.width = ev.status === "terminé" ? "100%" : bar.style.width; loadLibrary();
    }
  });
}
async function loadLibrary() {
  const { modeles, dernier, charge } = await api("GET", "/api/models");
  const box = $("#library"); box.replaceChildren();
  if (!modeles.length) box.append(el("p", { class: "hint" }, "Aucun modèle. Installe-en un, ou ajoute un store déjà construit."));
  for (const m of modeles) {
    const loaded = charge && charge.store === m.store;
    const last = dernier && dernier.store === m.store ? dernier : {};
    const ram = el("input", { type: "number", step: "0.5", min: "1", placeholder: "auto", value: last.ram_gb ?? "", "aria-label": "RAM pour les experts (Gio)" });
    const vram = el("input", { type: "number", step: "0.1", min: "0.5", placeholder: "auto", value: last.vram_gb ?? "", "aria-label": "VRAM pour les experts (Gio)" });
    const pol = el("select", { "aria-label": "Politique" }, ["lo", "hobbit", "hi"].map((p) => el("option", { value: p, selected: (last.policy || "lo") === p }, p)));
    box.append(el("div", { class: "lib-item" + (loaded ? " loaded" : "") },
      el("b", {}, m.nom + (loaded ? " (chargé)" : "")),
      el("div", { class: "meta" }, `${m.model_type} · ${m.precisions.join(" + ")} · ${n1(m.taille_gio)} Gio · ${m.store}`),
      el("div", { class: "tags" },
        el("span", { class: "tag" + (m.present ? " on" : "") }, m.present ? "store présent" : "store introuvable"),
        el("span", { class: "tag" + (m.autonome ? " on" : "") }, m.autonome ? "autonome" : "a besoin du checkpoint"),
        m.profil_autopilot ? el("span", { class: "tag on" }, "profil autopilot") : null),
      el("div", { class: "actions" },
        el("label", {}, "RAM (Gio) ", ram), el("label", {}, "VRAM (Gio) ", vram), el("label", {}, "Politique ", pol),
        el("button", { class: "primary", onclick: () => loadModel(m.id, ram.value, vram.value, pol.value) }, loaded ? "Recharger" : "Charger"),
        !m.autonome && m.checkpoint_present ? el("button", { onclick: () => makeStandalone(m.id) }, "Rendre autonome") : null,
        m.autonome && m.checkpoint_present ? el("button", { class: "danger", onclick: () => removeModel(m, { supprimer_checkpoint: true },
          `Supprimer le checkpoint ${m.checkpoint} ? Le store est autonome : le modèle reste utilisable.`) }, "Supprimer le checkpoint") : null,
        el("button", { onclick: () => removeModel(m, {}, `Retirer ${m.nom} de la bibliothèque ? (aucun fichier supprimé)`) }, "Retirer"))));
  }
}
async function loadModel(id, ram, vram, policy) {
  const job = await api("POST", "/api/models/load", { id, ram_gb: ram ? +ram : null, vram_gb: vram ? +vram : null, policy });
  toast("Chargement en cours… (environ une minute)");
  follow(job.id, (ev) => { if (ev.type === "done") { toast(ev.status === "terminé" ? "Modèle chargé." : "Échec : " + ev.error, 6000); loadLibrary(); pollStatus(); } });
}
async function makeStandalone(id) {
  const job = await api("POST", `/api/models/${id}/standalone`, {});
  toast("Copie de la partie dense dans le store…");
  follow(job.id, (ev) => { if (ev.type === "done") { toast(ev.status === "terminé" ? "Store autonome." : "Échec : " + ev.error); loadLibrary(); } });
}
async function removeModel(m, opts, question) {
  if (!confirm(question)) return;
  try { await api("POST", `/api/models/${m.id}/remove`, opts); } catch (e) { return toast(e.message); }
  loadLibrary();
}
$("#add-btn").onclick = async () => {
  try { await api("POST", "/api/models/add", { store: $("#add-store").value.trim(), checkpoint: $("#add-ckpt").value.trim() || null }); toast("Ajouté."); loadLibrary(); }
  catch (e) { toast(e.message); }
};

// ------------------------------------------------------------------ autopilot
const ap = { rows: [], job: null };
function drawChart() {
  const rows = ap.rows.filter((r) => r.reponse_type_s);
  const box = $("#ap-chart");
  if (!rows.length) { box.replaceChildren(el("p", { class: "hint" }, "Les essais s'afficheront ici.")); return; }
  const W = 600, H = 240, P = 36;
  const ys = rows.map((r) => r.reponse_type_s), max = Math.max(...ys) * 1.05, min = Math.min(...ys) * 0.9;
  const x = (i) => P + (i * (W - 2 * P)) / Math.max(1, rows.length - 1), y = (v) => H - P - ((v - min) / (max - min || 1)) * (H - 2 * P);
  let best = Infinity; const bestPts = [];
  rows.forEach((r, i) => { if (r.essai === 0 || String(r.verdict).startsWith("GARDÉ")) best = Math.min(best, r.reponse_type_s); bestPts.push(`${x(i)},${y(best)}`); });
  const pts = rows.map((r, i) => `${x(i)},${y(r.reponse_type_s)}`).join(" ");
  const dots = rows.map((r, i) => `<circle class="${String(r.verdict).startsWith("GARDÉ") || r.essai === 0 ? "kept" : "rejected"}" cx="${x(i)}" cy="${y(r.reponse_type_s)}" r="5"><title>Essai ${r.essai} : ${r.reponse_type_s} s</title></circle>`).join("");
  const labels = rows.map((r, i) => `<text x="${x(i)}" y="${H - 12}" text-anchor="middle">${r.essai}</text>`).join("");
  box.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Réponse type par essai">
    <line class="axis" x1="${P}" y1="${H - P}" x2="${W - P}" y2="${H - P}"/>
    <text x="4" y="${y(max) + 4}">${n1(max)} s</text><text x="4" y="${y(min) + 4}">${n1(min)} s</text>
    <polyline class="line" points="${pts}"/><polyline class="best" points="${bestPts.join(" ")}"/>${dots}${labels}</svg>`;
}
function addTrial(row) {
  ap.rows = ap.rows.filter((r) => r.essai !== row.essai).concat([row]).sort((a, b) => a.essai - b.essai);
  const tb = $("#ap-journal tbody"); tb.replaceChildren();
  for (const r of ap.rows) {
    const ch = r.changements === "départ" ? "départ" : Object.entries(r.changements).map(([k, v]) => `${k} = ${v}`).join(", ");
    tb.append(el("tr", { class: String(r.verdict).startsWith("GARDÉ") ? "kept" : "" },
      el("td", { class: "num" }, r.essai), el("td", {}, ch), el("td", { class: "num" }, n1(r.reponse_type_s) + " s"),
      el("td", { class: "num" }, n2(r.tok_s)), el("td", { class: "num" }, n2(r.premier_token_s) + " s"), el("td", {}, r.verdict)));
  }
  drawChart();
}
function followAutopilot(id) {
  ap.job = id; ap.rows = []; $("#ap-journal tbody").replaceChildren(); drawChart();
  const th = $("#ap-thoughts"); th.replaceChildren();
  let cur = el("div"); th.append(cur);
  $("#ap-go").disabled = true; $("#ap-stop").hidden = false;
  follow(id, (ev) => {
    if (ev.type === "trial") addTrial(ev.ligne);
    else if (ev.type === "text") { cur.textContent += ev.text; th.scrollTop = th.scrollHeight; }
    else if (ev.type === "tool") { th.append(el("div", { class: "t-tool" }, `${ev.nom}(${JSON.stringify(ev.arguments)})`)); cur = el("div"); th.append(cur); }
    else if (ev.type === "log" && /essai|dérive|départ|Vérification|meilleur/.test(ev.texte)) { th.append(el("div", { class: "t-log" }, ev.texte.trim())); cur = el("div"); th.append(cur); }
    else if (ev.type === "done") {
      $("#ap-go").disabled = false; $("#ap-stop").hidden = true; ap.job = null;
      const s = ev.result;
      th.append(el("p", {}, el("b", {}, ev.status === "terminé" && s
        ? `Fini : réponse type ${n1(s.depart_reponse_type_s)} → ${n1(s.meilleur_reponse_type_s)} s (${s.gain_pourcent > 0 ? "+" : ""}${n1(s.gain_pourcent)} %). ${s.profil ? "Profil enregistré : il sera appliqué au prochain chargement." : "Pas de profil écrit."}`
        : `Arrêté : ${ev.error || ev.status}`)));
      if (s && s.rapport) th.append(el("p", {}, s.rapport));
      loadReports();
    }
  });
}
async function loadReports() {
  const list = await api("GET", "/api/autopilot/reports");
  $("#ap-reports").replaceChildren(...list.map((r) => el("li", {}, el("button", { onclick: async () => {
    const d = await api("GET", "/api/autopilot/reports/" + r.nom); const w = window.open("", "_blank");
    w.document.write(`<pre style="white-space:pre-wrap;font:14px/1.5 system-ui;max-width:80ch;margin:2rem auto">${d.texte.replace(/[<&]/g, (c) => ({ "<": "&lt;", "&": "&amp;" }[c]))}</pre>`); } },
    r.nom.replace("autopilot_", "").replace(".md", "")))));
  if (!list.length) $("#ap-reports").append(el("li", { class: "empty" }, "Aucun rapport pour l'instant : chaque run en écrit un."));
}
loaders.autopilot = async () => {
  loadReports(); drawChart();
  if (!ap.job) { const jobs = await api("GET", "/api/jobs"); const run = jobs.find((j) => j.kind === "autopilot" && ["en cours", "en attente"].includes(j.status)); if (run) followAutopilot(run.id); }
};
$("#ap-form").onsubmit = async (e) => {
  e.preventDefault();
  if (!STATUS?.model) return toast("Charge d'abord un modèle (Mise en route).");
  const job = await api("POST", "/api/autopilot", { essais: +$("#ap-trials").value, minutes: +$("#ap-minutes").value,
                                                     depart: $("#ap-start").value, aveugle: $("#ap-blind").checked });
  followAutopilot(job.id);
};
$("#ap-stop").onclick = () => ap.job && api("POST", `/api/jobs/${ap.job}/cancel`);

// ------------------------------------------------------------------ outils et MCP
let mcpRows = [];
async function loadTools() {
  const t = await api("GET", "/api/tools");
  const tb = $("#tools-table tbody"); tb.replaceChildren();
  for (const o of t.outils) {
    const sel = el("select", { "aria-label": "Permission de " + o.nom, onchange: async (e) => {
      await api("POST", "/api/tools/permission", { nom: o.nom, permission: e.target.value }); toast(`${o.nom} : ${e.target.value}`); } },
      t.permissions.map((p) => el("option", { value: p, selected: p === o.permission }, p)));
    tb.append(el("tr", {}, el("td", {}, el("b", {}, o.nom)), el("td", {}, o.description), el("td", {}, sel)));
  }
  $("#folders").value = t.dossiers.join("\n");
  mcpRows = t.mcp_serveurs.map((s) => ({ ...s }));
  renderMcp(t.mcp_etat);
  $("#expose").checked = t.exposer_mcp; $("#expose").disabled = !t.mcp_disponible;
  renderExpose(t.exposer_mcp, t.mcp_disponible);
}
loaders.tools = loadTools;
function renderMcp(state = {}) {
  const box = $("#mcp-list"); box.replaceChildren();
  if (!mcpRows.length) box.append(el("p", { class: "hint" }, "Aucun serveur. Exemple : nom « fichiers », commande « npx », arguments « -y @modelcontextprotocol/server-filesystem C:\\Users\\Home\\Documents »."));
  mcpRows.forEach((s, i) => {
    const st = state[s.nom];
    box.append(el("div", { class: "panel form" },
      el("label", {}, "Nom", el("input", { value: s.nom || "", oninput: (e) => (s.nom = e.target.value) })),
      el("label", {}, "Commande", el("input", { value: s.commande || "", oninput: (e) => (s.commande = e.target.value) })),
      el("label", {}, "Arguments (séparés par des espaces)", el("input", { value: (s.args || []).join(" "), oninput: (e) => (s.args = e.target.value.split(/\s+/).filter(Boolean)) })),
      el("label", { class: "check" }, el("input", { type: "checkbox", checked: s.actif !== false, onchange: (e) => (s.actif = e.target.checked) }), "actif"),
      st ? el("div", { class: st.status === "connecté" ? "ok" : "ko" }, `${st.status}${st.outils.length ? " : " + st.outils.join(", ") : ""}`) : null,
      el("button", { class: "danger", onclick: () => { mcpRows.splice(i, 1); renderMcp(state); } }, "Retirer")));
  });
}
function renderExpose(on, available) {
  const box = $("#expose-info");
  box.hidden = !on && available;
  if (!available) { box.hidden = false; box.replaceChildren(el("p", { class: "ko" }, "Le paquet mcp n'est pas installé : pip install mcp")); return; }
  const url = `${location.origin}/mcp/`;
  box.replaceChildren(
    el("p", {}, `Adresse : ${url} (outils : demander, etat_moteur, lancer_autopilot).`),
    el("p", { class: "hint" }, "VS Code (.vscode/mcp.json) :"),
    el("pre", { class: "log" }, JSON.stringify({ servers: { moelab: { type: "http", url } } }, null, 2)),
    el("p", { class: "hint" }, "Claude Desktop (claude_desktop_config.json, avec le pont mcp-remote) :"),
    el("pre", { class: "log" }, JSON.stringify({ mcpServers: { moelab: { command: "npx", args: ["-y", "mcp-remote", url] } } }, null, 2)));
}
$("#folders-save").onclick = async () => { await api("POST", "/api/tools/folders", { dossiers: $("#folders").value.split("\n") }); toast("Dossiers enregistrés."); loadTools(); };
$("#mcp-add").onclick = () => { mcpRows.push({ nom: "", commande: "", args: [], actif: true }); renderMcp(); };
$("#mcp-save").onclick = async () => {
  $("#mcp-msg").textContent = "Connexion…";
  try { const r = await api("POST", "/api/tools/mcp", { serveurs: mcpRows });
    $("#mcp-msg").textContent = r.ok ? "Connecté." : "Erreurs : " + Object.entries(r.erreurs).map(([k, v]) => `${k} : ${v}`).join(" ; ");
    loadTools(); } catch (e) { $("#mcp-msg").textContent = e.message; }
};
$("#expose").onchange = async (e) => { await api("POST", "/api/tools/expose", { actif: e.target.checked }); renderExpose(e.target.checked, true); };

// ------------------------------------------------------------------ moteur
function gauge(cls, title, big, sub) { return el("div", { class: "gauge " + cls }, el("div", { class: "sub" }, title), el("div", { class: "big" }, big), el("div", { class: "sub" }, sub)); }
function renderEngineTop() {
  const st = STATUS; const box = $("#engine-top");
  if (!st || !st.model) { box.replaceChildren(el("p", { class: "empty" }, "Aucun modèle chargé. Va dans Mise en route.")); return; }
  const c = st.caches, s = st.stats, last = st.last;
  box.replaceChildren(
    gauge("vram", "Experts en VRAM", `${n2(c.vram.utilise_gio)} / ${n2(c.vram.budget_gio)} Gio`, `${c.vram.experts} experts · trouvés en VRAM : ${pct(s.hit_vram)}`),
    gauge("ram", "Experts en RAM", `${n2(c.ram.utilise_gio)} / ${n2(c.ram.budget_gio)} Gio`, `${c.ram.experts} experts` + (c.pool_hors_pool ? ` · ${c.pool_hors_pool} tampons hors pool` : "")),
    gauge("disk", "Lectures disque", `${n1(s.disque_par_token)} / token`, `store : ${st.store.couches} couches × ${st.store.experts} experts, ${n1(st.store.total_gio)} Gio`),
    gauge("", "Dernière réponse", last ? `${n2(last.tok_s)} tok/s` : "–", last ? `1er token ${n2(last.premier_token_s)} s · ${last.tokens} tokens` : "pas encore de réponse"),
    ...(st.gpu ? [gauge("", st.gpu.nom, `${n2(st.gpu.vram_libre_gio)} Gio libres`, `sur ${n1(st.gpu.vram_totale_gio)} Go de VRAM`)] : []));
}
const MODES = { live: "à chaud", caches: "à chaud, caches vidés", reload: "rechargement", store: "autre store" };
async function loadEngine() {
  renderEngineTop();
  const o = await api("GET", "/api/options");
  const box = $("#options"); box.replaceChildren();
  for (const [g, label] of Object.entries(o.groupes)) {
    const opts = o.options.filter((x) => x.group === g);
    if (!opts.length) continue;
    const grp = el("div", { class: "opt-group" }, el("h2", {}, label));
    for (const x of opts) {
      const cur = o.actuel[x.key];
      let input;
      const apply = async (v) => {
        try { const r = await api("POST", "/api/options", { cle: x.key, valeur: v }); toast(r.message); }
        catch (e) { toast(e.message, 6000); }
      };
      if (x.kind === "bool") input = el("input", { type: "checkbox", checked: !!cur, disabled: !o.actuel || x.apply === "reload" || x.apply === "store", onchange: (e) => apply(e.target.checked ? "on" : "off") });
      else if (x.kind === "choice") input = el("select", { disabled: x.apply === "reload" || x.apply === "store", onchange: (e) => apply(e.target.value) },
        x.choices.map((c) => el("option", { value: c, selected: String(c) === String(cur) }, String(c))));
      else {
        const f = el("input", { value: cur === null || cur === undefined ? "" : (typeof cur === "object" ? JSON.stringify(cur) : cur), placeholder: x.default === null ? "auto" : "", size: 8, disabled: x.apply === "reload" || x.apply === "store" });
        input = el("span", { class: "apply" }, f, el("button", { disabled: f.disabled, onclick: () => apply(f.value.trim() === "" ? "auto" : f.value.trim()) }, "Appliquer"));
      }
      grp.append(el("div", { class: "opt" },
        el("div", {}, el("div", { class: "name" }, x.label), el("div", { class: "mode" }, `${x.key} · ${MODES[x.apply] || x.apply}`)),
        el("div", {}, input),
        el("div", {}, el("div", { class: "help" }, x.help), x.measured ? el("div", { class: "measured" }, "Mesuré : " + x.measured) : null)));
    }
    box.append(grp);
  }
}
loaders.engine = loadEngine;

// ------------------------------------------------------------------ tâches
async function loadJobs() {
  const jobs = await api("GET", "/api/jobs");
  const ul = $("#jobs"); ul.replaceChildren();
  if (!jobs.length) ul.append(el("li", {}, "Rien dans la file."));
  for (const j of jobs.slice(0, 40)) {
    ul.append(el("li", {},
      el("div", {}, el("b", {}, j.title), el("div", { class: "st " + j.status.replace(" ", "-") }, `${j.kind} · ${j.status}` + (j.error ? ` : ${j.error}` : ""))),
      ["en attente", "en cours"].includes(j.status) ? el("button", { class: "danger", onclick: async () => { await api("POST", `/api/jobs/${j.id}/cancel`); loadJobs(); } }, "Annuler") : el("span"),
      j.progress !== null && j.progress !== undefined ? el("div", { class: "progress", style: "grid-column: 1 / -1" }, el("i", { style: `width:${Math.round(j.progress * 100)}%` })) : null));
  }
}
loaders.tasks = loadJobs;
setInterval(() => show.current === "tasks" && loadJobs(), 2500);
$("#batch").onsubmit = async (e) => {
  e.preventDefault();
  try {
    await api("POST", "/api/tasks/batch", { dossier: $("#b-folder").value, motif: $("#b-pattern").value, instruction: $("#b-instr").value, sortie: $("#b-out").value });
    toast("Tâche ajoutée à la file."); loadJobs();
  } catch (err) { toast(err.message); }
};

// ------------------------------------------------------------------ démarrage
pollStatus().then(() => show(location.hash.slice(1) || "chat"));
