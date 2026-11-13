"use strict";

// Test console for the SOP harness. Everything shown in the inspector comes from the
// server's state snapshot; the page holds no workflow logic of its own.

const SCENARIOS = [
  ["Demo case", "I'm the policyholder. My name is Margaret Chen, policy POL-9921. I'm calling about my denied healthcare claim from January. DOB is 1985-03-15, SSN last four is 4472."],
  ["Reason first, name only", "Hi, this is Margaret Chen. I'm calling about my denied healthcare claim from January."],
  ["Finish verifying", "My date of birth is 1985-03-15 and the last four of my SSN are 4472."],
  ["Frustrated", "I already told you who I am. This is ridiculous. Just tell me why my claim was denied."],
  ["Won't give SSN", "I'm not comfortable giving out my SSN. Why do you need all this?"],
  ["Off-topic", "What is RL?"],
  ["Representative", "I'm David Chen, calling on behalf of my mother Margaret Chen about her denied claim. Her date of birth is 1985-03-15, her phone is 650-521-2836 and her email is margaret@email.com."],
  ["Check consent", "She says she has approved it."],
  ["How to submit", "How do I submit the documents?"],
  ["Can't get a document", "What if I can't get the pathology report?"],
  ["Different claim", "What about my auto claim?"],
  ["Ask for a human", "I'd like to speak to a human representative."],
  ["Wrap up", "That's all, thanks."],
  ["Yes to email", "Yes please."],
  ["Skip email", "No, skip the email."],
];

const AUTONOMY_NOTE = {
  scripted: "scripted: harness decides, model phrases",
  guided: "guided: model picks a bounded action",
  open: "open: model reasons over grounded data",
};

const store = {
  get(key) { try { return sessionStorage.getItem(key); } catch { return null; } },
  set(key, value) { try { value ? sessionStorage.setItem(key, value) : sessionStorage.removeItem(key); } catch { /* private mode */ } },
};

const app = {
  config: null,
  sessionId: null,
  brain: null,
  apiKey: store.get("sop.apiKey") || "",
  consentScenario: store.get("sop.consent") || "default",
  busy: false,
  state: null,
};

const $ = (id) => document.getElementById(id);

function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of [].concat(children)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return node;
}

function badge(text, tone = "") {
  return el("span", { class: `badge ${tone}`.trim(), text });
}

function row(key, value) {
  return el("div", { class: "row" }, [el("span", { class: "k", text: key }), el("span", { class: "v" }, value)]);
}

async function api(path, options = {}) {
  const headers = { "Content-Type": "application/json" };
  if (app.apiKey) headers["X-API-Key"] = app.apiKey;
  const response = await fetch(path, { ...options, headers });
  let body = null;
  try { body = await response.json(); } catch { /* non-JSON error page */ }
  if (!response.ok) {
    const error = new Error(body?.error?.message || `Request failed (${response.status})`);
    error.code = body?.error?.code || "http_error";
    throw error;
  }
  return body;
}

// ---- conversation ----------------------------------------------------------

function addMessage(role, text, tag) {
  const node = el("div", { class: `msg ${role}` }, [
    el("div", { class: "bubble", text }),
    tag ? el("div", { class: "tag", text: tag }) : null,
  ]);
  $("messages").append(node);
  $("messages").scrollTop = $("messages").scrollHeight;
  return node;
}

function showTyping() {
  const dots = el("span", { class: "typing", "aria-label": "The agent is replying" }, [el("i"), el("i"), el("i")]);
  const node = el("div", { class: "msg agent" }, [el("div", { class: "bubble" }, dots)]);
  $("messages").append(node);
  $("messages").scrollTop = $("messages").scrollHeight;
  return node;
}

function setBanner(text, tone = "") {
  const banner = $("banner");
  banner.hidden = !text;
  banner.className = `banner ${tone}`.trim();
  banner.textContent = text || "";
}

function setBusy(busy) {
  app.busy = busy;
  const ended = Boolean(app.state?.ended);
  $("send").disabled = busy || ended;
  $("input").disabled = ended;
}

async function startSession() {
  // Drop the old conversation first: until the new one exists there is nothing to send a message to.
  app.sessionId = null;
  app.state = null;
  setBusy(true);
  setBanner("");
  $("messages").replaceChildren();
  try {
    const data = await api("/api/sessions", {
      method: "POST",
      body: JSON.stringify({ brain: app.brain, consent_scenario: app.consentScenario }),
    });
    app.sessionId = data.session_id;
    app.brain = data.brain;
    app.state = data.state;
    addMessage("agent", data.reply, "VERIFY_ID");
    render();
    $("input").focus();
  } catch (error) {
    setBanner(error.message, "bad");
  } finally {
    setBusy(false);
  }
}

async function send(text) {
  text = text.trim();
  if (!text || app.busy || !app.sessionId || app.state?.ended) return;
  addMessage("user", text);
  $("input").value = "";
  setBusy(true);
  setBanner("");
  const typing = showTyping();
  try {
    const data = await api(`/api/sessions/${app.sessionId}/messages`, { method: "POST", body: JSON.stringify({ text }) });
    typing.remove();
    app.state = data.state;
    addMessage("agent", data.reply, data.state.phase);
    render();
    const lastTrace = data.state.traces[data.state.traces.length - 1];
    const failure = lastTrace?.events.find((event) => event.kind === "error");
    if (failure) setBanner(`The model call had a problem, so the harness answered from its own script: ${failure.detail.error || failure.title}`);
    if (data.state.ended) {
      addMessage("system", data.state.phase === "ESCALATED"
        ? "Handed to a human representative. Start a new conversation to try another scenario."
        : "Conversation closed. Start a new conversation to try another scenario.");
    }
  } catch (error) {
    typing.remove();
    setBanner(error.message, "bad");
    if (error.code === "missing_api_key" || error.code === "model_setup") openSettings(error.message);
    if (error.code === "unknown_session") addMessage("system", "This conversation expired on the server. Start a new one.");
  } finally {
    setBusy(false);
    $("input").focus();
  }
}

// ---- inspector --------------------------------------------------------------

function render() {
  const s = app.state;
  if (!s) return;
  renderBadge();
  renderStepper(s);
  renderIdentity(s);
  renderMemory(s);
  renderCase(s);
  renderWrapup(s);
  renderTrace(s);
}

function renderBadge() {
  const node = $("brain-badge");
  if (app.brain === "stub") {
    node.textContent = "Offline stub (no model)";
    node.className = "badge warn";
  } else {
    const reader = app.config.nlu_model !== app.config.model ? ` writes · ${app.config.nlu_model} reads` : "";
    node.textContent = `Claude · ${app.config.model}${reader}`;
    node.className = "badge accent";
  }
}

function renderStepper(s) {
  const order = s.phases.map((p) => p.phase);
  const index = order.indexOf(s.phase);
  const finished = s.phase === "CLOSED";
  $("stepper").replaceChildren(...s.phases.map((p, i) => {
    const state = finished || (index >= 0 && i < index) ? "done" : i === index ? "current" : "";
    return el("li", { class: `step ${state}`.trim(), "aria-current": state === "current" ? "step" : null, title: AUTONOMY_NOTE[p.autonomy] }, [
      el("div", { class: "name", text: p.phase }),
      el("div", { class: "meta", text: p.autonomy }),
    ]);
  }));
  if (s.phase === "ESCALATED") setBanner(`Handed to a human: ${s.handoff?.reason_text || ""}`);
  if (s.phase === "CLOSED") setBanner("Workflow complete: verified, resolved, processed and wrapped up.", "ok");
}

function renderIdentity(s) {
  const v = s.verification;
  const matched = v.matched.length;
  const chips = Object.entries(v.labels).map(([field, label]) => {
    const short = label.replace(" on file", "").replace("last four digits of the SSN or national ID", "ID last four");
    if (v.matched.includes(field)) return badge(`✓ ${short}`, "ok");
    if (v.mismatched.includes(field)) return badge(`✗ ${short}`, "bad");
    return badge(short);
  });
  const [fails, maxFails] = s.counters.failed_attempts;
  const [stalls, maxStalls] = s.counters.stalls;
  const nodes = [
    row("Status", s.verified ? badge(`verified${s.caller_first_name ? ` · ${s.caller_first_name}` : ""}`, "ok") : badge("not verified", "warn")),
    el("div", { class: `meter ${matched >= v.required ? "full" : ""}`.trim(), role: "img", "aria-label": `${matched} of ${v.required} required fields matched` },
      el("span", { style: `width:${Math.min(100, (matched / v.required) * 100)}%` })),
    el("div", { class: "fields" }, chips),
    row("Matched", `${matched} of ${v.required} required`),
    row("Non-matching attempts", el("span", {}, badge(`${fails} / ${maxFails}`, fails ? "warn" : ""))),
    row("Refusals without progress", el("span", {}, badge(`${stalls} / ${maxStalls}`, stalls ? "warn" : ""))),
  ];
  if (s.caller_role === "representative") {
    const tone = { approved: "ok", pending: "warn", timed_out: "bad" }[s.consent.status] || "";
    nodes.push(row("Caller", "representative"));
    nodes.push(row("Policyholder consent", el("span", {}, [badge(s.consent.status.replace("_", " "), tone), ` (${s.consent.scenario} scenario)`])));
  }
  $("identity").replaceChildren(...nodes);
}

function renderMemory(s) {
  const items = [...s.memory.slots, ...s.memory.notes];
  if (!items.length) {
    $("memory").replaceChildren(el("div", { class: "empty", text: "Nothing remembered yet. Anything useful the caller says is stored here, whichever phase the call is in." }));
    return;
  }
  $("memory").replaceChildren(...items.map((item) => el("div", { class: "mem" }, [
    el("div", { class: "key", text: item.key }),
    el("div", { class: "val", text: String(item.value) }),
    el("div", { class: "src" }, [
      badge(`heard in ${item.phase}`, item.phase === "VERIFY_ID" && !item.key.startsWith("identity") ? "accent" : ""),
      `turn ${item.turn}`,
      item.used ? badge("used", "ok") : null,
    ]),
  ])));
}

function renderCase(s) {
  const [strikes, maxStrikes] = s.counters.out_of_scope_strikes;
  const [heat, maxHeat] = s.counters.heated_turns;
  const intentLabel = s.selected_intent ? app.config.sop.intents[s.selected_intent] : null;
  $("case").replaceChildren(
    row("Claim in focus", s.selected_case_id ? badge(s.selected_case_id, "accent") : el("span", { class: "muted", text: "none yet" })),
    row("Workflow path", intentLabel || el("span", { class: "muted", text: "none yet" })),
    row("Off-topic in a row", el("span", {}, badge(`${strikes} / ${maxStrikes}`, strikes ? "warn" : ""))),
    row("Heated turns in a row", el("span", {}, badge(`${heat} / ${maxHeat}`, heat ? "warn" : ""))),
  );
}

function renderWrapup(s) {
  const tone = { sent: "ok", skipped: "", offered: "warn", unavailable: "bad" }[s.email.status] || "";
  const nodes = [row("Email summary", badge(s.email.status.replace("_", " "), tone))];
  for (const mail of s.outbox) {
    nodes.push(el("details", { class: "email", open: true }, [
      el("summary", {}, [`To ${mail.to}: ${mail.subject} `, badge(`written by ${mail.drafted_by}`)]),
      el("pre", { text: mail.body }),
    ]));
  }
  if (s.handoff) {
    nodes.push(row("Handoff", badge(s.handoff.reason.replaceAll("_", " "), "warn")));
    nodes.push(el("details", { class: "email", open: true }, [
      el("summary", { text: "Ticket passed to the human representative" }),
      el("pre", { text: [s.handoff.reason_text, s.handoff.detail, ...s.handoff.summary].filter(Boolean).join("\n") }),
    ]));
  }
  $("wrapup").replaceChildren(...nodes);
}

function renderTrace(s) {
  if (!s.traces.length) {
    $("trace").replaceChildren(el("div", { class: "empty", text: "Each turn lists what was understood, remembered, gated, called and checked." }));
    return;
  }
  const turns = s.traces.map((trace, i) => {
    const moved = trace.phase_before !== trace.phase_after;
    const warnings = trace.events.filter((event) => !event.ok).length;
    return el("details", { class: "turn", open: i === s.traces.length - 1 }, [
      el("summary", {}, [
        `Turn ${trace.turn}`,
        trace.seconds ? el("span", { class: "phase", text: `${trace.seconds}s` }) : null,
        el("span", { class: "phase", text: moved ? `${trace.phase_before} → ${trace.phase_after}` : trace.phase_before }),
        warnings ? badge(`${warnings} flagged`, "warn") : null,
      ]),
      el("ul", { class: "events" }, trace.events.map((event) => {
        const hasDetail = event.detail && Object.keys(event.detail).length > 0;
        const head = [el("span", { class: "kind", text: event.kind }), event.title];
        const cls = `event kind-${event.kind} ${event.ok ? "" : "no"}`.trim();
        if (!hasDetail) return el("li", { class: cls }, head);
        return el("li", { class: cls }, el("details", {}, [el("summary", {}, head), el("pre", { text: JSON.stringify(event.detail, null, 2) })]));
      })),
    ]);
  });
  $("trace").replaceChildren(...turns.reverse());
}

// ---- settings and test data --------------------------------------------------

function openSettings(note) {
  const form = $("settings-form");
  form.elements.brain.value = app.brain || app.config.default_brain;
  $("api-key").value = app.apiKey;
  $("model-name").textContent = `${app.config.model}${app.config.nlu_model !== app.config.model ? `, reader ${app.config.nlu_model}` : ""}`;
  $("key-hint").textContent = app.config.server_key_configured ? "optional: the server already has one" : "required for Claude: the server has none";
  $("consent-scenario").replaceChildren(...app.config.consent_scenarios.map((name) => el("option", { value: name, text: name, selected: name === app.consentScenario })));
  $("settings-note").hidden = !note;
  $("settings-note").textContent = note || "";
  if (!$("dlg-settings").open) $("dlg-settings").showModal();
}

function table(rows, columns) {
  return el("div", { class: "table-wrap" }, el("table", {}, [
    el("thead", {}, el("tr", {}, columns.map((c) => el("th", { text: c })))),
    el("tbody", {}, rows.map((r) => el("tr", {}, columns.map((c) => el("td", { text: Array.isArray(r[c]) ? r[c].join(", ") : r[c] ?? "" }))))),
  ]));
}

async function openData() {
  $("dlg-data").showModal();
  try {
    const data = await api("/api/fixtures");
    $("data-body").replaceChildren(
      el("h3", { text: "Policyholders" }),
      table(data.policyholders, ["party_id", "name", "policy_number", "dob", "id_last4", "phone", "email"]),
      el("h3", { text: "Claims" }),
      table(data.claims, ["case_id", "party_id", "case_type", "created_at", "status", "summary", "documents_needed", "appeal_deadline"]),
      el("h3", { text: "Authorised representatives" }),
      table(data.representatives, ["rep_name", "relationship", "buyer_name", "buyer_party_id"]),
      el("p", { class: "muted", text: `The demo clock is fixed at ${app.config.demo_today}, so "from January" and the appeal deadlines line up with the sample data.` }),
    );
  } catch (error) {
    $("data-body").replaceChildren(el("p", { class: "muted", text: error.message }));
  }
}

// ---- wiring -------------------------------------------------------------------

async function init() {
  $("scenarios").replaceChildren(...SCENARIOS.map(([label, text]) => el("button", {
    type: "button",
    text: label,
    title: text,
    onclick: () => { $("input").value = text; $("input").focus(); },
  })));

  $("composer").addEventListener("submit", (event) => { event.preventDefault(); send($("input").value); });
  $("input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); send($("input").value); }
  });
  $("btn-new").addEventListener("click", startSession);
  $("btn-settings").addEventListener("click", () => openSettings());
  $("btn-data").addEventListener("click", openData);
  $("data-close").addEventListener("click", () => $("dlg-data").close());
  $("settings-cancel").addEventListener("click", () => $("dlg-settings").close());
  $("settings-form").addEventListener("submit", () => {
    app.brain = $("settings-form").elements.brain.value;
    app.apiKey = $("api-key").value.trim();
    app.consentScenario = $("consent-scenario").value;
    store.set("sop.apiKey", app.apiKey);
    store.set("sop.brain", app.brain);
    store.set("sop.consent", app.consentScenario);
    startSession();
  });

  try {
    app.config = await api("/api/config");
  } catch (error) {
    setBanner(`Could not reach the server: ${error.message}`, "bad");
    return;
  }
  app.brain = store.get("sop.brain") || app.config.default_brain;
  await startSession();
  if (app.brain === "claude" && !app.config.server_key_configured && !app.apiKey) {
    openSettings("No API key is set on the server. Enter an Anthropic API key to talk to Claude, or switch to the offline stub.");
  }
}

init();
