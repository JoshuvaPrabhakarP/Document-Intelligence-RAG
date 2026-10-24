"use strict";

/* Document Intelligence — frontend logic.
 * Vanilla JS, no framework. All dynamic content is inserted via
 * textContent/createElement (never raw innerHTML of untrusted data) so
 * nothing from a document's extracted text or a model's answer can ever
 * execute as HTML. */

const UPLOAD_SAMPLE_QUESTIONS = [
  "What is the total amount?",
  "Who is the vendor?",
  "What date is on this receipt?",
  "What payment method was used?",
  "What items were purchased?",
];

const MAX_FILE_MB = 10;
const MAX_PDF_PAGES = 20;

const state = {
  mode: "demo",
  workflow: "standard",
  uploading: false,
  asking: false,
  lastResult: { demo: null, upload: null },
};

const WORKFLOW_HINTS = {
  standard: "Retrieves once, then generates a grounded answer.",
  agent: "Lets the model search/fetch sources itself, assess the evidence, and ask for clarification if your question is ambiguous.",
};

const el = {
  tabDemo: document.getElementById("tab-demo"),
  tabUpload: document.getElementById("tab-upload"),
  panelDemo: document.getElementById("panel-demo"),
  panelUpload: document.getElementById("panel-upload"),
  wfStandard: document.getElementById("wf-standard"),
  wfAgent: document.getElementById("wf-agent"),
  workflowHint: document.getElementById("workflow-hint"),
  demoChips: document.getElementById("demo-chips"),
  uploadChips: document.getElementById("upload-chips"),
  dropzone: document.getElementById("dropzone"),
  fileInput: document.getElementById("file-input"),
  uploadStatus: document.getElementById("upload-status"),
  fileList: document.getElementById("file-list"),
  clearBtn: document.getElementById("clear-btn"),
  question: document.getElementById("question"),
  askBtn: document.getElementById("ask-btn"),
  askBtnLabel: document.getElementById("ask-btn-label"),
  result: document.getElementById("result"),
};

// ---------- small safe-DOM helpers ----------

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function text(tag, content, className) {
  const n = document.createElement(tag);
  if (className) n.className = className;
  n.textContent = content;
  return n;
}

// Fixed, hand-authored icon set — every SVG path here is static markup we
// wrote, never derived from a document's extracted text or a model's
// answer, so building these via innerHTML carries none of the risk that
// rule is meant to prevent (that risk is specifically about untrusted
// content, which icon names selected from a fixed list below are not).
const ICON_PATHS = {
  check: '<path d="M4 10l4 4 8-8" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>',
  warn: '<path d="M10 3l8 14H2L10 3z" stroke="currentColor" stroke-width="1.6" fill="none" stroke-linejoin="round"/><circle cx="10" cy="14" r="0.9" fill="currentColor"/><path d="M10 8v3.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>',
  error: '<circle cx="10" cy="10" r="8" stroke="currentColor" stroke-width="1.6" fill="none"/><path d="M7 7l6 6M13 7l-6 6" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>',
  dollar: '<path d="M10 2v16M13.5 5.5c0-1.4-1.6-2.5-3.5-2.5S6.5 4.1 6.5 5.5 8 7.5 10 8s3.5 1.3 3.5 3-1.6 2.5-3.5 2.5-3.5-1-3.5-2.5" stroke="currentColor" stroke-width="1.5" fill="none" stroke-linecap="round" stroke-linejoin="round"/>',
  store: '<path d="M3 8l1-4h12l1 4M3 8v8a1 1 0 001 1h3v-5h6v5h3a1 1 0 001-1V8M3 8h14" stroke="currentColor" stroke-width="1.5" fill="none" stroke-linejoin="round"/>',
  calendar: '<rect x="3" y="4" width="14" height="13" rx="1.5" stroke="currentColor" stroke-width="1.5" fill="none"/><path d="M3 8h14M7 2v4M13 2v4" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>',
  card: '<rect x="2" y="5" width="16" height="11" rx="1.5" stroke="currentColor" stroke-width="1.5" fill="none"/><path d="M2 9h16" stroke="currentColor" stroke-width="1.5"/>',
  list: '<path d="M7 5h10M7 10h10M7 15h10M3.5 5h.01M3.5 10h.01M3.5 15h.01" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>',
  question: '<circle cx="10" cy="10" r="8" stroke="currentColor" stroke-width="1.5" fill="none"/><path d="M7.8 7.8a2.2 2.2 0 014.1 1.1c0 1.5-2.1 1.5-2.1 3.2" stroke="currentColor" stroke-width="1.5" fill="none" stroke-linecap="round"/><circle cx="9.8" cy="14.3" r="0.9" fill="currentColor"/>',
  search: '<circle cx="9" cy="9" r="6.5" stroke="currentColor" stroke-width="1.6" fill="none"/><path d="M18 18l-4.3-4.3" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>',
};

function icon(name, size) {
  const span = document.createElement("span");
  span.setAttribute("aria-hidden", "true");
  span.style.display = "inline-flex";
  const s = size || 16;
  span.innerHTML = `<svg width="${s}" height="${s}" viewBox="0 0 20 20">${ICON_PATHS[name] || ICON_PATHS.question}</svg>`;
  return span;
}

// Picks a representative icon for a suggested-question chip based on its
// wording — purely a visual cue, never changes what question gets asked
// or how it's answered.
function chipIconFor(question) {
  const q = question.toLowerCase();
  if (/(total|amount|cost|price|tax)/.test(q)) return "dollar";
  if (/(vendor|store|merchant|business|who)/.test(q)) return "store";
  if (/(date|when)/.test(q)) return "calendar";
  if (/(payment|paid|card)/.test(q)) return "card";
  if (/(item|purchase|line|bought)/.test(q)) return "list";
  return "question";
}

// ---------- tabs ----------

function switchMode(mode) {
  state.mode = mode;
  el.tabDemo.classList.toggle("active", mode === "demo");
  el.tabUpload.classList.toggle("active", mode === "upload");
  el.tabDemo.setAttribute("aria-selected", String(mode === "demo"));
  el.tabUpload.setAttribute("aria-selected", String(mode === "upload"));
  el.panelDemo.hidden = mode !== "demo";
  el.panelUpload.hidden = mode !== "upload";

  el.question.placeholder = mode === "upload"
    ? "Ask about vendor, date, total, tax, or line items on your uploaded document…"
    : "Ask a question about the sample receipts…";

  // Preserve each mode's last answer when switching tabs, rather than
  // wiping it — restore whatever this mode last produced (or nothing).
  renderResult(state.lastResult[mode]);

  if (mode === "upload") refreshSessionFiles();
}

el.tabDemo.addEventListener("click", () => switchMode("demo"));
el.tabUpload.addEventListener("click", () => switchMode("upload"));

// ---------- workflow toggle (Standard RAG / Agent workflow) ----------

function switchWorkflow(workflow) {
  state.workflow = workflow;
  el.wfStandard.classList.toggle("active", workflow === "standard");
  el.wfAgent.classList.toggle("active", workflow === "agent");
  el.wfStandard.setAttribute("aria-checked", String(workflow === "standard"));
  el.wfAgent.setAttribute("aria-checked", String(workflow === "agent"));
  el.workflowHint.textContent = WORKFLOW_HINTS[workflow];
}

el.wfStandard.addEventListener("click", () => switchWorkflow("standard"));
el.wfAgent.addEventListener("click", () => switchWorkflow("agent"));

// ---------- chips ----------

async function loadDemoChips() {
  try {
    const res = await fetch("/sample-questions");
    const data = await res.json();
    clear(el.demoChips);
    data.questions.forEach((q) => el.demoChips.appendChild(makeChip(q)));
  } catch {
    // Non-fatal — the question box still works without suggestions.
  }
}

function loadUploadChips() {
  clear(el.uploadChips);
  UPLOAD_SAMPLE_QUESTIONS.forEach((q) => el.uploadChips.appendChild(makeChip(q)));
}

function makeChip(question) {
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "chip";
  const iconSpan = icon(chipIconFor(question), 14);
  iconSpan.classList.add("chip-icon");
  btn.appendChild(iconSpan);
  btn.appendChild(document.createTextNode(question)); // text node — never innerHTML
  btn.addEventListener("click", () => {
    el.question.value = question;
    ask();
  });
  return btn;
}

// ---------- upload: dropzone ----------

el.dropzone.addEventListener("click", () => el.fileInput.click());
el.dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    el.fileInput.click();
  }
});
["dragenter", "dragover"].forEach((evt) =>
  el.dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    el.dropzone.classList.add("drag-over");
  })
);
["dragleave", "drop"].forEach((evt) =>
  el.dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    el.dropzone.classList.remove("drag-over");
  })
);
el.dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  if (file) uploadFile(file);
});
el.fileInput.addEventListener("change", () => {
  const file = el.fileInput.files[0];
  if (file) uploadFile(file);
  el.fileInput.value = "";
});

function showUploadStatus(message, kind) {
  el.uploadStatus.className = "status-banner" + (kind ? " " + kind : "");
  clear(el.uploadStatus);
  if (kind === undefined) {
    const spinner = document.createElement("span");
    spinner.className = "spinner";
    el.uploadStatus.appendChild(spinner);
  }
  el.uploadStatus.appendChild(text("span", message));
}

function hideUploadStatus() {
  el.uploadStatus.className = "status-banner hidden";
}

const ALLOWED_TYPES = ["image/jpeg", "image/png", "application/pdf"];

async function uploadFile(file) {
  if (state.uploading) return; // prevent duplicate/overlapping submissions
  if (!ALLOWED_TYPES.includes(file.type)) {
    showUploadStatus(
      `"${file.name}" is a ${file.type || "file type"} that isn't supported. Upload a JPG, PNG, or PDF.`,
      "error"
    );
    return;
  }
  if (file.size > MAX_FILE_MB * 1024 * 1024) {
    showUploadStatus(
      `"${file.name}" is ${(file.size / 1e6).toFixed(1)}MB, which exceeds the ${MAX_FILE_MB}MB limit.`,
      "error"
    );
    return;
  }

  state.uploading = true;
  el.dropzone.setAttribute("aria-disabled", "true");
  // Honest, single-stage status — this is genuinely one blocking request
  // server-side (upload -> extract/OCR -> index), so it's described as
  // one step rather than invented sub-percentages we can't actually observe.
  showUploadStatus(`Processing "${file.name}" — extracting text (with OCR if needed) and indexing…`);

  try {
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch("/upload", { method: "POST", body: formData, credentials: "same-origin" });
    const data = await res.json();

    if (!res.ok) {
      showUploadStatus(data.detail || `Upload failed (HTTP ${res.status}).`, "error");
      return;
    }

    showUploadStatus(
      `Processed "${data.original_filename}" — ${data.page_count} page${data.page_count === 1 ? "" : "s"} indexed.`,
      "success"
    );
    await refreshSessionFiles();
  } catch (e) {
    showUploadStatus(`Network error: ${e.message}. Is the server running?`, "error");
  } finally {
    state.uploading = false;
    el.dropzone.removeAttribute("aria-disabled");
  }
}

// ---------- session / document cards ----------

async function refreshSessionFiles() {
  let data;
  try {
    const res = await fetch("/session", { credentials: "same-origin" });
    data = await res.json();
  } catch {
    return;
  }

  clear(el.fileList);
  if (!data.files.length) {
    el.fileList.appendChild(text("p", "No documents yet — drag a receipt above or click to browse.", "empty-state"));
    return;
  }

  data.files.forEach((f) => el.fileList.appendChild(buildFileCard(f)));
}

function buildFileCard(f) {
  const card = document.createElement("div");
  card.className = "file-card";

  const isImage = /\.(jpe?g|png)$/i.test(f.original_filename);
  const badge = document.createElement("div");
  badge.className = "thumb-icon " + (isImage ? "img" : "pdf");
  badge.textContent = isImage ? "IMG" : "PDF";
  card.appendChild(badge);

  const meta = document.createElement("div");
  meta.className = "meta";
  meta.appendChild(text("div", f.original_filename, "filename"));
  meta.appendChild(
    text("div", `${f.page_count} page${f.page_count === 1 ? "" : "s"} · ${f.extracted_char_count} characters extracted`, "details")
  );
  if (f.warning) {
    meta.appendChild(text("div", f.warning, "file-warning"));
  }
  card.appendChild(meta);

  return card;
}

el.clearBtn.addEventListener("click", async () => {
  if (!window.confirm("Clear all uploaded documents from this session? This can't be undone.")) return;
  el.clearBtn.disabled = true;
  try {
    await fetch("/session", { method: "DELETE", credentials: "same-origin" });
  } finally {
    el.clearBtn.disabled = false;
  }
  hideUploadStatus();
  state.lastResult.upload = null; // citations referenced now-deleted documents
  if (state.mode === "upload") renderResult(null);
  await refreshSessionFiles();
});

// ---------- asking questions ----------

el.askBtn.addEventListener("click", ask);
el.question.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) ask();
});

async function ask() {
  const question = el.question.value.trim();
  if (!question || state.asking) return; // prevent duplicate submissions

  state.asking = true;
  el.askBtn.disabled = true;
  el.askBtnLabel.textContent = "Asking…";
  renderWorking();

  try {
    const res = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "same-origin",
      body: JSON.stringify({ question, top_k: 5, mode: state.mode, workflow: state.workflow }),
    });
    const data = await res.json();

    if (!res.ok) {
      const errResult = { kind: "error", message: data.detail || `Request failed (HTTP ${res.status}).` };
      state.lastResult[state.mode] = errResult;
      renderResult(errResult);
      return;
    }
    if (data.error) {
      const errResult = { kind: "error", message: data.error, sources: data.sources };
      state.lastResult[state.mode] = errResult;
      renderResult(errResult);
      return;
    }

    const successResult = { kind: "answer", data };
    state.lastResult[state.mode] = successResult;
    renderResult(successResult);
  } catch (e) {
    const errResult = { kind: "error", message: `Network error: ${e.message}. Is the server running?` };
    state.lastResult[state.mode] = errResult;
    renderResult(errResult);
  } finally {
    state.asking = false;
    el.askBtn.disabled = false;
    el.askBtnLabel.textContent = "Ask document";
  }
}

function renderWorking() {
  clear(el.result);
  const banner = document.createElement("div");
  banner.className = "status-banner";
  banner.appendChild((() => { const s = document.createElement("span"); s.className = "spinner"; return s; })());
  banner.appendChild(text("span", "Retrieving relevant passages and generating an answer…"));
  el.result.appendChild(banner);
}

function renderEmptyState() {
  const wrap = document.createElement("div");
  wrap.className = "card empty-result";
  const svg = document.createElement("div");
  svg.setAttribute("aria-hidden", "true");
  svg.innerHTML = `<svg width="48" height="48" viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="1.4">
    <path d="M14 6h14l8 8v26a2 2 0 01-2 2H14a2 2 0 01-2-2V8a2 2 0 012-2z" stroke-linejoin="round"/>
    <path d="M28 6v8h8" stroke-linejoin="round"/>
    <path d="M17 24h14M17 29h14M17 19h7" stroke-linecap="round"/>
    <circle cx="34" cy="34" r="7" fill="white" stroke-width="1.4"/>
    <path d="M38.5 38.5L42 42" stroke-linecap="round"/>
  </svg>`;
  wrap.appendChild(svg);
  wrap.appendChild(text("p", "No answer yet", "empty-title"));
  wrap.appendChild(text("p", "Pick a suggested question or type your own — the answer will appear here with its supporting source."));
  el.result.appendChild(wrap);
}

function renderResult(result) {
  clear(el.result);
  if (!result) {
    renderEmptyState();
    return;
  }

  if (result.kind === "error") {
    const card = document.createElement("div");
    card.className = "result-card error";
    const status = document.createElement("div");
    status.className = "result-status";
    status.appendChild(icon("error"));
    status.appendChild(document.createTextNode("Error"));
    card.appendChild(status);
    card.appendChild(text("div", result.message, "result-answer"));
    el.result.appendChild(card);
    if (result.sources && result.sources.length) {
      el.result.appendChild(buildSourcesSection(result.sources));
    }
    return;
  }

  const data = result.data;

  if (data.route === "clarify") {
    const card = document.createElement("div");
    card.className = "result-card warn";
    const status = document.createElement("div");
    status.className = "result-status";
    status.appendChild(icon("question"));
    status.appendChild(document.createTextNode("Needs clarification"));
    card.appendChild(status);
    card.appendChild(text("div", data.clarification_question || "Could you clarify your question?", "result-answer"));
    card.appendChild(text("div", "Type your answer in the question box above and press Ask again.", "clarify-hint"));
    el.result.appendChild(card);
    if (data.execution_trace) el.result.appendChild(buildExecutionDetails(data));
    return;
  }

  const sufficient = data.has_sufficient_evidence !== false;
  const card = document.createElement("div");
  card.className = "result-card " + (sufficient ? "success" : "warn");

  const status = document.createElement("div");
  status.className = "result-status";
  status.appendChild(icon(sufficient ? "check" : "warn"));
  status.appendChild(document.createTextNode(sufficient ? "Answered" : "Insufficient evidence"));
  card.appendChild(status);

  card.appendChild(text("div", data.answer, "result-answer"));
  el.result.appendChild(card);

  if (data.sources && data.sources.length) {
    el.result.appendChild(buildSourcesSection(data.sources));
  }

  el.result.appendChild(buildTechDetails(data));
  if (data.execution_trace) el.result.appendChild(buildExecutionDetails(data));
}

function parsePage(chunkId) {
  const m = /_p(\d+)$/.exec(chunkId || "");
  return m ? m[1] : null;
}

function buildSourcesSection(sources) {
  const wrap = document.createElement("div");
  wrap.appendChild(text("div", "Sources", "sources-heading"));
  sources.forEach((s) => {
    const card = document.createElement("div");
    card.className = "source-card";
    const page = parsePage(s.chunk_id);
    const metaText = page ? `${s.source} · page ${page}` : s.source;
    card.appendChild(text("div", metaText, "source-meta"));
    card.appendChild(text("div", s.excerpt, "source-excerpt"));
    wrap.appendChild(card);
  });
  return wrap;
}

function buildTechDetails(data) {
  const details = document.createElement("details");
  details.className = "tech-details";
  const summary = document.createElement("summary");
  summary.textContent = "Technical details";
  details.appendChild(summary);

  const grid = document.createElement("div");
  grid.className = "tech-grid";
  const rows = [
    ["Mode", data.config],
    ["Retrieval", `${data.retrieval_latency_ms.toFixed(0)} ms`],
  ];
  if (data.rerank_latency_ms != null) rows.push(["Reranking", `${data.rerank_latency_ms.toFixed(0)} ms`]);
  if (data.generation_latency_ms != null) rows.push(["Generation", `${data.generation_latency_ms.toFixed(0)} ms`]);
  if (data.sources && data.sources.length) {
    data.sources.forEach((s, i) => {
      rows.push([`Source ${i + 1} chunk ID`, s.chunk_id]);
      rows.push([`Source ${i + 1} score`, s.score.toFixed(4)]);
    });
  }
  rows.forEach(([label, value]) => {
    grid.appendChild(text("div", label, ""));
    grid.appendChild(text("div", String(value), ""));
  });
  details.appendChild(grid);
  return details;
}

// Agent-workflow-only: node names, tool names, durations, and the route
// outcome for one agent run. Metadata only — the server never includes
// raw document content or full tool arguments here (app/agent/graph.py's
// _trace() builds this from short, fixed-shape detail strings).
function buildExecutionDetails(data) {
  const details = document.createElement("details");
  details.className = "tech-details";
  const summary = document.createElement("summary");
  summary.textContent = "Execution details";
  details.appendChild(summary);

  const table = document.createElement("div");
  table.className = "trace-table";
  ["Node", "Tool", "Duration", "Outcome"].forEach((h) => table.appendChild(text("div", h, "trace-head")));
  data.execution_trace.forEach((step) => {
    table.appendChild(text("div", step.node, "trace-cell"));
    table.appendChild(text("div", step.tool || "—", "trace-cell"));
    table.appendChild(text("div", `${step.duration_ms.toFixed(0)} ms`, "trace-cell"));
    table.appendChild(text("div", step.detail, "trace-cell trace-detail"));
  });
  details.appendChild(table);

  if (data.route) {
    details.appendChild(text("div", `Route taken: ${data.route}`, "trace-route"));
  }
  return details;
}

// ---------- init ----------

(async function init() {
  loadUploadChips();
  await loadDemoChips();
  switchMode("demo");
})();
