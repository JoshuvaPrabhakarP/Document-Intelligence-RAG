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
  uploading: false,
  asking: false,
  lastResult: { demo: null, upload: null },
};

const el = {
  tabDemo: document.getElementById("tab-demo"),
  tabUpload: document.getElementById("tab-upload"),
  panelDemo: document.getElementById("panel-demo"),
  panelUpload: document.getElementById("panel-upload"),
  demoChips: document.getElementById("demo-chips"),
  uploadChips: document.getElementById("upload-chips"),
  dropzone: document.getElementById("dropzone"),
  fileInput: document.getElementById("file-input"),
  uploadStatus: document.getElementById("upload-status"),
  fileList: document.getElementById("file-list"),
  clearBtn: document.getElementById("clear-btn"),
  question: document.getElementById("question"),
  askBtn: document.getElementById("ask-btn"),
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

function icon(name) {
  // Minimal inline SVGs — no icon-font/library dependency.
  const paths = {
    check: '<path d="M4 10l4 4 8-8" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/>',
    warn: '<path d="M10 3l8 14H2L10 3z" stroke="currentColor" stroke-width="1.6" fill="none" stroke-linejoin="round"/><circle cx="10" cy="14" r="0.9" fill="currentColor"/><path d="M10 8v3.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>',
    error: '<circle cx="10" cy="10" r="8" stroke="currentColor" stroke-width="1.6" fill="none"/><path d="M7 7l6 6M13 7l-6 6" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>',
  };
  const span = document.createElement("span");
  span.setAttribute("aria-hidden", "true");
  span.style.display = "inline-flex";
  span.innerHTML = `<svg width="16" height="16" viewBox="0 0 20 20">${paths[name] || ""}</svg>`;
  return span;
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
  btn.textContent = question;
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
  if (isImage) {
    const img = document.createElement("div");
    img.className = "thumb-icon";
    img.textContent = "IMG";
    card.appendChild(img);
  } else {
    const badge = document.createElement("div");
    badge.className = "thumb-icon";
    badge.textContent = "PDF";
    card.appendChild(badge);
  }

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
  el.askBtn.textContent = "Asking…";
  renderWorking();

  try {
    const res = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "same-origin",
      body: JSON.stringify({ question, top_k: 5, mode: state.mode }),
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
    el.askBtn.textContent = "Ask";
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

function renderResult(result) {
  clear(el.result);
  if (!result) return;

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

// ---------- init ----------

(async function init() {
  loadUploadChips();
  await loadDemoChips();
  switchMode("demo");
})();
