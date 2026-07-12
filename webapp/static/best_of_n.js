// Best-of-N showcase: fetch the precomputed artifact and render the aggregate story + an
// interactive candidate stepper. Vanilla ES module, no libraries — matches app.js conventions.

const $ = (selector, root = document) => root.querySelector(selector);

const TOAST_MS = 1800;
const LADDER_MAX = 0.9; // real-corpus ceiling ~0.89 -> full-width bar
let toastTimer = null;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function svgIcon(name, className = "icon") {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", className);
  svg.setAttribute("aria-hidden", "true");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `#icon-${name}`);
  svg.append(use);
  return svg;
}

function showToast(message) {
  const toast = $("#toast");
  window.clearTimeout(toastTimer);
  toast.textContent = message;
  toast.hidden = false;
  toastTimer = window.setTimeout(() => { toast.hidden = true; }, TOAST_MS);
}

function fmt(value, digits = 2) {
  return Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function fmtPct(value) {
  return Number.isFinite(value) ? `${Math.round(value * 100)}%` : "—";
}

function hideOnError(img) {
  img.addEventListener("error", () => { img.style.opacity = "0"; }, { once: true });
}

// ---- Aggregate: stat tiles + fidelity ladder ---------------------------------------------------

function renderStats(aggregate) {
  const row = $("#statRow");
  const greedy = aggregate.greedy_baseline;
  const best = aggregate.best_of_n_fidelity
    ?? aggregate.oracle_at_n?.["t=1.0"]?.best_of_N ?? aggregate.oracle_at_n?.["t=0.7"]?.best_of_N;
  const multiple = greedy ? best / greedy : null;
  const tiles = [
    { value: `${fmt(greedy)} → ${fmt(best)}`, label: "Behavioral fidelity: free-running greedy → best-of-N reranked" },
    { value: multiple ? `${multiple.toFixed(1)}×` : "—", label: "Fidelity gain from sample + rerank, no retraining" },
    { value: `${fmtPct(aggregate.greedy_collapse_rate)} → ${fmtPct(aggregate.best_pick_collapse_rate)}`,
      label: "Code-collapse rate: greedy → best-of-N pick" },
  ];
  tiles.forEach((tile) => {
    const card = element("div", "bon-stat");
    card.append(element("div", "bon-stat__value", tile.value), element("div", "bon-stat__label", tile.label));
    row.append(card);
  });
}

function renderLadder(aggregate) {
  const ladder = $("#ladder");
  (aggregate.ladder || []).forEach((rung) => {
    const rowEl = element("div", "ladder__row");
    const label = element("div", "ladder__label");
    label.append(element("span", "ladder__name", rung.label));
    if (Number.isFinite(rung.collapse_rate)) {
      label.append(element("span", "ladder__sub", `${fmtPct(rung.collapse_rate)} collapse`));
    }
    const track = element("div", "ladder__track");
    const fill = element("div", `ladder__fill ladder__fill--${rung.kind}`);
    fill.style.width = `${Math.max(3, Math.min(100, (rung.fidelity / LADDER_MAX) * 100))}%`;
    track.append(fill, element("span", "ladder__val", fmt(rung.fidelity)));
    rowEl.append(label, track);
    if (rung.note) rowEl.append(element("p", "ladder__note", rung.note));
    ladder.append(rowEl);
  });
  $("#ladderCaption").textContent = aggregate.source || "";
}

// ---- Aggregate: oracle@N line chart (inline SVG) -----------------------------------------------

const SVG_NS = "http://www.w3.org/2000/svg";
function svgEl(tag, attrs) {
  const node = document.createElementNS(SVG_NS, tag);
  Object.entries(attrs || {}).forEach(([k, v]) => node.setAttribute(k, String(v)));
  return node;
}

function renderOracle(aggregate) {
  const data = aggregate.oracle_at_n || {};
  const series = [
    { key: "t=0.7", color: "var(--accent)", label: "oracle@N · t=0.7" },
    { key: "t=1.0", color: "var(--teal)", label: "oracle@N · t=1.0" },
  ].filter((s) => data[s.key]);
  // Derive the N axis from the data's own numeric keys (excludes "best_of_N"), so a regenerated
  // artifact with a different sample budget still renders correctly.
  const ns = Array.from(new Set(series.flatMap((s) => Object.keys(data[s.key] || {}))))
    .filter((k) => /^\d+$/.test(k))
    .sort((a, b) => Number(a) - Number(b));
  const W = 640, H = 300, padL = 48, padR = 20, padT = 20, padB = 44;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const allVals = series.flatMap((s) => ns.map((n) => data[s.key][n]).filter(Number.isFinite));
  const maxY = Math.max(0.35, ...allVals, aggregate.greedy_baseline || 0);
  const yMax = Math.ceil(maxY * 20) / 20; // round up to nearest 0.05
  const denom = Math.max(1, ns.length - 1);
  const x = (i) => padL + (i / denom) * plotW;
  const y = (v) => padT + plotH - (v / yMax) * plotH;

  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", role: "img",
    "aria-label": "oracle@N fidelity versus number of samples" });

  // y gridlines + ticks
  for (let t = 0; t <= yMax + 1e-9; t += 0.05) {
    const gy = y(t);
    svg.append(svgEl("line", { x1: padL, y1: gy, x2: W - padR, y2: gy, stroke: "var(--border)", "stroke-width": 1 }));
    const lbl = svgEl("text", { x: padL - 8, y: gy + 4, "text-anchor": "end", fill: "var(--text-tertiary)",
      "font-size": 11, "font-family": "var(--font-mono)" });
    lbl.textContent = t.toFixed(2);
    svg.append(lbl);
  }
  // x ticks
  ns.forEach((n, i) => {
    const lbl = svgEl("text", { x: x(i), y: H - padB + 20, "text-anchor": "middle", fill: "var(--text-tertiary)",
      "font-size": 11, "font-family": "var(--font-mono)" });
    lbl.textContent = `N=${n}`;
    svg.append(lbl);
  });
  const xTitle = svgEl("text", { x: padL + plotW / 2, y: H - 6, "text-anchor": "middle",
    fill: "var(--text-secondary)", "font-size": 12 });
  xTitle.textContent = "samples reranked";
  svg.append(xTitle);

  // greedy baseline
  if (Number.isFinite(aggregate.greedy_baseline)) {
    const gy = y(aggregate.greedy_baseline);
    svg.append(svgEl("line", { x1: padL, y1: gy, x2: W - padR, y2: gy, stroke: "var(--error)",
      "stroke-width": 2, "stroke-dasharray": "5 5" }));
  }
  // series polylines + dots
  series.forEach((s) => {
    const pts = ns.map((n, i) => `${x(i)},${y(data[s.key][n])}`).join(" ");
    svg.append(svgEl("polyline", { points: pts, fill: "none", stroke: s.color, "stroke-width": 2.5,
      "stroke-linejoin": "round" }));
    ns.forEach((n, i) => svg.append(svgEl("circle", { cx: x(i), cy: y(data[s.key][n]), r: 3.5, fill: s.color })));
  });
  $("#oracleChart").append(svg);

  const legend = $("#oracleLegend");
  series.forEach((s) => {
    const item = element("span");
    const swatch = element("i");
    swatch.style.borderTopColor = s.color;
    item.append(swatch, document.createTextNode(s.label));
    legend.append(item);
  });
  const greedyItem = element("span");
  const gSwatch = element("i");
  gSwatch.style.borderTopColor = "var(--error)";
  gSwatch.style.borderTopStyle = "dashed";
  greedyItem.append(gSwatch, document.createTextNode(`greedy baseline ${fmt(aggregate.greedy_baseline)}`));
  legend.append(greedyItem);
}

// ---- Interactive run: tabs + compare + candidate rail ------------------------------------------

let EXAMPLES = [];
let META = {};
let activeExample = 0;
let selectedCandidate = 0;

// Human-readable provenance for a candidate's `source`, so a viewer can tell a real corpus look
// from the illustrative synthetic-collapse item and (in --from-model mode) a real model sample.
function sourceLabel(source) {
  const s = String(source || "");
  if (s === "corpus:self") return "corpus ground-truth";
  if (s.startsWith("corpus:")) return "corpus look";
  if (s === "synthetic:collapse") return "synthetic · illustrative";
  if (s === "model:greedy") return "greedy baseline";
  if (s.startsWith("model:")) return "model sample";
  return s || "candidate";
}

function compareView(originalUrl, gradedUrl) {
  const compare = element("div", "compare");
  const after = element("img", "compare__image compare__after");
  after.alt = "Selected candidate LUT applied";
  after.src = gradedUrl;
  hideOnError(after);
  const beforeLayer = element("div", "compare__before-layer");
  const before = element("img", "compare__image");
  before.src = originalUrl;
  before.alt = "Ungraded original image";
  hideOnError(before);
  beforeLayer.append(before);
  const beforeLabel = element("span", "compare__label compare__label--before", "original");
  const afterLabel = element("span", "compare__label compare__label--after", "best-of-N pick");
  const range = element("input", "compare__range");
  range.type = "range";
  range.min = "0";
  range.max = "100";
  range.value = "50";
  range.setAttribute("aria-label", "Reveal original versus graded");
  range.addEventListener("input", () => compare.style.setProperty("--split", `${range.value}%`));
  const divider = element("span", "compare__divider");
  divider.append(element("span", "compare__handle"));
  after.addEventListener("load", () => {
    if (after.naturalWidth && after.naturalHeight) {
      compare.style.aspectRatio = `${after.naturalWidth} / ${after.naturalHeight}`;
    }
  });
  compare.append(after, beforeLayer, beforeLabel, afterLabel, range, divider);
  compare.style.setProperty("--split", "50%");
  return { compare, after, afterLabel };
}

function referenceCard(preview) {
  const card = element("button", "ref-card");
  card.type = "button";
  card.setAttribute("aria-label", `${preview.name}: toggle graded and original`);
  card.setAttribute("aria-pressed", "false");
  const well = element("span", "ref-card__image-well");
  const graded = element("img", "ref-card__image");
  graded.src = preview.graded_url;
  graded.alt = `${preview.name} with the winning LUT applied`;
  graded.loading = "lazy";
  hideOnError(graded);
  const original = element("img", "ref-card__image ref-card__original");
  original.src = preview.original_url;
  original.alt = "";
  original.loading = "lazy";
  hideOnError(original);
  well.append(graded, original);
  const footer = element("span", "ref-card__footer");
  footer.append(element("span", "ref-card__name", preview.name), element("span", "ref-card__mode", "graded"));
  card.append(well, footer);
  card.addEventListener("click", () => {
    const showBefore = card.classList.toggle("is-before");
    card.setAttribute("aria-pressed", String(showBefore));
    $(".ref-card__mode", card).textContent = showBefore ? "original" : "graded";
  });
  return card;
}

async function downloadCube(url, filename, button) {
  const span = button.querySelector("span");
  const label = span?.textContent;
  button.disabled = true;
  if (span) span.textContent = "Preparing…";
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const blob = await response.blob();
    const objectUrl = URL.createObjectURL(blob);
    const anchor = element("a");
    anchor.href = objectUrl;
    anchor.download = filename;
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    window.setTimeout(() => URL.revokeObjectURL(objectUrl), TOAST_MS);
    showToast(`${filename} downloaded`);
  } catch (error) {
    showToast(`Download failed: ${error.message}`);
  } finally {
    button.disabled = false;
    if (span) span.textContent = label;
  }
}

function candidateBadges(candidate) {
  const wrap = element("div", "result-heading__meta");
  wrap.append(element("span", "quality-badge", `fidelity ${fmt(candidate.behavioral_fidelity)}`));
  if (candidate.source) wrap.append(element("span", "quality-badge", sourceLabel(candidate.source)));
  if (candidate.is_winner) wrap.append(element("span", "status-badge", "reranker pick"));
  if (candidate.collapsed) wrap.append(element("span", "quality-badge", "collapsed"));
  return wrap;
}

function renderRun() {
  const example = EXAMPLES[activeExample];
  const body = $("#runBody");
  body.replaceChildren();
  if (!example) return;
  selectedCandidate = example.winner_index ?? 0;

  // Left column: prompt + compare + candidate rail
  const left = element("div");
  const prompt = element("div", "bon-run__prompt", `“${example.instruction}”`);
  const spec = element("details", "spec");
  spec.append(element("summary", "", "Requested spec (what the reranker scores against)"),
    element("code", "", example.spec_text));
  const { compare, after, afterLabel } = compareView(example.hero_original_url,
    example.candidates[selectedCandidate].graded_url);

  const rail = element("div", "cand-rail");
  const scroll = element("div", "cand-scroll");
  const chips = [];
  example.candidates.forEach((candidate, index) => {
    const chip = element("button", "cand-chip");
    chip.type = "button";
    chip.setAttribute("aria-label", `Candidate ${index + 1}, fidelity ${fmt(candidate.behavioral_fidelity)}`);
    if (candidate.is_winner) chip.classList.add("is-winner");
    if (candidate.collapsed) chip.classList.add("is-collapsed");
    const img = element("img", "cand-chip__img");
    img.src = candidate.graded_url;
    img.alt = "";
    img.loading = "lazy";
    hideOnError(img);
    const meta = element("div", "cand-chip__meta");
    meta.append(element("span", "", `#${index + 1}`), element("span", "", fmt(candidate.behavioral_fidelity)));
    chip.append(img, meta);
    if (candidate.is_winner) {
      const crown = element("span", "cand-chip__crown");
      crown.append(svgIcon("check", "icon"));
      chip.append(crown);
    }
    chip.addEventListener("click", () => selectCandidate(index));
    scroll.append(chip);
    chips.push(chip);
  });
  rail.append(scroll);
  left.append(prompt, spec, compare, element("p", "bon-run__hint",
    "Drag the divider to compare the ungraded original with the selected candidate’s LUT."));
  if (META.candidate_source !== "model_sampled") {
    left.append(element("p", "bon-run__hint",
      "This GPU-free demo puts the corpus ground-truth look in the candidate pool, so the reranker’s top " +
      "pick reaches ~1.0 — that is a corpus ceiling, not the deploy number. The real free-running win is the " +
      "0.159 → 0.307 headline above (regenerate with --from-model to rerank the generator’s own samples)."));
  }
  left.append(rail);

  // Right column: selected-candidate detail + reference consistency + download
  const detail = element("div", "cand-detail");

  // References (winner LUT applied to reference photos)
  const refsWrap = element("div", "bon-refs");
  if (Array.isArray(example.references) && example.references.length) {
    refsWrap.append(element("p", "eyebrow", "Winner LUT · consistency check"));
    const grid = element("div", "bon-ref-grid");
    example.references.forEach((preview) => grid.append(referenceCard(preview)));
    refsWrap.append(grid);
  }
  const download = element("button", "btn btn--secondary");
  download.type = "button";
  download.append(svgIcon("download"), element("span", "", "Download winner .cube"));
  download.addEventListener("click", () => downloadCube(example.winner_cube_url,
    `chroma_best_of_n_${example.id}.cube`, download));

  body.append(left, detail);

  // stash refs so selectCandidate can update in place (detail is filled by updateDetail)
  renderRun._state = { example, after, afterLabel, chips, detail, refsWrap, download };
  updateDetail();
}

function updateDetail() {
  const state = renderRun._state;
  if (!state) return;
  const { example, after, afterLabel, chips, detail, refsWrap, download } = state;
  const candidate = example.candidates[selectedCandidate];

  after.src = candidate.graded_url;
  afterLabel.textContent = candidate.is_winner ? "best-of-N pick" : `candidate #${selectedCandidate + 1}`;
  chips.forEach((chip, index) => chip.setAttribute("aria-current", String(index === selectedCandidate)));

  detail.replaceChildren();
  const top = element("div", "cand-detail__top");
  top.append(element("h3", "cand-detail__title", candidate.label || `Candidate ${selectedCandidate + 1}`),
    candidateBadges(candidate));
  detail.append(top);

  const metrics = element("div", "cand-metrics");
  const fidStrongClass = candidate.is_winner ? "good" : (candidate.collapsed ? "warn" : "");
  metrics.append(
    metric("behavioral fidelity", fmt(candidate.behavioral_fidelity), fidStrongClass),
    metric("collapsed", candidate.collapsed ? "yes" : "no", candidate.collapsed ? "warn" : "good"),
    metric("code entropy", fmt(candidate.entropy_norm), ""),
    metric("dominant share", fmtPct(candidate.dominant_share), candidate.dominant_share >= 0.5 ? "warn" : ""),
  );
  detail.append(metrics);

  if (candidate.note) detail.append(element("p", "cand-note", candidate.note));
  const codes = element("p", "cand-codes",
    `codes[0:12] = [${(candidate.codes_preview || []).join(", ")}, …]`);
  detail.append(codes);

  const nav = element("div", "cand-nav");
  const prev = navButton("arrow-left", "Prev", () => selectCandidate((selectedCandidate - 1 + example.n) % example.n));
  const next = navButton("arrow", "Next", () => selectCandidate((selectedCandidate + 1) % example.n));
  const jump = element("button", "btn btn--ghost");
  jump.type = "button";
  jump.append(svgIcon("check"), element("span", "", "Jump to reranker pick"));
  jump.addEventListener("click", () => selectCandidate(example.winner_index ?? 0));
  nav.append(prev, next, jump);
  detail.append(nav, refsWrap, download);
}

function metric(label, value, strongClass) {
  const box = element("div", "cand-metric");
  box.append(element("span", "", label));
  box.append(element("strong", strongClass || "", value));
  return box;
}

function navButton(icon, label, handler) {
  const button = element("button", "btn btn--secondary");
  button.type = "button";
  button.append(svgIcon(icon), element("span", "", label));
  button.addEventListener("click", handler);
  return button;
}

function selectCandidate(index) {
  selectedCandidate = index;
  updateDetail();
}

function renderTabs() {
  const tabs = $("#exampleTabs");
  EXAMPLES.forEach((example, index) => {
    const tab = element("button", "bon-tab");
    tab.type = "button";
    tab.setAttribute("role", "tab");
    tab.setAttribute("aria-selected", String(index === activeExample));
    tab.textContent = shortTitle(example);
    tab.addEventListener("click", () => {
      activeExample = index;
      [...tabs.children].forEach((t, i) => t.setAttribute("aria-selected", String(i === index)));
      renderRun();
    });
    tabs.append(tab);
  });
}

function shortTitle(example) {
  const text = example.instruction || example.id;
  return text.length > 34 ? `${text.slice(0, 32)}…` : text;
}

function renderProvenance(meta) {
  const block = $("#provenance");
  const source = meta.candidate_source === "model_sampled"
    ? "Candidates are the generator's own free-running samples."
    : "Candidates are real LUT looks drawn from the corpus, scored by the deployable reranker (this is not the generator's own sampling — regenerate with --from-model for that).";
  block.append(element("strong", "", "How this was built. "));
  block.append(document.createTextNode(
    `${source} Each candidate is decoded by the frozen VQ decoder and scored by ${meta.reranker || "the behavioral-fidelity reranker"}. ` +
    `${meta.note || ""}`));
}

async function init() {
  try {
    const response = await fetch("./best_of_n_showcase.json", { headers: { Accept: "application/json" } });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    META = data.meta || {};
    renderStats(data.aggregate || {});
    renderLadder(data.aggregate || {});
    renderOracle(data.aggregate || {});
    EXAMPLES = Array.isArray(data.examples) ? data.examples : [];
    renderProvenance(data.meta || {});
    if (EXAMPLES.length) {
      renderTabs();
      renderRun();
    } else {
      $("#runBody").append(element("p", "bon-error", "No showcase examples found. Run: python -m scripts.build_best_of_n_showcase"));
    }
  } catch (error) {
    const body = $("#runBody");
    body.append(element("div", "bon-error",
      `Could not load best_of_n_showcase.json (${error.message}). Generate it with: python -m scripts.build_best_of_n_showcase`));
  }
}

document.addEventListener("keydown", (event) => {
  if (!EXAMPLES.length) return;
  const example = EXAMPLES[activeExample];
  if (event.key === "ArrowRight") selectCandidate((selectedCandidate + 1) % example.n);
  else if (event.key === "ArrowLeft") selectCandidate((selectedCandidate - 1 + example.n) % example.n);
});

init();
