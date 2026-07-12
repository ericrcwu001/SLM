const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const query = new URLSearchParams(window.location.search);
const mockValue = query.get("mock");
const MOCK_MODE = mockValue !== null;
const MOCK_ROUTES = new Set(["grade", "clarify", "refuse", "error"]);
const MOCK_FORCED_ROUTE = MOCK_ROUTES.has(mockValue) ? mockValue : null;
const MAX_FILE_BYTES = 12 * 1024 * 1024;
const POPOVER_DELAY_MS = 120;
const LONG_PRESS_MS = 480;
const TOAST_MS = 1800;
const COPIED_MS = 1000;
const REFERENCE_NAMES = ["City", "Landscape", "Portrait", "Close-up", "Food", "Interior"];

const dom = {
  inputPanel: $("#input-panel"),
  uploadZone: $("#upload-zone"),
  fileInput: $("#image-input"),
  uploadEmpty: $("#upload-empty"),
  uploadFilled: $("#upload-filled"),
  uploadThumb: $("#upload-thumb"),
  uploadFilename: $("#upload-filename"),
  uploadMeta: $("#upload-meta"),
  uploadClear: $("#upload-clear"),
  sampleImage: $("#sample-image"),
  inputError: $("#input-error"),
  prompt: $("#prompt-text"),
  promptCount: $("#prompt-count"),
  generate: $("#generate"),
  generateLabel: $("#generate-label"),
  generateArrow: $(".generate-arrow"),
  spinner: $(".spinner"),
  results: $("#results"),
  popover: $("#term-popover"),
  termsOpen: $("#terms-open"),
  drawerShell: $("#drawer-shell"),
  drawer: $("#terms-drawer"),
  drawerBackdrop: $("#drawer-backdrop"),
  termsClose: $("#terms-close"),
  termSearch: $("#term-search"),
  drawerCount: $("#drawer-count"),
  glossary: $("#glossary"),
  toast: $("#toast"),
  modeBadge: $("#mode-badge"),
};

const state = {
  file: null,
  fileUrl: null,
  busy: false,
  terms: [],
  termMap: new Map(),
  termRegistry: new Map(),
  nextTermId: 0,
  popoverTimer: null,
  activePopoverChip: null,
  longPressTimer: null,
  toastTimer: null,
  lastFocused: null,
  mockObjectUrls: [],
};

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

function resultState(name) {
  const node = element("div", `state state--${name}`);
  dom.results.replaceChildren(node);
  dom.results.setAttribute("data-state", name);
  return node;
}

function sectionTitle(eyebrow, heading) {
  const wrap = element("div");
  wrap.append(element("p", "eyebrow", eyebrow), element("h2", "", heading));
  return wrap;
}

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function setInputError(message = "") {
  dom.inputError.textContent = message;
  dom.inputError.hidden = !message;
}

function autoGrowPrompt() {
  dom.prompt.style.height = "auto";
  dom.prompt.style.height = `${Math.min(dom.prompt.scrollHeight, 288)}px`;
}

function updateGenerateAvailability() {
  dom.generate.disabled = state.busy || !state.file || !dom.prompt.value.trim();
}

function setBusy(busy) {
  state.busy = busy;
  dom.inputPanel.inert = busy;
  dom.inputPanel.classList.toggle("is-loading", busy);
  dom.generate.setAttribute("aria-busy", String(busy));
  dom.generateLabel.textContent = busy ? "Generating…" : "Generate LUT";
  dom.generateArrow.hidden = busy;
  dom.spinner.hidden = !busy;
  dom.results.setAttribute("aria-busy", String(busy));
  updateGenerateAvailability();
}

function imageDimensions(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve({ width: image.naturalWidth, height: image.naturalHeight });
    image.onerror = () => reject(new Error("The selected file could not be decoded as an image."));
    image.src = url;
  });
}

async function selectFile(file) {
  setInputError();
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    setInputError("Choose a JPG, PNG, or WEBP image.");
    return;
  }
  if (file.size > MAX_FILE_BYTES) {
    setInputError("That image is over 12 MB. Choose a smaller file.");
    return;
  }

  const nextUrl = URL.createObjectURL(file);
  try {
    const dimensions = await imageDimensions(nextUrl);
    if (state.fileUrl) URL.revokeObjectURL(state.fileUrl);
    state.file = file;
    state.fileUrl = nextUrl;
    dom.uploadThumb.src = nextUrl;
    dom.uploadFilename.textContent = file.name || "untitled image";
    dom.uploadMeta.textContent = `${formatBytes(file.size)} · ${dimensions.width}×${dimensions.height}`;
    dom.uploadEmpty.hidden = true;
    dom.uploadFilled.hidden = false;
    dom.uploadClear.hidden = false;
    dom.fileInput.value = "";
    updateGenerateAvailability();
  } catch (error) {
    URL.revokeObjectURL(nextUrl);
    setInputError(error.message);
  }
}

function clearFile() {
  if (state.fileUrl) URL.revokeObjectURL(state.fileUrl);
  state.file = null;
  state.fileUrl = null;
  dom.fileInput.value = "";
  dom.uploadThumb.removeAttribute("src");
  dom.uploadEmpty.hidden = false;
  dom.uploadFilled.hidden = true;
  dom.uploadClear.hidden = true;
  setInputError();
  updateGenerateAvailability();
  dom.uploadZone.focus();
}

async function selectSampleImage() {
  setInputError();
  dom.sampleImage.disabled = true;
  try {
    const response = await fetch("/assets/references/portrait.jpg");
    if (!response.ok) throw new Error(`Sample image returned HTTP ${response.status}.`);
    const blob = await response.blob();
    await selectFile(new File([blob], "sample-portrait.jpg", { type: blob.type || "image/jpeg" }));
  } catch (error) {
    setInputError(`Could not load the sample portrait: ${error.message}`);
  } finally {
    dom.sampleImage.disabled = false;
  }
}

function renderLoading() {
  const root = resultState("loading");
  const heading = element("div", "result-heading");
  heading.append(sectionTitle("Result", "Developing your grade"));
  const badge = element("span", "quality-badge", "routing / sampling / rendering");
  heading.append(badge);
  const hero = element("div", "skeleton skeleton--hero");
  hero.setAttribute("aria-hidden", "true");
  const grid = element("div", "skeleton-grid");
  for (let index = 0; index < 6; index += 1) {
    const card = element("span", "skeleton skeleton--card");
    card.setAttribute("aria-hidden", "true");
    grid.append(card);
  }
  const line = element("div", "skeleton skeleton--line");
  const shortLine = element("div", "skeleton skeleton--line skeleton--line-short");
  line.setAttribute("aria-hidden", "true");
  shortLine.setAttribute("aria-hidden", "true");
  root.append(heading, hero, grid, line, shortLine);
  window.requestAnimationFrame(() => {
    dom.results.scrollIntoView({
      behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth",
      block: "start",
    });
  });
}

function qualityBadges(quality) {
  const wrap = element("div", "result-heading__meta");
  wrap.append(element("span", "status-badge", "LUT ready"));
  if (Number.isFinite(quality?.behavioral_fidelity)) {
    wrap.append(element("span", "quality-badge", `fidelity ${quality.behavioral_fidelity.toFixed(2)}`));
  }
  if (quality?.fell_back_greedy) wrap.append(element("span", "quality-badge", "greedy fallback"));
  if (quality?.collapsed) wrap.append(element("span", "quality-badge", "low code diversity"));
  return wrap;
}

function hideOnError(img) {
  // A 404'd preview keeps its tile geometry (the well/compare has a fixed aspect-ratio) rather than
  // rendering the browser's broken-image glyph.
  img.addEventListener("error", () => { img.style.opacity = "0"; }, { once: true });
}

function compareView(preview) {
  const compare = element("div", "compare");
  const after = element("img", "compare__image compare__after");
  after.src = preview.graded_url;
  after.alt = "Image with the generated LUT applied";
  hideOnError(after);
  const beforeLayer = element("div", "compare__before-layer");
  const before = element("img", "compare__image");
  before.src = preview.original_url;
  before.alt = "Original image before grading";
  hideOnError(before);
  beforeLayer.append(before);

  const beforeLabel = element("span", "compare__label compare__label--before", "before");
  const afterLabel = element("span", "compare__label compare__label--after", "after");
  const range = element("input", "compare__range");
  range.type = "range";
  range.min = "0";
  range.max = "100";
  range.value = "50";
  range.setAttribute("aria-label", "Reveal original versus graded image");
  range.addEventListener("input", () => {
    compare.style.setProperty("--split", `${range.value}%`);
  });

  const divider = element("span", "compare__divider");
  divider.append(element("span", "compare__handle"));
  after.addEventListener("load", () => {
    if (after.naturalWidth && after.naturalHeight) {
      compare.style.aspectRatio = `${after.naturalWidth} / ${after.naturalHeight}`;
    }
  }, { once: true });
  compare.append(after, beforeLayer, beforeLabel, afterLabel, range, divider);
  return compare;
}

function slug(text) {
  const cleaned = text.toLowerCase().normalize("NFKD")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 48);
  return cleaned || "custom-look";
}

function cubeFilename() {
  const date = new Date();
  const stamp = [date.getFullYear(), String(date.getMonth() + 1).padStart(2, "0"), String(date.getDate()).padStart(2, "0")].join("");
  return `chroma_${slug(dom.prompt.value)}_${stamp}.cube`;
}

async function downloadCube(url, filename, button) {
  const originalText = button.querySelector("span")?.textContent;
  button.disabled = true;
  if (button.querySelector("span")) button.querySelector("span").textContent = "Preparing…";
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Download returned HTTP ${response.status}.`);
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
    if (button.querySelector("span")) button.querySelector("span").textContent = originalText;
  }
}

function referenceCard(preview, fallbackName, index) {
  const name = preview.name || fallbackName || `Reference ${index + 1}`;
  const card = element("button", "ref-card");
  card.type = "button";
  card.setAttribute("aria-label", `${name}: toggle graded and original reference`);
  card.setAttribute("aria-pressed", "false");

  const well = element("span", "ref-card__image-well");
  const graded = element("img", "ref-card__image");
  graded.src = preview.graded_url;
  graded.alt = `${name} with the generated LUT applied`;
  graded.loading = "lazy";
  graded.decoding = "async";
  hideOnError(graded);
  const original = element("img", "ref-card__image ref-card__original");
  original.src = preview.original_url;
  original.alt = "";
  original.loading = "lazy";
  original.decoding = "async";
  hideOnError(original);
  well.append(graded, original);

  const footer = element("span", "ref-card__footer");
  footer.append(element("span", "ref-card__name", name), element("span", "ref-card__mode", "graded"));
  card.append(well, footer);
  card.addEventListener("click", () => {
    const showBefore = card.classList.toggle("is-before");
    card.setAttribute("aria-pressed", String(showBefore));
    $(".ref-card__mode", card).textContent = showBefore ? "original" : "graded";
  });
  card.addEventListener("mouseenter", () => {
    $(".ref-card__mode", card).textContent = "original";
  });
  card.addEventListener("mouseleave", () => {
    $(".ref-card__mode", card).textContent = card.classList.contains("is-before") ? "original" : "graded";
  });
  return card;
}

function validGroundedTerms(feedback) {
  return (Array.isArray(feedback?.suggested_terms) ? feedback.suggested_terms : [])
    .filter((term) => term && typeof term.term === "string" && term.term.trim() && term.grounded === true);
}

function termBucket(term) {
  const axis = String(term.axis || "").toLowerCase();
  const category = String(term.category || "").toLowerCase();
  const label = String(term.term || "").toLowerCase();
  if (axis.includes("magnitude") || axis.includes("intensity") || axis.includes("strength") ||
      /^(barely|slight|slightly|subtle|subtly|moderate|moderately|strong|strongly|heavy|heavily|extreme|extremely)$/.test(label)) {
    return "magnitude";
  }
  if (axis.includes("style") || axis.includes("composite") || category.includes("style")) return "style";
  return "direction";
}

function registerTerm(button, term) {
  const key = `term-${state.nextTermId += 1}`;
  state.termRegistry.set(key, term);
  button.dataset.termKey = key;
}

function termChip(term, bucket, { prominent = false, reprompt = false } = {}) {
  const chip = element("button", `chip chip--${bucket}${prominent ? " chip--prominent" : ""}`);
  chip.type = "button";
  chip.append(element("span", "", term.term));
  registerTerm(chip, term);
  chip.addEventListener("click", async () => {
    await copyTerm(term.term, chip);
    if (reprompt) dom.prompt.focus();
  });
  return chip;
}

function feedbackGroups(feedback, options = {}) {
  const terms = validGroundedTerms(feedback);
  const groups = {
    magnitude: [],
    direction: [],
    style: [],
  };
  terms.forEach((term) => groups[termBucket(term)].push(term));
  const labels = {
    magnitude: "Add magnitude",
    direction: "Clarify direction",
    style: "Refine style",
  };
  const wrap = element("div", "feedback__groups");
  Object.entries(groups).forEach(([bucket, bucketTerms]) => {
    if (!bucketTerms.length) return;
    const group = element("div", `term-group term-group--${bucket}`);
    group.append(element("p", "term-group__title", labels[bucket]));
    const chips = element("div", "term-group__chips");
    bucketTerms.forEach((term) => chips.append(termChip(term, bucket, options)));
    group.append(chips);
    wrap.append(group);
  });
  return wrap;
}

function feedbackPanel(feedback) {
  const panel = element("section", "feedback");
  panel.setAttribute("aria-labelledby", "feedback-heading");
  const header = element("div", "feedback__header");
  const copy = element("div");
  const heading = element("h2", "", "Sharpen your prompt");
  heading.id = "feedback-heading";
  copy.append(heading, element("p", "feedback__assessment", feedback?.assessment || "Use a measured direction and an explicit strength for a more controlled grade."));
  const hint = element("span", "feedback__hint");
  hint.append(svgIcon("info"), document.createTextNode("Hover for definition · click to copy"));
  header.append(copy, hint);
  panel.append(header);
  const groups = feedbackGroups(feedback);
  if (groups.childElementCount) panel.append(groups);
  return panel;
}

function renderGrade(data) {
  const previews = Array.isArray(data.previews) ? data.previews : [];
  const userPreview = previews[0];
  if (!userPreview?.original_url || !userPreview?.graded_url || !data.lut?.cube_url) {
    throw new Error("The grade response is missing its user preview or LUT artifact.");
  }

  const root = resultState("grade");
  const heading = element("div", "result-heading");
  heading.append(sectionTitle("Result", "A portable color transform"), qualityBadges(data.quality));
  root.append(heading);

  const hero = element("section", "hero");
  const toolbar = element("div", "hero-toolbar");
  const label = element("div", "hero-toolbar__label");
  label.append(element("span", "", "Your image"), element("span", "", "drag the divider to inspect the transform"));
  const filename = cubeFilename();
  const downloadWrap = element("div", "download-wrap");
  const downloadButton = element("button", "btn btn--secondary");
  downloadButton.type = "button";
  downloadButton.append(svgIcon("download"), element("span", "", "Download .cube"));
  downloadButton.addEventListener("click", () => downloadCube(data.lut.cube_url, filename, downloadButton));
  downloadWrap.append(downloadButton, element("span", "download__filename", filename));
  toolbar.append(label, downloadWrap);
  hero.append(toolbar, compareView(userPreview));
  if (data.attribute_spec_text) {
    const details = element("details", "spec");
    details.append(element("summary", "", "What the model understood"), element("code", "", data.attribute_spec_text));
    hero.append(details);
  }
  root.append(hero);

  const references = previews.slice(1, 7);
  if (references.length) {
    const section = element("section", "reference-section");
    section.setAttribute("aria-labelledby", "references-heading");
    const referenceHeading = element("div", "reference-heading");
    const titleWrap = element("div");
    const title = element("h2", "", "Applied to reference shots");
    title.id = "references-heading";
    titleWrap.append(element("p", "eyebrow", "Consistency check"), title);
    referenceHeading.append(titleWrap, element("p", "", "The same LUT across skin, sky, foliage, food, architecture, and mixed light."));
    const grid = element("div", "reference-grid");
    references.forEach((preview, index) => grid.append(referenceCard(preview, REFERENCE_NAMES[index], index)));
    section.append(referenceHeading, grid);
    root.append(section);
  }
  root.append(feedbackPanel(data.prompt_feedback || {}));
}

function renderClarify(data) {
  const root = resultState("clarify");
  const card = element("section", "route-card route-card--clarify");
  card.append(iconCircle("adjust", "route-card__icon"));
  card.append(element("h2", "", "Let’s pin down the direction"));
  card.append(element("p", "route-card__message", data.clarify_message || "That look could go a few ways. Name a color direction and how strongly it should read."));
  card.append(element("p", "route-card__coach", "Try adding one of these measured terms, then generate again:"));
  const groups = feedbackGroups(data.prompt_feedback || {}, { prominent: true, reprompt: true });
  if (groups.childElementCount) card.append(groups);
  card.append(element("p", "route-card__hint", "Copied terms are ready to add to your prompt above."));
  root.append(card);
}

function humanRefuse(reason) {
  if (reason === "out_of_gamut") {
    return "That color request falls outside a reliable photographic gamut for a single LUT. Try a smaller hue or temperature shift.";
  }
  if (reason === "out_of_scope") {
    return "Selecting, removing, or changing an object is a local edit. A LUT applies the same color mapping everywhere in the frame.";
  }
  return "That request needs more than a global color transform. A LUT can shape color and tone, but it cannot understand or edit individual objects.";
}

function refuseRedirectTerms(feedback) {
  const supplied = validGroundedTerms(feedback);
  if (supplied.length) return supplied.slice(0, 5);
  const preferred = ["warmer", "lifted_blacks", "more_contrast", "muted"]
    .map((name) => state.termMap.get(name))
    .filter((term) => term?.grounded === true);
  const fallback = state.terms.filter((term) => term?.grounded === true && !preferred.includes(term));
  return [...preferred, ...fallback].slice(0, 5);
}

function renderRefuse(data) {
  const root = resultState("refuse");
  const card = element("section", "route-card route-card--refuse scope-card");
  card.append(iconCircle("scope", "route-card__icon"));
  card.append(element("h2", "", "This one’s outside what a single LUT can do"));
  card.append(element("p", "route-card__message", humanRefuse(data.refuse_reason)));
  card.append(element("p", "route-card__coach", "What it can do: shape global contrast, color balance, saturation, and tonal character."));
  const chips = element("div", "scope-card__terms");
  refuseRedirectTerms(data.prompt_feedback || {}).forEach((term) => {
    chips.append(termChip(term, termBucket(term), { prominent: true, reprompt: true }));
  });
  if (chips.childElementCount) card.append(chips);
  card.append(element("p", "route-card__hint", "Try a global color look instead."));
  root.append(card);
}

function iconCircle(name, className) {
  const circle = element("span", className);
  circle.append(svgIcon(name, "icon icon--large"));
  return circle;
}

function renderError(error) {
  const root = resultState("error");
  const card = element("section", "error-card");
  card.append(iconCircle("warning", "error-card__icon"));
  card.append(element("h2", "", "Something went wrong generating the LUT"));
  const primary = error?.code === "generation_timeout"
    ? "The grade took longer than the configured inference window. Your image and prompt are still ready to retry."
    : "The pipeline could not complete this render. Your image and prompt are still ready to retry.";
  card.append(element("p", "error-card__message", primary));
  const details = element("details", "error-detail");
  details.append(element("summary", "", "Technical detail"));
  details.append(element("code", "", `${error?.code || "client_error"}: ${error?.message || "Unknown error"}`));
  const retry = element("button", "btn btn--secondary");
  retry.type = "button";
  retry.append(svgIcon("arrow"), element("span", "", "Retry"));
  retry.addEventListener("click", generate);
  card.append(details, retry);
  root.append(card);
}

async function parseApiError(response) {
  let body = null;
  try {
    body = await response.json();
  } catch {
    // The status text below remains useful when a proxy returns non-JSON.
  }
  const error = new Error(body?.error?.message || response.statusText || `HTTP ${response.status}`);
  error.code = body?.error?.code || `http_${response.status}`;
  error.status = response.status;
  return error;
}

async function requestGeneration() {
  if (MOCK_MODE) return mockGenerate();
  const form = new FormData();
  form.append("image", state.file);
  form.append("prompt", dom.prompt.value.trim());
  const response = await fetch("/api/generate", { method: "POST", body: form });
  if (!response.ok) throw await parseApiError(response);
  return response.json();
}

async function generate() {
  if (state.busy || !state.file || !dom.prompt.value.trim()) return;
  setInputError();
  setBusy(true);
  renderLoading();
  try {
    const data = await requestGeneration();
    if (!data || typeof data.route !== "string") throw new Error("The server returned a malformed response without a route.");
    if (data.route === "grade") renderGrade(data);
    else if (data.route === "clarify") renderClarify(data);
    else if (data.route === "refuse") renderRefuse(data);
    else throw new Error(`The server returned an unknown route: ${data.route}`);
  } catch (error) {
    renderError(error);
  } finally {
    setBusy(false);
  }
}

async function copyTerm(text, chip) {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
    } else {
      const fallback = element("textarea");
      fallback.value = text;
      fallback.className = "visually-hidden";
      document.body.append(fallback);
      fallback.select();
      document.execCommand("copy");
      fallback.remove();
    }
    chip.classList.add("is-copied");
    const check = svgIcon("check");
    chip.append(check);
    showToast(`“${text}” copied`);
    window.setTimeout(() => {
      chip.classList.remove("is-copied");
      check.remove();
    }, COPIED_MS);
  } catch {
    showToast(`Copy unavailable: select “${text}” manually`);
  }
}

function showToast(message) {
  window.clearTimeout(state.toastTimer);
  dom.toast.textContent = message;
  dom.toast.hidden = false;
  state.toastTimer = window.setTimeout(() => {
    dom.toast.hidden = true;
  }, TOAST_MS);
}

function termForChip(chip) {
  return state.termRegistry.get(chip?.dataset.termKey);
}

function enrichedTerm(chip) {
  const resultTerm = termForChip(chip);
  if (!resultTerm) return null;
  const glossaryTerm = state.termMap.get(resultTerm.term.toLowerCase());
  return { ...resultTerm, ...(glossaryTerm || {}) };
}

function populatePopover(term) {
  const header = element("div", "popover__header");
  header.append(element("span", "popover__term", term.term));
  header.append(element("span", "popover__category", term.category || termBucket(term)));
  dom.popover.replaceChildren(header);
  dom.popover.append(element("p", "popover__definition", term.definition || "A grounded term in the grading vocabulary."));
  if (term.axis) dom.popover.append(element("p", "popover__axis", `Pipeline axis · ${term.axis}`));
  if (term.example_usage) dom.popover.append(element("p", "popover__example", `e.g. “${term.example_usage}”`));
}

function positionPopover(chip) {
  const chipRect = chip.getBoundingClientRect();
  const popRect = dom.popover.getBoundingClientRect();
  const gutter = 8;
  const gap = 12;
  const left = Math.max(gutter, Math.min(window.innerWidth - popRect.width - gutter, chipRect.left + (chipRect.width - popRect.width) / 2));
  let top = chipRect.top - popRect.height - gap;
  if (top < gutter) top = chipRect.bottom + gap;
  dom.popover.style.left = `${Math.round(left)}px`;
  dom.popover.style.top = `${Math.round(top)}px`;
}

function openPopover(chip, immediate = false) {
  window.clearTimeout(state.popoverTimer);
  const term = enrichedTerm(chip);
  if (!term) return;
  const show = () => {
    if (state.activePopoverChip && state.activePopoverChip !== chip) {
      state.activePopoverChip.removeAttribute("aria-describedby");
    }
    state.activePopoverChip = chip;
    chip.setAttribute("aria-describedby", "term-popover");  // only linked while actually shown
    populatePopover(term);
    dom.popover.hidden = false;
    positionPopover(chip);
  };
  if (immediate) show();
  else state.popoverTimer = window.setTimeout(show, POPOVER_DELAY_MS);
}

function closePopover(immediate = false) {
  window.clearTimeout(state.popoverTimer);
  const hide = () => {
    dom.popover.hidden = true;
    if (state.activePopoverChip) state.activePopoverChip.removeAttribute("aria-describedby");
    state.activePopoverChip = null;
  };
  if (immediate) hide();
  else state.popoverTimer = window.setTimeout(hide, 80);
}

function renderGlossary(filter = "") {
  const normalized = filter.trim().toLowerCase();
  const visible = state.terms.filter((term) => {
    if (!normalized) return true;
    return [term.term, term.axis, term.category, term.definition, term.example_usage]
      .some((value) => String(value || "").toLowerCase().includes(normalized));
  });
  dom.drawerCount.textContent = `${visible.length} ${visible.length === 1 ? "term" : "terms"}`;
  dom.glossary.replaceChildren();
  if (!visible.length) {
    dom.glossary.append(element("p", "glossary-empty", state.terms.length ? "No terms match that filter." : "The glossary is unavailable, but result terms will still include their definitions."));
    return;
  }

  const byCategory = new Map();
  visible.forEach((term) => {
    const category = term.category || "Other";
    if (!byCategory.has(category)) byCategory.set(category, []);
    byCategory.get(category).push(term);
  });
  [...byCategory.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .forEach(([category, terms]) => {
      const group = element("section", "glossary__group");
      group.append(element("h3", "glossary__group-title", category));
      terms.sort((a, b) => a.term.localeCompare(b.term)).forEach((term) => {
        const row = element("article", "glossary-row");
        const top = element("div", "glossary-row__top");
        const badge = element("span", `glossary-badge${term.grounded === true ? " glossary-badge--grounded" : ""}`, term.grounded === true ? "measured" : "reference");
        top.append(element("span", "glossary-row__term", term.term), badge);
        row.append(top, element("p", "glossary-row__definition", term.definition || "Definition unavailable."));
        if (term.axis) row.append(element("p", "glossary-row__axis", `axis · ${term.axis}`));
        if (term.example_usage) row.append(element("p", "glossary-row__example", `e.g. “${term.example_usage}”`));
        group.append(row);
      });
      dom.glossary.append(group);
    });
}

async function loadTerms() {
  try {
    let terms;
    if (MOCK_MODE) {
      terms = MOCK_TERMS;
    } else {
      const response = await fetch("/api/terms", { headers: { Accept: "application/json" } });
      if (!response.ok) throw new Error(`Glossary returned HTTP ${response.status}.`);
      terms = await response.json();
    }
    if (!Array.isArray(terms)) throw new Error("Glossary response was not a list.");
    state.terms = terms.filter((term) => term && typeof term.term === "string");
    state.termMap = new Map(state.terms.map((term) => [term.term.toLowerCase(), term]));
    renderGlossary();
  } catch (error) {
    console.warn("Chroma glossary unavailable:", error.message);
    state.terms = [];
    state.termMap.clear();
    renderGlossary();
  }
}

function openDrawer() {
  state.lastFocused = document.activeElement;
  dom.drawerShell.hidden = false;
  document.body.classList.add("is-drawer-open");
  dom.termSearch.value = "";
  renderGlossary();
  window.requestAnimationFrame(() => dom.termSearch.focus());
}

function closeDrawer() {
  if (dom.drawerShell.hidden) return;
  dom.drawerShell.hidden = true;
  document.body.classList.remove("is-drawer-open");
  if (state.lastFocused instanceof HTMLElement) state.lastFocused.focus();
}

function trapDrawerFocus(event) {
  if (dom.drawerShell.hidden || event.key !== "Tab") return;
  const focusable = $$("button:not(:disabled), input:not(:disabled), [href], [tabindex]:not([tabindex='-1'])", dom.drawer)
    .filter((node) => !node.hidden && node.offsetParent !== null);
  if (!focusable.length) return;
  const first = focusable[0];
  const last = focusable[focusable.length - 1];
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault();
    first.focus();
  }
}

function wireEvents() {
  dom.fileInput.addEventListener("change", () => selectFile(dom.fileInput.files?.[0]));
  dom.uploadZone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      dom.fileInput.click();
    }
  });
  dom.uploadClear.addEventListener("click", clearFile);
  dom.sampleImage.addEventListener("click", selectSampleImage);

  let dragDepth = 0;
  dom.uploadZone.addEventListener("dragenter", (event) => {
    event.preventDefault();
    dragDepth += 1;
    dom.uploadZone.classList.add("is-dragover");
  });
  dom.uploadZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
  });
  dom.uploadZone.addEventListener("dragleave", () => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) dom.uploadZone.classList.remove("is-dragover");
  });
  dom.uploadZone.addEventListener("drop", (event) => {
    event.preventDefault();
    dragDepth = 0;
    dom.uploadZone.classList.remove("is-dragover");
    selectFile(event.dataTransfer?.files?.[0]);
  });

  dom.prompt.addEventListener("input", () => {
    dom.promptCount.textContent = `${dom.prompt.value.length} / 280`;
    autoGrowPrompt();
    updateGenerateAvailability();
  });
  dom.prompt.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter" && !dom.generate.disabled) {
      event.preventDefault();
      generate();
    }
  });
  dom.generate.addEventListener("click", generate);

  document.addEventListener("mouseover", (event) => {
    const chip = event.target.closest?.("[data-term-key]");
    if (chip && !chip.contains(event.relatedTarget)) openPopover(chip);
  });
  document.addEventListener("mouseout", (event) => {
    const chip = event.target.closest?.("[data-term-key]");
    if (chip && !chip.contains(event.relatedTarget)) closePopover();
  });
  document.addEventListener("focusin", (event) => {
    const chip = event.target.closest?.("[data-term-key]");
    if (chip) openPopover(chip, true);
  });
  document.addEventListener("focusout", (event) => {
    const chip = event.target.closest?.("[data-term-key]");
    if (chip && !chip.contains(event.relatedTarget)) closePopover();
  });
  document.addEventListener("pointerdown", (event) => {
    const chip = event.target.closest?.("[data-term-key]");
    if (!chip || event.pointerType !== "touch") return;
    window.clearTimeout(state.longPressTimer);
    state.longPressTimer = window.setTimeout(() => openPopover(chip, true), LONG_PRESS_MS);
  });
  ["pointerup", "pointercancel"].forEach((name) => document.addEventListener(name, () => window.clearTimeout(state.longPressTimer)));
  window.addEventListener("wheel", () => closePopover(true), { passive: true });
  window.addEventListener("touchmove", () => closePopover(true), { passive: true });
  window.addEventListener("resize", () => closePopover(true));

  dom.termsOpen.addEventListener("click", openDrawer);
  dom.termsClose.addEventListener("click", closeDrawer);
  dom.drawerBackdrop.addEventListener("click", closeDrawer);
  dom.termSearch.addEventListener("input", () => renderGlossary(dom.termSearch.value));
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      closePopover(true);
      closeDrawer();
    }
    trapDrawerFocus(event);
  });
}

const MOCK_TERMS = [
  { term: "slight", axis: "(magnitude bucket)", category: "Magnitude", definition: "Keep the requested adjustment restrained and only gently visible.", example_usage: "slightly warmer", grounded: true },
  { term: "moderate", axis: "(magnitude bucket)", category: "Magnitude", definition: "Apply the adjustment at a clear but balanced strength.", example_usage: "moderate contrast", grounded: true },
  { term: "strong", axis: "(magnitude bucket)", category: "Magnitude", definition: "Make the requested adjustment unmistakable without changing its direction.", example_usage: "strong cool shadows", grounded: true },
  { term: "warmer", axis: "temperature_delta_b", category: "Temperature", definition: "Move the overall balance toward amber and away from blue.", example_usage: "slightly warmer", grounded: true },
  { term: "cooler", axis: "temperature_delta_b", category: "Temperature", definition: "Move the overall balance toward blue and away from amber.", example_usage: "moderately cooler", grounded: true },
  { term: "lifted_blacks", axis: "black_point_l_delta (+)", category: "Direction", definition: "Raise the darkest tones for a softer, less absolute shadow floor.", example_usage: "lifted blacks", grounded: true },
  { term: "more_contrast", axis: "contrast_l_spread_delta (+)", category: "Direction", definition: "Increase separation between darker and brighter tonal regions.", example_usage: "strongly more contrast", grounded: true },
  { term: "muted", axis: "chroma_delta (-)", category: "Direction", definition: "Reduce color intensity while preserving the scene’s overall hue relationships.", example_usage: "moderately muted color", grounded: true },
  { term: "teal-orange", axis: "composite (calibration window)", category: "Style reference", definition: "A documented composite balancing cooler cyan shadows against warmer highlights.", example_usage: "subtle teal-orange", grounded: false },
];

function mockRouteForPrompt() {
  if (MOCK_FORCED_ROUTE) return MOCK_FORCED_ROUTE;
  const prompt = dom.prompt.value.trim().toLowerCase();
  if (/(remove|replace|erase|relight|background)/.test(prompt)) return "refuse";
  if (prompt.length < 14 || /(make it pop|look good|better|cinematic)$/.test(prompt)) return "clarify";
  return "grade";
}

function mockReferenceSvg(index, graded) {
  const neutral = [
    ["#A9B0B4", "#4A555C", "#D5C9B5"], ["#AAB9C3", "#637768", "#D2C5A9"],
    ["#B99C88", "#604E49", "#D7C2AB"], ["#8D9A8B", "#4F5B50", "#C9B598"],
    ["#B9855D", "#6E493B", "#D4B788"], ["#B5AA9C", "#565B5B", "#D8D1C4"],
  ];
  const warm = [
    ["#C3A47F", "#314E55", "#E6BC83"], ["#9DB2B2", "#416C58", "#E2B66E"],
    ["#D19B74", "#3B5158", "#E9B77F"], ["#9DAB8A", "#315A50", "#DDB16C"],
    ["#D78B55", "#4D4B43", "#E6B56D"], ["#C4A17D", "#38565A", "#E1B97F"],
  ];
  const [sky, shadow, light] = (graded ? warm : neutral)[index];
  const shapes = [
    `<path d="M0 385h800v215H0z" fill="${shadow}"/><path d="M70 190h120v195H70zm145 85h85v110h-85zm125-145h150v255H340zm180 95h95v160h-95zm125-70h90v230h-90z" fill="${light}" opacity=".72"/>`,
    `<path d="m0 420 170-185 115 98 150-203 165 190 95-105 105 125v260H0z" fill="${shadow}"/><path d="m0 475 185-105 130 54 160-91 170 61 155-74v280H0z" fill="${light}" opacity=".62"/>`,
    `<ellipse cx="400" cy="278" rx="142" ry="176" fill="${light}"/><path d="M245 600c12-146 75-215 155-215s143 69 155 215H245Z" fill="${shadow}"/><path d="M283 252c18-126 202-163 239-28-54-41-165-38-239 28Z" fill="${shadow}"/>`,
    `<circle cx="260" cy="290" r="185" fill="${shadow}"/><circle cx="500" cy="250" r="145" fill="${light}" opacity=".82"/><circle cx="595" cy="440" r="96" fill="${shadow}" opacity=".75"/>`,
    `<ellipse cx="400" cy="350" rx="252" ry="175" fill="${light}"/><ellipse cx="400" cy="350" rx="188" ry="122" fill="${shadow}"/><circle cx="345" cy="320" r="52" fill="${sky}"/><circle cx="468" cy="375" r="60" fill="${light}"/>`,
    `<path d="M65 65h670v470H65z" fill="${light}" opacity=".45"/><path d="M65 375h670v160H65z" fill="${shadow}"/><path d="M105 105h260v220H105z" fill="${sky}"/><path d="M505 135h145v240H505z" fill="${shadow}" opacity=".78"/>`,
  ][index];
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop stop-color="${sky}"/><stop offset="1" stop-color="${light}"/></linearGradient></defs><path fill="url(#g)" d="M0 0h800v600H0z"/>${shapes}<path d="M0 0h800v600H0z" fill="none" stroke="${light}" stroke-opacity=".22" stroke-width="18"/></svg>`;
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
}

async function mockUserImages() {
  try {
    const image = new Image();
    image.src = state.fileUrl;
    await image.decode();
    const maxEdge = 1400;
    const scale = Math.min(1, maxEdge / Math.max(image.naturalWidth, image.naturalHeight));
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(image.naturalWidth * scale));
    canvas.height = Math.max(1, Math.round(image.naturalHeight * scale));
    const context = canvas.getContext("2d");
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    const original = canvas.toDataURL("image/jpeg", 0.9);
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.filter = "contrast(1.12) saturate(0.88) sepia(0.12) brightness(0.97)";
    context.drawImage(image, 0, 0, canvas.width, canvas.height);
    return { original, graded: canvas.toDataURL("image/jpeg", 0.9) };
  } catch {
    return { original: state.fileUrl, graded: state.fileUrl };
  }
}

function mockCubeUrl() {
  const cube = [
    "TITLE \"Chroma mock warm-neutral\"", "LUT_3D_SIZE 2", "DOMAIN_MIN 0.0 0.0 0.0", "DOMAIN_MAX 1.0 1.0 1.0",
    "0.000000 0.000000 0.000000", "0.050000 0.000000 0.950000", "0.000000 0.950000 0.050000", "0.050000 0.950000 1.000000",
    "0.950000 0.050000 0.000000", "1.000000 0.050000 0.950000", "0.950000 1.000000 0.050000", "1.000000 1.000000 1.000000",
  ].join("\n");
  const url = URL.createObjectURL(new Blob([cube], { type: "text/plain" }));
  state.mockObjectUrls.push(url);
  return url;
}

async function mockGenerate() {
  for (const url of state.mockObjectUrls.splice(0)) URL.revokeObjectURL(url);
  await new Promise((resolve) => window.setTimeout(resolve, 650));
  const route = mockRouteForPrompt();
  if (route === "error") {
    const error = new Error("Opt-in mock failure for error-state verification.");
    error.code = "generation_failure";
    throw error;
  }
  const base = {
    route,
    refuse_reason: null,
    clarify_message: null,
    attribute_spec_text: null,
    lut: null,
    previews: [],
    prompt_feedback: { assessment: "", suggested_terms: [] },
    quality: null,
  };
  if (route === "clarify") {
    base.clarify_message = `“${dom.prompt.value.trim()}” could go a few ways. Tell me which direction you mean and how strongly it should read.`;
    base.prompt_feedback = {
      assessment: "The request needs a measurable direction and strength.",
      suggested_terms: MOCK_TERMS.filter((term) => ["slight", "moderate", "strong", "warmer", "cooler", "more_contrast"].includes(term.term)),
    };
    return base;
  }
  if (route === "refuse") {
    base.refuse_reason = "out_of_scope";
    base.prompt_feedback = { assessment: "A LUT cannot isolate or alter scene content.", suggested_terms: [] };
    return base;
  }

  const userImages = await mockUserImages();
  const previews = [{ name: "user_image", original_url: userImages.original, graded_url: userImages.graded }];
  REFERENCE_NAMES.forEach((name, index) => {
    previews.push({ name, original_url: mockReferenceSvg(index, false), graded_url: mockReferenceSvg(index, true) });
  });
  return {
    ...base,
    route: "grade",
    attribute_spec_text: "route=grade | warmer=+2.0 more_contrast=+1.5 lifted_blacks=+0.8",
    lut: { cube_url: mockCubeUrl() },
    previews,
    prompt_feedback: {
      assessment: "The direction is clear; adding an explicit strength would make the intended magnitude more repeatable.",
      suggested_terms: MOCK_TERMS.filter((term) => ["slight", "moderate", "strong", "lifted_blacks", "more_contrast", "muted", "teal-orange"].includes(term.term)),
    },
    quality: { behavioral_fidelity: 0.91, collapsed: false, fell_back_greedy: false },
  };
}

function init() {
  dom.modeBadge.hidden = !MOCK_MODE;
  wireEvents();
  autoGrowPrompt();
  updateGenerateAvailability();
  loadTerms();
}

init();
