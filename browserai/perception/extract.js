// In-page DOM extractor. Evaluated by Playwright in every frame as `(opts) => {...}`.
//
// Walks the DOM (descending into open shadow roots), finds elements a user could
// interact with, filters out ones that are not actually visible, and tags each
// one with a stable `data-bai-id` attribute so Python can locate it again later.
//
// Returns a plain JSON object; coordinates are relative to THIS frame's viewport
// (Python adds the iframe offset).
(opts) => {
  const MAX = (opts && opts.maxElements) || 400;
  const W = window;

  // Per-document state survives between snapshots (until navigation replaces the document).
  if (!W.__bai) {
    W.__bai = { next: 1, docId: Math.random().toString(36).slice(2, 10) };
  }
  const state = W.__bai;

  const vw = W.innerWidth;
  const vh = W.innerHeight;

  const SKIP_TAGS = new Set([
    "SCRIPT", "STYLE", "NOSCRIPT", "TEMPLATE", "HEAD", "META", "LINK", "IFRAME", "FRAME", "OBJECT", "EMBED",
  ]);
  const ROLES = new Set([
    "button", "link", "checkbox", "radio", "switch", "tab", "menuitem", "menuitemcheckbox",
    "menuitemradio", "option", "combobox", "textbox", "searchbox", "listbox", "slider",
    "spinbutton", "treeitem", "gridcell",
  ]);
  const TEXTLIKE_INPUTS = new Set([
    "text", "email", "password", "search", "tel", "url", "number", "date", "datetime-local",
    "month", "week", "time", "color", "range", "",
  ]);

  const clean = (s, n = 100) => {
    if (!s) return "";
    s = String(s).replace(/\s+/g, " ").trim();
    return s.length > n ? s.slice(0, n - 1) + "…" : s;
  };
  const textOf = (node) => clean(node.innerText ?? node.textContent ?? "", 200);

  // Parent that crosses shadow-root boundaries.
  const composedParent = (el) => {
    if (el.parentElement) return el.parentElement;
    const root = el.getRootNode();
    return root && root.host ? root.host : null;
  };
  const composedContains = (ancestor, node) => {
    for (let n = node; n; n = n.parentNode || (n.host ?? null)) {
      if (n === ancestor) return true;
    }
    return false;
  };

  function isInteractive(el, style) {
    const tag = el.tagName;
    if (tag === "INPUT") return el.type !== "hidden";
    if (tag === "BUTTON" || tag === "SELECT" || tag === "TEXTAREA" || tag === "SUMMARY") return true;
    if (tag === "A" && el.hasAttribute("href")) return true;
    if (el.isContentEditable) {
      const p = composedParent(el);
      return !(p && p.isContentEditable); // only the editable root, not every child
    }
    const role = el.getAttribute("role");
    if (role && ROLES.has(role)) return true;
    if (el.hasAttribute("onclick")) return true;
    const ti = el.getAttribute("tabindex");
    if (ti !== null && parseInt(ti, 10) >= 0) return true;
    if (style.cursor === "pointer") {
      // `cursor` is inherited: only count the element where it starts.
      const p = composedParent(el);
      return !p || getComputedStyle(p).cursor !== "pointer";
    }
    return false;
  }

  function visibleRect(el, style, allowTransparent = false) {
    if (style.visibility === "hidden" || style.visibility === "collapse") return null;
    if (!allowTransparent && parseFloat(style.opacity) === 0) return null;
    if (el.checkVisibility && !el.checkVisibility({ checkOpacity: !allowTransparent, checkVisibilityCSS: true })) {
      return null;
    }
    const r = el.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) return null;
    return r;
  }

  // Is something else painted on top of the element's center (e.g. a modal backdrop)?
  function isOccluded(el, r) {
    const cx = r.left + r.width / 2;
    const cy = r.top + r.height / 2;
    if (cx < 0 || cy < 0 || cx >= vw || cy >= vh) return false; // cannot test off-screen
    let top = document.elementFromPoint(cx, cy);
    while (top && top.shadowRoot) {
      const inner = top.shadowRoot.elementFromPoint(cx, cy);
      if (!inner || inner === top) break;
      top = inner;
    }
    if (!top) return false;
    // Hit the element itself, a child of it, or an ancestor (element has pointer-events:none).
    return !(composedContains(el, top) || composedContains(top, el));
  }

  function accessibleName(el) {
    const aria = el.getAttribute("aria-label");
    if (aria && aria.trim()) return clean(aria);

    const labelledBy = el.getAttribute("aria-labelledby");
    if (labelledBy) {
      const root = el.getRootNode();
      const t = labelledBy
        .split(/\s+/)
        .map((id) => (root.getElementById ? root.getElementById(id) : document.getElementById(id)))
        .filter(Boolean)
        .map(textOf)
        .join(" ");
      if (t.trim()) return clean(t);
    }

    if (el.labels && el.labels.length) {
      const t = Array.from(el.labels).map(textOf).join(" ");
      if (t.trim()) return clean(t);
    }

    const tag = el.tagName;
    if (tag === "INPUT" && ["submit", "button", "reset"].includes(el.type) && el.value) return clean(el.value);
    if (tag === "INPUT" && el.type === "image" && el.alt) return clean(el.alt);

    if (!["INPUT", "TEXTAREA", "SELECT"].includes(tag)) {
      const t = textOf(el);
      if (t) return clean(t);
      const img = el.querySelector("img[alt]");
      if (img && img.alt.trim()) return clean(img.alt);
      const svgTitle = el.querySelector("svg title");
      if (svgTitle && svgTitle.textContent.trim()) return clean(svgTitle.textContent);
    }

    const title = el.getAttribute("title");
    if (title && title.trim()) return clean(title);
    const ph = el.getAttribute("placeholder");
    if (ph && ph.trim()) return clean(ph);

    // Unlabeled checkbox/radio: use the text right next to it (sibling label/span, or the
    // surrounding list item / row), which is what a sighted user reads as its label.
    if (tag === "INPUT" && (el.type === "checkbox" || el.type === "radio")) {
      for (const sib of [el.nextElementSibling, el.previousElementSibling]) {
        const t = sib && textOf(sib);
        if (t) return clean(t, 80);
      }
      const container = el.closest("li, tr, [role=row], [role=listitem]") || el.parentElement;
      const t = container && textOf(container);
      if (t) return clean(t, 80);
    }
    return "";
  }

  // A short "kind" label for the model: link, button, checkbox, input[email], select, ...
  function kindOf(el) {
    const role = el.getAttribute("role");
    if (role && ROLES.has(role)) return role;
    const tag = el.tagName;
    if (tag === "A") return "link";
    if (tag === "BUTTON" || tag === "SUMMARY") return "button";
    if (tag === "SELECT") return el.multiple ? "multiselect" : "select";
    if (tag === "TEXTAREA") return "textarea";
    if (tag === "INPUT") {
      const t = (el.type || "text").toLowerCase();
      if (t === "checkbox" || t === "radio") return t;
      if (["submit", "button", "reset", "image"].includes(t)) return "button";
      if (t === "file") return "file";
      return `input[${t}]`;
    }
    if (el.isContentEditable) return "editable";
    return "clickable";
  }

  function describe(el, r, viaLabel) {
    if (!el.__baiId) el.__baiId = state.next++;
    el.setAttribute("data-bai-id", String(el.__baiId));

    const tag = el.tagName;
    const kind = kindOf(el);
    const item = {
      bai_id: el.__baiId,
      tag: tag.toLowerCase(),
      kind,
      name: accessibleName(el),
      box: {
        x: Math.round(r.left),
        y: Math.round(r.top),
        w: Math.round(r.width),
        h: Math.round(r.height),
      },
      in_viewport: r.bottom > 0 && r.right > 0 && r.top < vh && r.left < vw,
      occluded: false,
      via_label: viaLabel,
      in_shadow: !!(el.getRootNode() && el.getRootNode().host),
    };
    item.occluded = item.in_viewport && !viaLabel && isOccluded(el, r);

    const disabled = el.disabled === true || el.getAttribute("aria-disabled") === "true";
    if (disabled) item.disabled = true;

    const expanded = el.getAttribute("aria-expanded");
    if (expanded !== null) item.expanded = expanded === "true";

    const popup = el.getAttribute("aria-haspopup");
    if (popup && popup !== "false" && kind !== "combobox") item.popup = popup === "true" ? "menu" : popup;

    // Validation state: explicit aria-invalid, or native constraint failures the user has
    // already triggered (:user-invalid only matches after interaction / a submit attempt).
    let invalid = el.getAttribute("aria-invalid") === "true";
    if (!invalid && el.validity && !el.validity.valid) {
      try { invalid = el.matches(":user-invalid"); } catch (e) { /* older engines */ }
    }
    if (invalid) {
      item.invalid = true;
      let msg = el.validationMessage || "";
      const describedBy = el.getAttribute("aria-describedby") || el.getAttribute("aria-errormessage");
      if (describedBy) {
        const root = el.getRootNode();
        const t = describedBy.split(/\s+/)
          .map((id) => (root.getElementById ? root.getElementById(id) : null))
          .filter(Boolean).map(textOf).join(" ");
        if (t.trim()) msg = t;
      }
      if (msg) item.error = clean(msg, 100);
    }

    if (kind === "checkbox" || kind === "radio" || kind === "switch") {
      item.checked = tag === "INPUT" ? el.checked : el.getAttribute("aria-checked") === "true";
    }

    if (tag === "INPUT" || tag === "TEXTAREA") {
      const t = (el.type || "").toLowerCase();
      if (tag === "TEXTAREA" || TEXTLIKE_INPUTS.has(t)) {
        // Never echo real passwords back into the model context.
        item.value = t === "password" ? (el.value ? "••••••" : "") : clean(el.value, 120);
      }
      const ph = el.getAttribute("placeholder");
      if (ph) item.placeholder = clean(ph, 80);
      if (el.required) item.required = true;

      // Fields the agent must never fill (enforced in Python by the safety layer).
      const ac = (el.getAttribute("autocomplete") || "").toLowerCase();
      const ident = `${el.name || ""} ${el.id || ""} ${item.name}`.toLowerCase();
      if (t === "password" || ac.includes("password")) item.sensitive = "password";
      else if (ac.startsWith("cc-") || /card.?num|cvv|cvc|security.?code|expir|iban|routing|account.?num/.test(ident))
        item.sensitive = "payment";
      else if (ac === "one-time-code" || /\b(otp|one.?time|verification code|2fa)\b/.test(ident))
        item.sensitive = "otp";
    }

    if (tag === "SELECT") {
      item.options = Array.from(el.options).slice(0, 15).map((o) => clean(o.text, 40));
      if (el.options.length > 15) item.options_total = el.options.length;
      item.value = Array.from(el.selectedOptions).map((o) => clean(o.text, 40)).join(", ");
    }

    if (tag === "A") {
      const href = el.getAttribute("href") || "";
      if (href && !href.startsWith("javascript:")) item.href = clean(href, 120);
      if (el.target === "_blank" && !el.hasAttribute("download")) item.new_tab = true;
    }

    if (el.isContentEditable) item.value = clean(el.innerText, 120);
    return item;
  }

  // ---- Walk the DOM in document order, descending into open shadow roots. ----
  const found = [];
  const stack = [document.documentElement];
  while (stack.length) {
    const el = stack.pop();
    if (!el || el.nodeType !== 1 || SKIP_TAGS.has(el.tagName)) continue;

    const style = getComputedStyle(el);
    if (style.display === "none") continue; // nothing inside can be visible

    if (isInteractive(el, style)) {
      let r = visibleRect(el, style);
      let viaLabel = false;
      // Custom-styled checkboxes/radios/file inputs are often invisible; the label is the click target.
      if (!r && el.tagName === "INPUT" && el.labels && el.labels.length) {
        const label = el.labels[0];
        const lr = visibleRect(label, getComputedStyle(label));
        if (lr) {
          r = lr;
          viaLabel = true;
        }
      }
      // No label: a transparent (opacity:0) checkbox/radio/file input still receives clicks,
      // e.g. TodoMVC's item toggles drawn as a background image next to the input.
      if (!r && el.tagName === "INPUT" && ["checkbox", "radio", "file"].includes(el.type)) {
        r = visibleRect(el, style, true);
      }
      if (r) found.push(describe(el, r, viaLabel));
    }

    if (el.tagName === "svg" || el.tagName === "SVG") continue; // don't descend into SVG internals

    const kids = [];
    if (el.shadowRoot) kids.push(...el.shadowRoot.children);
    kids.push(...el.children);
    for (let i = kids.length - 1; i >= 0; i--) stack.push(kids[i]);
  }

  // Keep the element budget for what matters most: on-screen elements first, then document order.
  let elements = found;
  const truncated = found.length > MAX;
  if (truncated) {
    const onScreen = found.filter((e) => e.in_viewport);
    const offScreen = found.filter((e) => !e.in_viewport);
    elements = onScreen.concat(offScreen).slice(0, MAX);
    const order = new Map(found.map((e, i) => [e, i]));
    elements.sort((a, b) => order.get(a) - order.get(b));
  }

  const se = document.scrollingElement || document.documentElement;
  return {
    doc_id: state.docId,
    url: location.href,
    title: document.title,
    viewport: { w: vw, h: vh },
    scroll: { x: Math.round(W.scrollX), y: Math.round(W.scrollY) },
    page: { w: se.scrollWidth, h: se.scrollHeight },
    elements,
    // Visible text, used only on the Python side to detect what changed after an action.
    text: document.body ? document.body.innerText.slice(0, 50000) : "",
    total_found: found.length,
    truncated,
  };
}
