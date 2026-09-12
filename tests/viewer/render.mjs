/* Headless render harness for the EvalLens viewer.
 *
 * No browser-automation extension was available on this machine, so the viewer could not be
 * confirmed by screenshot. This harness is the next best thing that is still *execution*:
 * it runs the real `viewer/viewer.js` against a real `record.json` under a minimal DOM shim
 * and prints what the page would contain.
 *
 * The shim deliberately serves only the element ids that actually appear in `index.html`,
 * and returns null for anything else. That is what makes this more than a smoke test: if
 * viewer.js reaches for an element the markup does not define, the render throws here
 * instead of silently producing a blank panel in a browser.
 *
 * Usage: node render.mjs <record.json|--missing|--corrupt>
 * Prints a JSON object: { section, html, error }.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

const here = dirname(fileURLToPath(import.meta.url));
const viewerDir = resolve(here, "..", "..", "viewer");
const indexHtml = readFileSync(resolve(viewerDir, "index.html"), "utf8");
const viewerJs = readFileSync(resolve(viewerDir, "viewer.js"), "utf8");

// The set of ids the markup really defines. Anything outside it is a bug in viewer.js.
const declaredIds = new Set([...indexHtml.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1]));

class ClassList {
  constructor() {
    this.names = new Set();
  }
  toggle(name, force) {
    if (force === undefined) {
      this.names.has(name) ? this.names.delete(name) : this.names.add(name);
    } else if (force) {
      this.names.add(name);
    } else {
      this.names.delete(name);
    }
    return this.names.has(name);
  }
  contains(name) {
    return this.names.has(name);
  }
}

class Element {
  constructor(id) {
    this.id = id;
    this.classList = new ClassList();
    this.innerHTML = "";
    this.textContent = "";
    this.listeners = new Map();
  }
  addEventListener(type, handler) {
    this.listeners.set(type, handler);
  }
}

const elements = new Map();
const document = {
  getElementById(id) {
    if (!declaredIds.has(id)) {
      throw new Error(`viewer.js asked for #${id}, which index.html does not define`);
    }
    if (!elements.has(id)) elements.set(id, new Element(id));
    return elements.get(id);
  },
};

const arg = process.argv[2];
let fetchImpl;
if (arg === "--missing") {
  fetchImpl = async () => ({ ok: false, status: 404, json: async () => ({}) });
} else if (arg === "--corrupt") {
  fetchImpl = async () => ({
    ok: true,
    status: 200,
    json: async () => ({ not: "a record" }),
  });
} else {
  const record = JSON.parse(readFileSync(resolve(arg), "utf8"));
  fetchImpl = async () => ({ ok: true, status: 200, json: async () => record });
}

const context = vm.createContext({
  document,
  fetch: fetchImpl,
  FileReader: class {},
  console,
  Math,
  Number,
  String,
  JSON,
  Object,
  Array,
  Error,
  Promise,
  Set,
  Map,
});

let error = null;
try {
  vm.runInContext(viewerJs, context, { filename: "viewer.js" });
  // loadRecord() runs at load and settles a promise chain; let the microtasks drain.
  await new Promise((r) => setTimeout(r, 0));
  await new Promise((r) => setTimeout(r, 0));
} catch (exc) {
  error = String(exc && exc.message ? exc.message : exc);
}

const visible = ["loading", "empty", "error", "content"].filter(
  (id) => elements.has(id) && !elements.get(id).classList.contains("hidden")
);

process.stdout.write(
  JSON.stringify({
    section: visible.length === 1 ? visible[0] : visible,
    html: elements.has("content") ? elements.get("content").innerHTML : "",
    errorDetail: elements.has("error-detail") ? elements.get("error-detail").textContent : "",
    error,
    declaredIds: [...declaredIds].sort(),
  })
);
