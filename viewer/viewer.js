/* EvalLens offline viewer.
 *
 * Reads a run's record.json and renders exactly what it contains. Two rules govern every
 * function below:
 *
 *   1. A missing field renders as "not recorded", never as a plausible-looking default.
 *      A viewer that invents a number is worse than one that shows nothing.
 *   2. Status is carried by a text label as well as a color, so the display works without
 *      color vision.
 */

"use strict";

const $ = (id) => document.getElementById(id);

function show(id) {
  for (const section of ["loading", "empty", "error", "content"]) {
    $(section).classList.toggle("hidden", section !== id);
  }
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]
  );
}

/** Render a value, or an explicit "not recorded" marker when it is absent. */
function field(value, { mono = true } = {}) {
  if (value === undefined || value === null || value === "") {
    return '<span class="muted">not recorded</span>';
  }
  const text = escapeHtml(value);
  return mono ? text : text;
}

function num(value, digits = 6) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return '<span class="muted">not recorded</span>';
  }
  if (value !== 0 && (Math.abs(value) < 1e-3 || Math.abs(value) >= 1e5)) {
    return value.toExponential(digits - 3);
  }
  return String(Number(value.toFixed(digits)));
}

function kv(pairs) {
  const rows = pairs
    .map(([key, value]) => `<dt>${escapeHtml(key)}</dt><dd>${value}</dd>`)
    .join("");
  return `<dl class="kv">${rows}</dl>`;
}

function verdictPill(verdict) {
  const value = (verdict || "unknown").toLowerCase();
  const cls = value === "pass" ? "pass" : value === "fail" ? "fail" : "other";
  return `<span class="pill ${cls}">${escapeHtml(value.toUpperCase())}</span>`;
}

function stat(label, value) {
  return `<div class="stat"><span class="label">${escapeHtml(label)}</span>
    <span class="value">${value}</span></div>`;
}

/* --- sections ------------------------------------------------------------------------- */

function renderBanners(record) {
  let html = "";
  if (record.injected_fault) {
    html += `<div class="banner injected">
      <strong>Deliberately injected fault</strong>
      ${escapeHtml(record.fault_banner || "This fault was injected by EvalLens on purpose.")}
      ${record.fault_description ? `<br><br>${escapeHtml(record.fault_description)}` : ""}
    </div>`;
  }
  return html;
}

function renderVerdict(record) {
  const comparison = record.comparison || {};
  const policy = record.policy || {};
  const detection = (record.detection && record.detection.data) || {};

  return `<section class="panel">
    <h2>Verdict</h2>
    <div class="verdict-row">
      ${verdictPill(comparison.verdict)}
      <span class="muted">${field(comparison.detail)}</span>
    </div>
    <div class="stats">
      ${stat("max |Δ|", num(comparison.max_abs_err))}
      ${stat("atol", num(policy.atol, 8))}
      ${stat("rtol", num(policy.rtol, 8))}
      ${stat("cases examined", field(detection.cases_examined))}
      ${stat("time to detect", detection.seconds_to_detection !== undefined
        ? `${num(detection.seconds_to_detection, 3)}s` : field(undefined))}
    </div>
  </section>`;
}

function renderIdentities(record) {
  const config = record.model_config || {};
  const generator = record.generator || {};
  return `<section class="panel">
    <h2>Reference and candidate</h2>
    <p class="lead">Both run the same weights on the same inputs. Only the implementation differs.</p>
    ${kv([
      ["reference", field(record.reference_adapter)],
      ["candidate", field(record.candidate_adapter)],
      ["difference", field(record.candidate_behavior_description)],
      ["model config", field(config.config_id)],
      ["weights sha256", field(record.weights_sha256)],
      ["tolerance policy", `${field(policyName(record))} [${field((record.policy || {}).policy_id)}]`],
      ["generator", `${field(generator.name)}@${field(generator.version)} seed=${field(generator.seed)} budget=${field(generator.budget)}`],
    ])}
  </section>`;
}

function policyName(record) {
  return (record.policy || {}).name;
}

function renderCase(title, lead, caseObj) {
  if (!caseObj) {
    return `<section class="panel"><h2>${escapeHtml(title)}</h2>
      <p class="muted">not recorded</p></section>`;
  }
  const requests = (caseObj.requests || [])
    .map((request) => {
      const tokens = (request.token_ids || [])
        .map((token, index) => {
          const inPrefill = index < request.prefix_length;
          const cls = inPrefill ? "token prefill" : "token";
          const title = inPrefill ? "prefill" : "decode step";
          return `<span class="${cls}" title="position ${index}, ${title}">${escapeHtml(token)}</span>`;
        })
        .join("");
      return `<div class="request-block">
        <h3>${escapeHtml(request.request_id)} — ${request.token_ids.length} token(s),
          prefill ${escapeHtml(request.prefix_length)}, pad_left ${escapeHtml(request.pad_left)}</h3>
        <div class="tokens">${tokens}</div>
      </div>`;
    })
    .join("");

  const totalTokens = (caseObj.requests || []).reduce((sum, r) => sum + r.token_ids.length, 0);
  return `<section class="panel">
    <h2>${escapeHtml(title)}</h2>
    <p class="lead">${escapeHtml(lead)}</p>
    ${kv([
      ["case id", field(caseObj.case_id)],
      ["execution mode", field(caseObj.execution_mode)],
      ["category", field(caseObj.category)],
      ["requests / tokens", `${(caseObj.requests || []).length} / ${totalTokens}`],
    ])}
    <div style="margin-top:12px">${requests}</div>
    <p class="muted" style="margin-top:8px;font-size:12px">
      Outlined tokens are consumed in the prefill call; the rest are individual decode steps.
    </p>
  </section>`;
}

function renderLocalization(record) {
  const localization = record.localization;
  if (!localization) {
    return `<section class="panel"><h2>Earliest observed divergence</h2>
      <p class="muted">not recorded</p></section>`;
  }
  if (!localization.available) {
    return `<section class="panel">
      <h2>Earliest observed divergence</h2>
      <div class="verdict-row"><span class="pill other">UNAVAILABLE</span>
        <span class="muted">${field(localization.reason)}</span></div>
      <p class="muted">The output-level failure stands on its own. Nothing is guessed here.</p>
    </section>`;
  }

  const comparisons = localization.comparisons || [];
  const earliest = localization.earliest_observed_str;
  const rows = comparisons
    .map((comparison) => {
      const classes = [];
      if (comparison.diverged) classes.push("diverged");
      if (comparison.address_str === earliest) classes.push("earliest");
      const diff = comparison.diff || {};
      return `<tr class="${classes.join(" ")}">
        <td class="mono">${escapeHtml(comparison.address_str)}</td>
        <td>${comparison.diverged ? "diverged" : comparison.values_available ? "within tolerance" : "values dropped"}</td>
        <td class="mono">${num(diff.max_abs_err)}</td>
        <td class="mono">${diff.n_violations !== undefined ? `${diff.n_violations}/${diff.n_elements}` : field(undefined)}</td>
      </tr>`;
    })
    .join("");

  const reconverged = localization.reconverged
    ? `<div class="banner note"><strong>Discrepancy reconverges</strong>
       At least one aligned checkpoint returns inside tolerance after an earlier one left it.
       The "diverged" predicate is not monotone along this traversal, which is why every
       aligned checkpoint is compared rather than bisected.</div>`
    : "";

  return `<section class="panel">
    <h2>Earliest observed divergence</h2>
    <p class="lead">${escapeHtml(localization.interpretation || "")}</p>
    ${kv([
      ["earliest observed", field(earliest)],
      ["divergent checkpoints", `${localization.n_divergent} of ${localization.n_compared}`],
      ["divergent layers", field((localization.divergent_layers || []).join(", "))],
      ["aligned", `${localization.alignment ? localization.alignment.n_matched : "?"} checkpoints, fully aligned: ${localization.alignment ? localization.alignment.fully_aligned : "?"}`],
    ])}
    ${reconverged}
    <div class="scroll" style="margin-top:12px">
      <table>
        <thead><tr><th class="mono">checkpoint</th><th>status</th><th>max |Δ|</th><th>violations</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
    ${localization.comparisons_truncated
      ? '<p class="muted" style="font-size:12px">Checkpoint list truncated by the record\'s capture budget.</p>'
      : ""}
  </section>`;
}

function renderReduction(record) {
  const reduction = record.reduction;
  if (!reduction) {
    return `<section class="panel"><h2>Reduction</h2><p class="muted">not recorded</p></section>`;
  }
  const counters = reduction.counters || {};
  const steps = (reduction.steps || []).filter((step) => step.accepted);
  const rows = steps
    .map(
      (step) => `<tr>
        <td class="mono">${escapeHtml(step.operation)}</td>
        <td class="mono">${escapeHtml(step.strategy)}</td>
        <td class="mono">${formatSize(step.before)} → ${formatSize(step.after)}</td>
        <td class="mono">${escapeHtml(step.queries_at_step)}</td>
      </tr>`
    )
    .join("");

  const minimal = reduction.minimality === "one_minimal_wrt_declared_operations";
  return `<section class="panel">
    <h2>Reduction</h2>
    <p class="lead">Size is ordered lexicographically: requests, valid tokens, padding tokens,
      token-value complexity. Every accepted step must strictly decrease it.</p>
    <div class="stats">
      ${stat("token ratio", reduction.token_reduction_ratio ? `${reduction.token_reduction_ratio.toFixed(1)}x` : field(undefined))}
      ${stat("logical queries", field(counters.logical_queries))}
      ${stat("model runs", field(counters.model_runs))}
      ${stat("cache hits", field(counters.cache_hits))}
      ${stat("accepted steps", String(steps.length))}
      ${stat("wall time", reduction.wall_time_s !== undefined ? `${reduction.wall_time_s}s` : field(undefined))}
    </div>
    <div class="verdict-row" style="margin-top:14px">
      <span class="pill ${minimal ? "info" : "other"}">${minimal ? "1-MINIMAL" : "MINIMALITY NOT ESTABLISHED"}</span>
      <span class="muted">${field(reduction.minimality_note)}</span>
    </div>
    <div class="scroll" style="margin-top:12px">
      <table>
        <thead><tr><th>operation</th><th>strategy</th><th>size</th><th>queries</th></tr></thead>
        <tbody>${rows || '<tr><td colspan="4" class="muted">no accepted steps</td></tr>'}</tbody>
      </table>
    </div>
  </section>`;
}

function formatSize(size) {
  if (!size) return "?";
  return `(${size.n_requests}, ${size.n_valid_tokens}, ${size.n_padding_tokens})`;
}

function renderExport(record) {
  const exported = record.export || {};
  const verification = exported.verification || {};
  const replay = record.subprocess_replay || {};
  const ok = exported.verified === true;
  return `<section class="panel">
    <h2>Reproduction</h2>
    <p class="lead">A self-contained package, verified by running it from a fresh temporary
      directory in an isolated interpreter.</p>
    <div class="verdict-row">
      <span class="pill ${ok ? "pass" : "other"}">${ok ? "VERIFIED" : "NOT VERIFIED"}</span>
      <span class="muted">${ok
        ? "reproduced the recorded mismatch without importing EvalLens"
        : "verification was skipped or did not succeed"}</span>
    </div>
    ${kv([
      ["package", field(exported.path)],
      ["exit code", `${field(verification.exit_code)} (expected ${field(verification.expected_exit_code)})`],
      ["imported evallens", verification.imported_evallens === undefined
        ? field(undefined) : String(verification.imported_evallens)],
      ["ran from", field(verification.ran_from)],
      ["fresh-process replay", `${field(replay.verdict)}, stable=${field(replay.stable)}`],
    ])}
    <p class="muted" style="margin-top:10px;font-size:12px">
      Run it yourself: <code>python ${escapeHtml(exported.path || "repro")}/repro.py --expect-mismatch</code>
    </p>
  </section>`;
}

function renderTimeline(record) {
  const steps = record.steps || [];
  const items = steps
    .map(
      (step) => `<li>
        <span class="elapsed">${step.elapsed_s.toFixed(2)}s</span>
        <span class="step">${escapeHtml(step.name)}</span>
        <span class="detail">${escapeHtml(step.detail)}</span>
      </li>`
    )
    .join("");
  return `<section class="panel">
    <h2>Run timeline</h2>
    <p class="lead">Measured during this run. Diagnostic capture is a separate pass and is
      never inside the detection timing.</p>
    <ul class="timeline">${items || '<li class="muted">no steps recorded</li>'}</ul>
  </section>`;
}

function renderEnvironment(record) {
  const env = record.environment || {};
  const resources = record.resources || {};
  return `<section class="panel">
    <h2>Environment</h2>
    ${kv([
      ["python", field(env.python_version)],
      ["torch / numpy", `${field(env.torch_version)} / ${field(env.numpy_version)}`],
      ["platform", field(env.platform)],
      ["chip / cpus", `${field(env.chip)} / ${field(env.cpu_count)}`],
      ["torch threads", field(env.torch_num_threads)],
      ["peak RSS", resources.peak_rss_mib !== undefined && resources.peak_rss_mib !== null
        ? `${resources.peak_rss_mib} MiB of ${resources.limit_mib} MiB budget` : field(undefined)],
      ["git commit", field(env.git_commit)],
      ["working tree", env.git_dirty === undefined ? field(undefined) : (env.git_dirty ? "dirty" : "clean")],
      ["run id", field(record.run_id)],
    ])}
  </section>`;
}

/* --- entry point ------------------------------------------------------------------------ */

function render(record) {
  $("content").innerHTML = [
    renderBanners(record),
    renderVerdict(record),
    renderIdentities(record),
    renderCase("Original failing input", "The case the generator produced, before reduction.", record.original_case),
    renderCase("Reduced input", "The output of the actual reducer. Not handwritten.", record.reduced_case),
    renderLocalization(record),
    renderReduction(record),
    renderExport(record),
    renderTimeline(record),
    renderEnvironment(record),
  ].join("");
  show("content");
}

function loadRecord() {
  show("loading");
  fetch("run/record.json", { cache: "no-store" })
    .then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    })
    .then((record) => {
      if (!record || typeof record !== "object" || !record.run_id) {
        throw new Error("record.json does not look like an EvalLens run record");
      }
      render(record);
    })
    .catch((error) => {
      if (String(error).includes("HTTP 404")) {
        show("empty");
        return;
      }
      $("error-detail").textContent = String(error);
      show("error");
    });
}

$("reload").addEventListener("click", loadRecord);

$("filepick").addEventListener("change", (event) => {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => {
    try {
      render(JSON.parse(String(reader.result)));
    } catch (error) {
      $("error-detail").textContent = `${file.name}: ${error}`;
      show("error");
    }
  };
  reader.onerror = () => {
    $("error-detail").textContent = `could not read ${file.name}`;
    show("error");
  };
  reader.readAsText(file);
});

loadRecord();
