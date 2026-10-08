// Placebo: shared page code. Text goes in through textContent, never parsed as HTML.
const $ = (id) => document.getElementById(id);
const pct = (x) => (x * 100).toFixed(1) + "%";
const f2 = (x) => (x == null ? "-" : x.toFixed(2));
const when = (t) => new Date(t).toISOString().slice(5, 16).replace("T", " ");
const whenFull = (t) => new Date(t).toISOString().slice(0, 16).replace("T", " ") + " UTC";
const SVG = "http://www.w3.org/2000/svg";
// Build DOM nodes; text always goes in through textContent, never parsed as HTML.
function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) k === "style" ? Object.assign(el.style, v) : el.setAttribute(k, v);
  for (const c of kids.flat()) el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return el;
}
function s(tag, attrs = {}, text) {
  const el = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (text != null) el.textContent = text;
  return el;
}
const MONO = { "font-family": "IBM Plex Mono, monospace", "font-size": "14" };
const HEADS = ["arm", "Sharpe", "return", "max DD", "win", "trades/day", "costs"];
const VERDICT = {
  "DEPLOY": (p, c) => `Deploy. It beat ${p}% of its placebo twins, beat the momentum rule, and agreed with itself ${c}% of the time.`,
  "HALF SIZE": (p) => `Half size at most. Better than ${p}% of placebo twins, which is suggestive, not proof.`,
  "DO NOT DEPLOY": (p) => `Do not deploy. It beat only ${p}% of its placebo twins. A coin flip with the same timing does about as well.`,
};
function state(p) {
  if (p <= 0.05) return ["Beats placebo", "--green"];
  if (p >= 0.95) return ["Worse than placebo", "--red"];
  return ["No detectable edge", "--ink-2"];
}
const mono = (text) => h("span", { class: "m" }, text);
let current = null;

// Shareable link: the API returns a token built from the parsed input; the page only copies the URL.
function linkFor(token) { return `${location.origin}/t/${token}`; }
function showCopy(token) {
  const box = document.getElementById("share");
  if (!box) return;
  box.replaceChildren();
  if (!token) return;
  const btn = h("button", { type: "button", class: "secondary", id: "copy" }, "Copy link to this trial");
  const url = linkFor(token);
  btn.addEventListener("click", async () => {
    let ok = false;
    try { await navigator.clipboard.writeText(url); ok = true; } catch (e) {
      const tmp = h("input", { type: "text", value: url, readonly: "", "aria-hidden": "true", style: { position: "fixed", opacity: "0" } });
      document.body.append(tmp); tmp.select(); try { ok = document.execCommand("copy"); } catch (e2) { ok = false; } tmp.remove();
    }
    const label = btn.textContent; btn.textContent = ok ? "Link copied" : "Copy failed, select the link"; btn.disabled = ok;
    setTimeout(() => { btn.textContent = label; btn.disabled = false; }, 2000);
  });
  box.append(btn, h("span", { class: "m" }, url));
}

function drawField(pl) {
  const svg = $("field"), W = Math.max(300, Math.round(svg.getBoundingClientRect().width) || 640);
  const n = pl.histogram.length, bw = W / n, perCol = Math.max(2, Math.floor(bw / 3.4)), r = 1.3, step = 3.0;
  const maxRows = Math.ceil(Math.max(...pl.histogram) / perCol), H = maxRows * step + 44;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const lo = pl.bin_edges[0], hi = pl.bin_edges[n], x = (v) => ((v - lo) / (hi - lo)) * W;
  const nodes = [];
  let seen = 0;
  const beaten = Math.round(pl.percentile * 20);  // twins with a lower Sharpe than the agent, of 2,000
  pl.histogram.forEach((c, i) => {
    for (let k = 0; k < c; k++) {
      const beat = seen++ < beaten, col = k % perCol, row = Math.floor(k / perCol);
      nodes.push(s("circle", { cx: (i * bw + 2 + col * (bw - 4) / Math.max(1, perCol - 1)).toFixed(1), cy: (H - 22 - row * step).toFixed(1), r, fill: `var(${beat ? "--ink" : "--ink-3"})` }));
    }
  });
  const ax = Math.min(W - 1, Math.max(1, x(pl.agent_sharpe))), right = ax > W - 120;
  nodes.push(s("line", { x1: ax, x2: ax, y1: 14, y2: H - 18, stroke: "var(--accent)", "stroke-width": 2 }));
  nodes.push(s("text", { ...MONO, x: ax + (right ? -6 : 6), y: 12, fill: "var(--accent)", "text-anchor": right ? "end" : "start" }, `agent ${f2(pl.agent_sharpe)}`));
  if (lo < 0 && hi > 0) {
    nodes.push(s("line", { x1: x(0), x2: x(0), y1: H - 18, y2: H - 12, stroke: "var(--ink-2)" }));
    nodes.push(s("text", { ...MONO, x: x(0), y: H - 2, fill: "var(--ink-2)", "text-anchor": "middle" }, "0"));
  }
  nodes.push(s("text", { ...MONO, x: 0, y: H - 2, fill: "var(--ink-2)" }, f2(lo)));
  nodes.push(s("text", { ...MONO, x: W, y: H - 2, fill: "var(--ink-2)", "text-anchor": "end" }, f2(hi)));
  svg.replaceChildren(...nodes);
}

function drawForest(F) {
  const names = { all: "All hours", us_closed: "US market closed", us_open: "US market open" };
  const vals = Object.values(F).flatMap((q) => [q.p05, q.p95, q.agent_sharpe]);
  const lo = Math.min(...vals) - 0.5, hi = Math.max(...vals) + 0.5, x = (v) => ((v - lo) / (hi - lo)) * 100 + "%";
  const cells = Object.entries(F).flatMap(([k, q]) => [
    h("div", {}, names[k] || k),
    h("div", { class: "track" },
      h("i", { class: "band", style: { left: x(q.p05), width: `calc(${x(q.p95)} - ${x(q.p05)})` } }),
      h("i", { class: "cap", style: { left: x(q.p05) } }), h("i", { class: "cap", style: { left: x(q.p95) } }),
      h("i", { class: "mid", style: { left: x(q.mean) } }), h("i", { class: "est", style: { left: x(q.agent_sharpe) } })),
    h("div", { class: "num" }, `${Math.round(q.percentile)}%`),
  ]);
  cells.push(h("div"), h("div", { class: "axis" }, h("span", {}, f2(lo)), h("span", {}, "Sharpe"), h("span", {}, f2(hi))), h("div"));
  $("forest").replaceChildren(...cells);
}

function render(r, opts = {}) {
  current = r;
  const pl = r.placebo, p = Math.round(pl.percentile), c = r.consistency == null ? null : Math.round(r.consistency * 100);
  const [word, colour] = state(pl.p_value);
  $("tag").textContent = word; $("tag").style.color = `var(${colour})`;
  const w = { stroke: `var(${colour})`, "stroke-width": 1.5 };
  $("whisker").replaceChildren(
    s("line", { ...w, x1: 1, x2: 27, y1: 6, y2: 6 }), s("line", { ...w, x1: 1, x2: 1, y1: 2, y2: 10 }),
    s("line", { ...w, x1: 27, x2: 27, y1: 2, y2: 10 }), s("rect", { x: 10, y: 2, width: 8, height: 8, rx: 1.5, fill: "var(--accent)" }));
  $("stamp").textContent = r.verdict;
  $("vtext").textContent = (VERDICT[r.verdict] || VERDICT["DO NOT DEPLOY"])(p, c);
  const runs = r.p_values_per_run.length, sh = (r.sharpe_per_run || []).map(f2).join(" and ");
  let line = `p = ${pl.p_value.toFixed(3)} against 2,000 placebo twins (one-sided).` +
    (runs > 1 ? ` Worst of ${runs} runs;${sh ? ` run Sharpes ${sh};` : ""} consistency ${c}%.`
      : " One run: consistency needs 2 or more runs, so this upload tops out at HALF SIZE. Add a run column to be judged for DEPLOY.");
  if (r.hours && r.hours !== "all") line += ` Positions kept only while the US exchange is ${r.hours === "us_closed" ? "closed" : "open"}.`;
  $("pline").textContent = line;
  $("example").hidden = !opts.example;
  $("forestfig").hidden = !opts.forest;
  drawField(pl);
  const names = { agent: r.agent_source === "uploaded" ? "your agent" : "reference agent", rule: "momentum rule", hold: "buy and hold" };
  $("arms").replaceChildren(
    h("tr", {}, HEADS.map((t) => h("th", {}, t))),
    ...Object.entries(r.arms).map(([k, v]) => h("tr", { class: k },
      [names[k] || k, f2(v.sharpe), pct(v.total_return), pct(v.max_drawdown), `${Math.round(v.win_rate * 100)}%`, f2(v.turnover_per_day), pct(v.costs_paid)].map((t, i) => h("td", { "data-k": HEADS[i] }, t)))));
  const snap = r.data_snapshot && r.data_snapshot.fetched_at;
  $("snapshot").textContent = snap ? `Candles, fees and funding pulled from api.bitget.com on ${whenFull(snap)}.` : "";
  $("mood").textContent = "";  // market mood is optional; when Bitget's data service answers with an error the line stays empty
  if (opts.press) { const st = $("stamp"); st.classList.remove("pressed"); void st.offsetWidth; st.classList.add("pressed"); }
  showCopy(opts.token || null);
}

// Current page in the header nav.
(() => {
  const path = location.pathname;
  const here = (path === "/methods" || path === "/methods.html") ? "/methods" : "/";  // a shared trial is a trial
  for (const a of document.querySelectorAll("header nav a")) if (a.getAttribute("href") === here) a.setAttribute("aria-current", "page");
})();

async function postTrial(body) {
  const res = await fetch("/api/trial", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await res.json().catch(() => ({ error: `server returned ${res.status}` }));
  if (!res.ok || data.error) throw new Error(data.error || `server returned ${res.status}`);
  return data;
}
let resizeTimer;
addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { if (current) drawField(current.placebo); }, 200); });


