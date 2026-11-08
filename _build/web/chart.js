// Tiny SVG chart engine shared by the interactive pages (taken from the Ethanol-Water-VLE page of the same author).
function interp(q, xs, ys) {
  if (q <= xs[0]) return ys[0]; const n = xs.length; if (q >= xs[n - 1]) return ys[n - 1];
  let lo = 0, hi = n - 1; while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xs[m] <= q) lo = m; else hi = m; }
  return ys[lo] + (q - xs[lo]) / (xs[hi] - xs[lo]) * (ys[hi] - ys[lo]);
}
const NS = "http://www.w3.org/2000/svg";
function mk(tag, at, parent) { const e = document.createElementNS(NS, tag); if (at) for (const k in at) e.setAttribute(k, at[k]); if (parent) parent.appendChild(e); return e; }
function parts(str) { const out = []; const re = /([_^])\{([^}]*)\}/g; let last = 0, m; while ((m = re.exec(str))) { if (m.index > last) out.push([str.slice(last, m.index), ""]); out.push([m[2], m[1]]); last = re.lastIndex; } if (last < str.length) out.push([str.slice(last), ""]); return out; }
function richSVG(str, el) { let sh = 0; for (const [t, k] of parts(str)) { const s = mk("tspan", null, el); const tgt = k === "_" ? 3 : k === "^" ? -4 : 0; if (tgt - sh) s.setAttribute("dy", tgt - sh); sh = tgt; if (k) s.setAttribute("font-size", "8.5"); s.textContent = t; } }
function richHTML(str, el) { for (const [t, k] of parts(str)) { if (!k) el.appendChild(document.createTextNode(t)); else { const n = document.createElement(k === "_" ? "sub" : "sup"); n.textContent = t; el.appendChild(n); } } }
function nice(min, max, n) {
  const span = max - min; if (!(span > 0)) return [min];
  const raw = span / Math.max(1, n), mag = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / mag;
  const step = (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * mag;
  const out = []; for (let v = Math.ceil(min / step - 1e-9) * step; v <= max + step * 1e-9; v += step) out.push(+v.toFixed(12));
  return out;
}
function logTicks(min, max) { const o = []; for (let e = Math.floor(Math.log10(min)); e <= Math.ceil(Math.log10(max)); e++) for (const m of [1, 2, 5]) { const v = m * 10 ** e; if (v >= min * 0.999 && v <= max * 1.001) o.push(v); } return o; }
const fmtTick = (v) => { const a = Math.abs(v); if (a === 0) return "0"; if (a >= 1000) return v.toFixed(0); if (a >= 10) return +v.toFixed(1) + ""; if (a >= 1) return +v.toFixed(2) + ""; return +v.toFixed(3) + ""; };
function lineDash(d) { return d === "dash" ? "6 4" : d === "dot" ? "1.5 3" : d === "dashdot" ? "7 3 1.5 3" : ""; }
function keySVG(it) {
  const s = mk("svg", { width: 22, height: 10, viewBox: "0 0 22 10", "aria-hidden": "true" });
  if (it.kind === "line" || !it.kind) { const l = mk("line", { x1: 1, x2: 21, y1: 5, y2: 5 }, s); l.style.stroke = it.color; l.style.strokeWidth = it.width || 2; if (it.dash) l.setAttribute("stroke-dasharray", lineDash(it.dash)); l.setAttribute("stroke-linecap", "round"); }
  else marker(s, it.kind, 11, 5, it.color, it.hollow, 3.6);
  return s;
}
function marker(g, kind, x, y, color, hollow, r = 4) {
  let e;
  if (kind === "tri") e = mk("path", { d: `M${x},${y - r * 1.15}L${x + r},${y + r * 0.75}L${x - r},${y + r * 0.75}Z` }, g);
  else if (kind === "diamond") e = mk("path", { d: `M${x},${y - r * 1.25}L${x + r * 1.05},${y}L${x},${y + r * 1.25}L${x - r * 1.05},${y}Z` }, g);
  else if (kind === "square") e = mk("rect", { x: x - r * 0.9, y: y - r * 0.9, width: r * 1.8, height: r * 1.8 }, g);
  else e = mk("circle", { cx: x, cy: y, r }, g);
  if (hollow) { e.style.fill = "var(--surface)"; e.style.stroke = color; e.style.strokeWidth = 1.5; }
  else { e.style.fill = color; e.style.stroke = "var(--surface)"; e.style.strokeWidth = 1.5; }
  return e;
}
let chartSeq = 0;
class Chart {
  constructor(host, opt = {}) {
    this.opt = Object.assign({ height: 300, square: false, maxH: 400, m: { l: 50, r: 12, t: 10, b: 40 } }, opt);
    this.id = "c" + ++chartSeq;
    this.leg = document.createElement("div"); this.leg.className = "legend"; host.appendChild(this.leg);
    this.box = document.createElement("div"); this.box.className = "plotbox"; this.box.tabIndex = 0; host.appendChild(this.box);
    this.svg = mk("svg", { role: "img" }, this.box);
    this.tip = document.createElement("div"); this.tip.className = "tip"; this.tip.hidden = true; this.box.appendChild(this.tip);
    this.spec = null; this.hx = null; this.W = 0;
    new ResizeObserver(() => { const w = this.box.clientWidth; if (this.spec && w && Math.abs(w - this.W) > 0.5) this.draw(); }).observe(this.box);
    const move = (ev) => { if (!this.sx) return; const r = this.svg.getBoundingClientRect(); this.hover(ev.clientX - r.left, ev.clientY - r.top); };
    this.svg.addEventListener("pointermove", move);
    this.svg.addEventListener("pointerdown", move);
    this.svg.addEventListener("pointerleave", () => this.clear());
    this.svg.addEventListener("click", (ev) => {
      if (!this.sx || !this.spec) return; const r = this.svg.getBoundingClientRect(), px = ev.clientX - r.left, py = ev.clientY - r.top;
      let best = null, bd = 12;
      for (const se of this.spec.series) if (se.onPick) for (const p of se.pts) { if (!isFinite(p[0]) || !isFinite(p[1])) continue; const d = Math.hypot(this.sx(p[0]) - px, this.sy(p[1]) - py); if (d < bd) { bd = d; best = [se, p]; } }
      if (best) best[0].onPick(best[1]);
    });
    this.box.addEventListener("focus", () => { if (!this.sx) return; this.kx = this.kx ?? (this.spec.x.min + this.spec.x.max) / 2; this.hover(this.sx(this.kx), null, true); });
    this.box.addEventListener("blur", () => this.clear());
    this.box.addEventListener("keydown", (e) => {
      if (!this.sx) return; const sp = this.spec.x, st = (sp.max - sp.min) / 100;
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") { e.preventDefault(); this.kx = Math.min(sp.max, Math.max(sp.min, (this.kx ?? (sp.min + sp.max) / 2) + (e.key === "ArrowRight" ? st : -st))); this.hover(this.sx(this.kx), null, true); }
      if (e.key === "Escape") this.clear();
    });
  }
  set(spec) { this.spec = spec; this.svg.setAttribute("aria-label", spec.aria || ""); this.legend(); this.draw(); if (this.last) this.hover(...this.last); }
  legend() {
    this.leg.textContent = "";
    for (const it of this.spec.series.filter((s) => s.name && !s.noLegend).concat(this.spec.marks?.filter((m) => m.legend) || [])) {
      const sp = document.createElement("span"); sp.appendChild(keySVG({ kind: it.legendKind || it.kind === "points" && (it.shape || "circle") || it.kind === "point" && it.shape || "line", color: it.color, dash: it.dash, hollow: it.hollow, width: it.width }));
      const t = document.createElement("span"); richHTML(it.legend || it.name, t); sp.appendChild(t); this.leg.appendChild(sp);
    }
  }
  draw() {
    const s = this.spec, o = this.opt, m = o.m;
    const W = Math.max(240, this.box.clientWidth); this.W = this.box.clientWidth;
    const H = o.square ? Math.min(o.maxH, Math.round(W * 0.92)) : o.height;
    const svg = this.svg; svg.textContent = ""; svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("width", W); svg.setAttribute("height", H);
    const x0 = m.l, x1 = W - m.r, y0 = H - m.b, y1 = m.t;
    const lx = (v) => Math.log10(v);
    const sx = s.x.log ? (v) => x0 + (lx(v) - lx(s.x.min)) / (lx(s.x.max) - lx(s.x.min)) * (x1 - x0)
                       : (v) => x0 + (v - s.x.min) / (s.x.max - s.x.min) * (x1 - x0);
    const sy = s.y.log ? (v) => y0 - (Math.log10(v) - Math.log10(s.y.min)) / (Math.log10(s.y.max) - Math.log10(s.y.min)) * (y0 - y1)
                       : (v) => y0 - (v - s.y.min) / (s.y.max - s.y.min) * (y0 - y1);
    this.sx = sx; this.sy = sy; this.box_ = { x0, x1, y0, y1, W, H };
    this.ix = s.x.log ? (px) => Math.pow(10, lx(s.x.min) + (px - x0) / (x1 - x0) * (lx(s.x.max) - lx(s.x.min)))
                      : (px) => s.x.min + (px - x0) / (x1 - x0) * (s.x.max - s.x.min);
    const defs = mk("defs", null, svg); const cp = mk("clipPath", { id: this.id + "clip" }, defs); mk("rect", { x: x0, y: y1, width: x1 - x0, height: y0 - y1 }, cp);
    const bg = mk("rect", { x: x0, y: y1, width: x1 - x0, height: y0 - y1 }, svg); bg.style.fill = "var(--surface)";
    // grid & ticks
    const gx = mk("g", null, svg), gy = mk("g", null, svg);
    const xt = s.x.ticks || (s.x.log ? logTicks(s.x.min, s.x.max) : nice(s.x.min, s.x.max, Math.max(3, Math.floor((x1 - x0) / 70))));
    const yt = s.y.ticks || (s.y.log ? logTicks(s.y.min, s.y.max) : nice(s.y.min, s.y.max, Math.max(3, Math.floor((y0 - y1) / 42))));
    for (const v of xt) { const X = sx(v); const l = mk("line", { x1: X, x2: X, y1: y1, y2: y0 }, gx); l.style.stroke = "var(--grid)"; l.style.strokeWidth = 1; const t = mk("text", { x: X, y: y0 + 15, "text-anchor": "middle" }, gx); t.textContent = (s.x.fmt || fmtTick)(v); }
    for (const v of yt) { const Y = sy(v); const l = mk("line", { x1: x0, x2: x1, y1: Y, y2: Y }, gy); l.style.stroke = "var(--grid)"; l.style.strokeWidth = 1; const t = mk("text", { x: x0 - 6, y: Y + 3.5, "text-anchor": "end" }, gy); t.textContent = (s.y.fmt || fmtTick)(v); }
    const ax = mk("path", { d: `M${x0},${y1}V${y0}H${x1}` }, svg); ax.style.stroke = "var(--axis)"; ax.style.fill = "none";
    if (!s.y.log && s.y.min < 0 && s.y.max > 0) { const z = mk("line", { x1: x0, x2: x1, y1: sy(0), y2: sy(0) }, svg); z.style.stroke = "var(--axis)"; z.style.strokeWidth = 1; }
    const xl = mk("text", { class: "lab", x: (x0 + x1) / 2, y: H - 6, "text-anchor": "middle" }, svg); richSVG(s.x.label, xl);
    const yl = mk("text", { class: "lab", x: 0, y: 0, "text-anchor": "middle", transform: `translate(12,${(y0 + y1) / 2}) rotate(-90)` }, svg); richSVG(s.y.label, yl);
    const plot = mk("g", { "clip-path": `url(#${this.id}clip)` }, svg);
    for (const mk_ of s.marks || []) if (mk_.kind === "hline" || mk_.kind === "vline" || mk_.kind === "seg") this.drawMark(plot, mk_);
    for (const se of s.series) {
      if (se.kind === "points") {
        const g = mk("g", null, plot);
        for (const p of se.pts) {
          if (se.err && p[2]) { const a = mk("path", { d: se.errX ? `M${sx(p[0] - p[2])},${sy(p[1])}H${sx(p[0] + p[2])}` : `M${sx(p[0])},${sy(p[1] - p[2])}V${sy(p[1] + p[2])}M${sx(p[0]) - 2.5},${sy(p[1] - p[2])}h5M${sx(p[0]) - 2.5},${sy(p[1] + p[2])}h5` }, g); a.style.stroke = "var(--muted)"; a.style.strokeWidth = 1; a.style.fill = "none"; }
        }
        for (const p of se.pts) { if (!isFinite(p[0]) || !isFinite(p[1])) continue; const e = marker(g, se.shape || "circle", sx(p[0]), sy(p[1]), se.color, se.hollow, se.r || 3.6); if (se.opacity) e.style.opacity = se.opacity; }
      } else {
        let d = "", pen = false;
        for (const p of se.pts) { if (!isFinite(p[0]) || !isFinite(p[1]) || (s.y.log && p[1] <= 0)) { pen = false; continue; } const X = sx(p[0]), Y = Math.max(-1e4, Math.min(1e4, sy(p[1]))); d += (pen ? "L" : "M") + X.toFixed(2) + "," + Y.toFixed(2); pen = true; }
        const path = mk("path", { d }, plot); path.style.fill = "none"; path.style.stroke = se.color; path.style.strokeWidth = se.width || 2;
        path.setAttribute("stroke-linejoin", "round"); path.setAttribute("stroke-linecap", "round"); if (se.dash) path.setAttribute("stroke-dasharray", lineDash(se.dash));
      }
    }
    for (const mk_ of s.marks || []) if (mk_.kind === "point") this.drawMark(svg, mk_);
    this.hl = mk("g", null, svg);
  }
  drawMark(g, mk_) {
    const { sx, sy } = this, b = this.box_;
    if (mk_.kind === "hline" || mk_.kind === "vline") {
      const l = mk_.kind === "hline" ? mk("line", { x1: b.x0, x2: b.x1, y1: sy(mk_.at), y2: sy(mk_.at) }, g) : mk("line", { x1: sx(mk_.at), x2: sx(mk_.at), y1: b.y1, y2: b.y0 }, g);
      l.style.stroke = mk_.color || "var(--muted)"; l.style.strokeWidth = mk_.width || 1; if (mk_.dash) l.setAttribute("stroke-dasharray", lineDash(mk_.dash));
      if (mk_.label) { const t = mk("text", { class: "ann", x: mk_.kind === "hline" ? b.x1 - 4 : sx(mk_.at) + 4, y: mk_.kind === "hline" ? sy(mk_.at) - 4 : b.y1 + 12, "text-anchor": mk_.kind === "hline" ? "end" : "start" }, g); richSVG(mk_.label, t); }
    } else if (mk_.kind === "seg") {
      const l = mk("line", { x1: sx(mk_.a[0]), y1: sy(mk_.a[1]), x2: sx(mk_.b[0]), y2: sy(mk_.b[1]) }, g);
      l.style.stroke = mk_.color || "var(--muted)"; l.style.strokeWidth = mk_.width || 1.2; if (mk_.dash) l.setAttribute("stroke-dasharray", lineDash(mk_.dash));
    } else if (mk_.kind === "point") {
      const X = sx(mk_.x), Y = sy(mk_.y);
      if (X < b.x0 - 1 || X > b.x1 + 1 || Y < b.y1 - 1 || Y > b.y0 + 1) return;
      marker(g, mk_.shape || "diamond", X, Y, mk_.color, false, mk_.r || 4.6);
      if (mk_.label) { const right = X > (b.x0 + b.x1) / 2; const t = mk("text", { class: "ann", x: X + (right ? -8 : 8), y: Y + (mk_.dy ?? -8), "text-anchor": right ? "end" : "start" }, g); richSVG(mk_.label, t); }
    }
  }
  clear() { this.last = null; this.tip.hidden = true; if (this.hl) this.hl.textContent = ""; }
  hover(px, py, kb) {
    const s = this.spec, b = this.box_; if (!s || !b) return;
    if (px < b.x0 - 6 || px > b.x1 + 6) { this.clear(); return; }
    px = Math.min(b.x1, Math.max(b.x0, px)); this.last = [px, py, kb];
    const xq = this.ix(px); if (kb) this.kx = xq;
    const hl = this.hl; hl.textContent = "";
    if (s.crosshair !== false) { const vl = mk("line", { x1: px, x2: px, y1: b.y1, y2: b.y0 }, hl); vl.style.stroke = "var(--ink-2)"; vl.style.strokeWidth = 1; vl.setAttribute("stroke-dasharray", "2 3"); }
    const rows = [];
    for (const se of s.series) {
      if (se.tip === false) continue;
      if (se.kind === "points") {
        if (py == null && !kb) continue;
        let best = null, bd = 1e9;
        for (const p of se.pts) { if (!isFinite(p[0]) || !isFinite(p[1])) continue; const d = py == null ? Math.abs(this.sx(p[0]) - px) : Math.hypot(this.sx(p[0]) - px, this.sy(p[1]) - py); if (d < bd) { bd = d; best = p; } }
        if (best && bd < 16) { rows.push({ se, text: se.fmtPt ? se.fmtPt(best) : best[1].toFixed(3), pt: best }); }
        continue;
      }
      const pts = se.sorted || (se.sorted = se.pts.filter((p) => isFinite(p[0]) && isFinite(p[1])).slice().sort((a, b) => a[0] - b[0]));
      if (!pts.length || xq < pts[0][0] - 1e-9 || xq > pts[pts.length - 1][0] + 1e-9) continue;
      const v = interp(xq, pts.map((p) => p[0]), pts.map((p) => p[1]));
      if (!isFinite(v) || (s.y.log && v <= 0)) continue;
      rows.push({ se, text: (se.fmt || ((z) => z.toFixed(3)))(v), pt: [xq, v] });
    }
    for (const r of rows) { const Y = this.sy(r.pt[1]); if (Y < b.y1 - 2 || Y > b.y0 + 2) continue; marker(hl, r.se.kind === "points" ? (r.se.shape || "circle") : "circle", this.sx(r.pt[0]), Y, r.se.color, false, r.se.kind === "points" ? 5 : 3.6); }
    const tip = this.tip; tip.textContent = "";
    const titled = rows.find((r) => r.se.title);
    const t = document.createElement("div"); t.className = "t"; richHTML(titled ? titled.se.title(titled.pt) : (s.tipTitle || ((x) => "x = " + x.toFixed(3)))(xq), t); tip.appendChild(t);
    for (const r of rows) {
      const d = document.createElement("div"); d.className = "r";
      d.appendChild(keySVG({ kind: r.se.kind === "points" ? (r.se.shape || "circle") : "line", color: r.se.color, dash: r.se.dash, hollow: r.se.hollow }));
      const bb = document.createElement("b"); bb.textContent = r.text; d.appendChild(bb);
      const i = document.createElement("i"); richHTML(r.se.tipName || r.se.name || "", i); d.appendChild(i); tip.appendChild(d);
    }
    if (!rows.length) { this.tip.hidden = true; return; }
    tip.hidden = false;
    const tw = tip.offsetWidth, left = px + 14 + tw > b.W ? px - 14 - tw : px + 14;
    tip.style.left = Math.max(0, left) + "px"; tip.style.top = (b.y1 + 4) + "px";
  }
}
