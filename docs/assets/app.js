const CSS = getComputedStyle(document.documentElement);
const COLOR_SDB = CSS.getPropertyValue("--series-serenedb").trim();
const COLOR_QDR = CSS.getPropertyValue("--series-qdrant").trim();
const COLOR_SDB_LIGHT = CSS.getPropertyValue("--series-serenedb-light").trim();
const COLOR_QDR_LIGHT = CSS.getPropertyValue("--series-qdrant-light").trim();
const COLOR_GRID = CSS.getPropertyValue("--gridline").trim();
const COLOR_TEXT = CSS.getPropertyValue("--text-secondary").trim();

const FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif";
Chart.defaults.font.family = FONT;
Chart.defaults.color = COLOR_TEXT;

function bandStart(label) {
  return parseFloat(label.split("-")[0]);
}

function fmtInt(n) {
  return Math.round(n).toLocaleString();
}

// Only label "clean" log-scale ticks (1/2/5 x 10^n) so the axis doesn't fill
// with every minor gridline value.
function niceLogTick(v) {
  const exp = Math.floor(Math.log10(v) + 1e-9);
  const base = Math.round((v / Math.pow(10, exp)) * 10) / 10;
  return [1, 2, 5].includes(base) ? fmtInt(v) : null;
}

// Pick a "nice" (1/2/5 x 10^n) gridline step and a max that's an exact
// multiple of it, with a little headroom over the raw data max. Two charts
// given the *same* raw value here get the *same* {max, step} back, so their
// y-axes carry identical gridlines at identical positions -- not just the
// same range, but the same horizontal lines -- letting you compare bar
// heights across the two charts directly instead of cross-checking numbers.
function niceAxisScale(value, targetTicks = 6) {
  const roughStep = (value * 1.08) / targetTicks;
  const magnitude = Math.pow(10, Math.floor(Math.log10(roughStep)));
  const residual = roughStep / magnitude;
  const niceResidual = residual > 5 ? 10 : residual > 2 ? 5 : residual > 1 ? 2 : 1;
  const step = niceResidual * magnitude;
  const max = Math.ceil((value * 1.08) / step) * step;
  return { max, step };
}

function setMetaLine(meta) {
  document.getElementById("meta-line").textContent =
    `dataset=${meta.dataset} · nb=${meta.nb.toLocaleString()} · dim=${meta.dim} ` +
    `· k=${meta.k} · ${meta.clients} concurrent clients (closed-loop QPS)`;
}

function allBands(...seriesLists) {
  const set = new Set();
  for (const rows of seriesLists) {
    for (const r of rows) set.add(r.band);
  }
  return Array.from(set).sort((a, b) => bandStart(a) - bandStart(b));
}

function byBand(rows) {
  const m = new Map();
  for (const r of rows) m.set(r.band, r);
  return m;
}

// ---- Chart 1: recall vs QPS (line) ----------------------------------------

function renderRecallQps(data) {
  const sdb = data.serenedb;
  const qdr = data.qdrant;

  new Chart(document.getElementById("chart-recall-qps"), {
    type: "line",
    data: {
      datasets: [
        {
          label: "SereneDB",
          data: sdb.map((r) => ({ x: r.recall, y: r.qps, config: r.config, band: r.band })),
          borderColor: COLOR_SDB,
          backgroundColor: COLOR_SDB,
          borderWidth: 2,
          pointRadius: 4,
          pointHoverRadius: 6,
          pointBackgroundColor: COLOR_SDB,
          pointBorderColor: CSS.getPropertyValue("--surface-1").trim(),
          pointBorderWidth: 2,
          tension: 0,
        },
        {
          label: "Qdrant",
          data: qdr.map((r) => ({ x: r.recall, y: r.qps, config: r.config, band: r.band })),
          borderColor: COLOR_QDR,
          backgroundColor: COLOR_QDR,
          borderWidth: 2,
          pointRadius: 4,
          pointHoverRadius: 6,
          pointBackgroundColor: COLOR_QDR,
          pointBorderColor: CSS.getPropertyValue("--surface-1").trim(),
          pointBorderWidth: 2,
          tension: 0,
        },
      ],
    },
    options: {
      responsive: true,
      interaction: { mode: "nearest", intersect: false },
      scales: {
        x: {
          type: "linear",
          title: { display: true, text: "Recall@10" },
          grid: { color: COLOR_GRID },
          ticks: { callback: (v) => v.toFixed(2) },
          min: 0.55,
          max: 1.0,
        },
        y: {
          type: "logarithmic",
          title: { display: true, text: "QPS (32 clients, log scale)" },
          grid: { color: COLOR_GRID },
          ticks: { callback: (v) => niceLogTick(v) },
        },
      },
      plugins: {
        legend: { position: "top", align: "start", labels: { usePointStyle: true, boxWidth: 8 } },
        tooltip: {
          callbacks: {
            title: (items) => `recall band ${items[0].raw.band}`,
            label: (item) =>
              `${fmtInt(item.raw.y)} qps  ·  recall ${item.raw.x.toFixed(4)}  ·  ${item.dataset.label} (${item.raw.config})`,
          },
        },
      },
    },
  });

  renderTable("table-recall-qps", ["Band", "Engine", "Recall", "QPS", "p50 ms", "p95 ms", "Config"],
    interleave(sdb, qdr, "SereneDB", "Qdrant").map((r) => [
      r.band, r.engine, r.recall.toFixed(4), fmtInt(r.qps), r.p50_ms.toFixed(2), r.p95_ms.toFixed(2), r.config,
    ]));
}

// ---- Build-time / index-size panels: single-engine breakdown bars ---------
//
// Each panel plots one engine's own knobs against itself (e.g. SereneDB's
// quantizer x segment-merge policy, or Qdrant's (m, ef_construct) x quant) --
// not an engine-vs-engine comparison, so both series in a panel share one
// hue family (solid = the more expensive/precise variant, tinted = the
// leaner one) rather than the two engines' identity colors.

function pivot(rows, catFn, seriesFn) {
  const categories = [];
  const seriesKeys = [];
  const cell = new Map();
  for (const r of rows) {
    const cat = catFn(r);
    const series = seriesFn(r);
    if (!categories.includes(cat)) categories.push(cat);
    if (!seriesKeys.includes(series)) seriesKeys.push(series);
    cell.set(`${cat}||${series}`, r);
  }
  return { categories, seriesKeys, cell };
}

function renderBreakdownBar(canvasId, rows, opts) {
  const { categories, seriesKeys, cell } = pivot(rows, opts.catFn, opts.seriesFn);
  const datasets = seriesKeys.map((s) => ({
    label: opts.seriesLabel(s),
    data: categories.map((c) => {
      const row = cell.get(`${c}||${s}`);
      return row ? row[opts.valueKey] : null;
    }),
    backgroundColor: opts.seriesColor(s),
    borderRadius: 4,
    maxBarThickness: 28,
    categoryPercentage: 0.6,
    barPercentage: 0.9,
  }));

  new Chart(document.getElementById(canvasId), {
    type: "bar",
    data: { labels: categories, datasets },
    options: {
      responsive: true,
      // The chart-card gets a fixed CSS height so paired panels (see
      // .panel-grid in style.css) render at the exact same size --
      // maintainAspectRatio would otherwise override that with a
      // width-derived height.
      maintainAspectRatio: false,
      scales: {
        x: { title: { display: true, text: opts.xLabel }, grid: { display: false } },
        y: { title: { display: true, text: opts.yLabel }, grid: { color: COLOR_GRID }, beginAtZero: true,
             max: opts.yMax, ticks: { stepSize: opts.yStep } },
      },
      plugins: {
        legend: seriesKeys.length > 1
          ? { position: "top", align: "start", labels: { usePointStyle: true, boxWidth: 8 } }
          : { display: false },
        tooltip: {
          callbacks: {
            label: (item) => {
              const row = cell.get(`${item.label}||${seriesKeys[item.datasetIndex]}`);
              return row ? `${item.dataset.label}: ${item.formattedValue}` : `${item.dataset.label}: no data`;
            },
          },
        },
      },
    },
  });
}

// ---- shared table + interleave helpers -------------------------------------

function interleave(sdbRows, qdrRows, sdbLabel, qdrLabel) {
  const bands = allBands(sdbRows, qdrRows);
  const sdbMap = byBand(sdbRows);
  const qdrMap = byBand(qdrRows);
  const out = [];
  for (const b of bands) {
    if (sdbMap.has(b)) out.push({ ...sdbMap.get(b), engine: sdbLabel });
    if (qdrMap.has(b)) out.push({ ...qdrMap.get(b), engine: qdrLabel });
  }
  return out;
}

function renderTable(tableId, headers, rows) {
  const table = document.getElementById(tableId);
  const thead = document.createElement("thead");
  const trh = document.createElement("tr");
  for (const h of headers) {
    const th = document.createElement("th");
    th.textContent = h;
    trh.appendChild(th);
  }
  thead.appendChild(trh);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  for (const row of rows) {
    const tr = document.createElement("tr");
    for (const cell of row) {
      const td = document.createElement("td");
      td.textContent = cell;
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
}

const SDB_SETTLE_LABEL = { compact: "Compact", "no-compact": "No-compact" };
const QDR_QUANT_LABEL = { none: "Full precision", scalar: "Scalar quant" };

// One stacked bar per quantizer: the solid base segment is the index build
// itself (CREATE INDEX + VACUUM REFRESH), the light/dashed segment stacked
// on top is the extra VACUUM COMPACT merge cost -- both numbers come from
// the same compact-settle measurement, so the stack is a real decomposition
// of one build's total time, not two builds side by side.
function renderSerenedbBuildTimeStacked(chartId, tableId, rows, yMax, yStep) {
  const categories = rows.map((r) => r.quant);

  new Chart(document.getElementById(chartId), {
    type: "bar",
    data: {
      labels: categories,
      datasets: [
        {
          label: "Index build (CREATE INDEX)",
          data: rows.map((r) => r.index_build_s),
          backgroundColor: COLOR_SDB,
          borderRadius: { topLeft: 0, topRight: 0, bottomLeft: 4, bottomRight: 4 },
          maxBarThickness: 40,
          categoryPercentage: 0.5,
          barPercentage: 0.9,
        },
        {
          label: "Compact merge (VACUUM COMPACT)",
          data: rows.map((r) => r.compact_s),
          backgroundColor: COLOR_SDB_LIGHT,
          borderColor: COLOR_SDB,
          borderWidth: 2,
          borderDash: [4, 3],
          borderRadius: { topLeft: 4, topRight: 4, bottomLeft: 0, bottomRight: 0 },
          maxBarThickness: 40,
          categoryPercentage: 0.5,
          barPercentage: 0.9,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { stacked: true, title: { display: true, text: "Quantizer" }, grid: { display: false } },
        y: { stacked: true, title: { display: true, text: "Build time (seconds)" }, grid: { color: COLOR_GRID }, beginAtZero: true,
             max: yMax, ticks: { stepSize: yStep } },
      },
      plugins: {
        legend: { position: "top", align: "start", labels: { usePointStyle: true, boxWidth: 8 } },
        tooltip: {
          callbacks: {
            label: (item) => `${item.dataset.label}: ${item.formattedValue}s`,
            footer: (items) => `Total: ${items.reduce((sum, i) => sum + i.parsed.y, 0).toFixed(1)}s`,
          },
        },
      },
    },
  });

  renderTable(tableId, ["Quantizer", "nlist", "Index build (s)", "Compact merge (s)", "Total (s)"],
    rows.map((r) => [r.quant, fmtInt(r.nlist), r.index_build_s.toFixed(1), r.compact_s.toFixed(1),
                     (r.index_build_s + r.compact_s).toFixed(1)]));
}

function renderSerenedbBreakdown(chartId, tableId, rows, valueKey, yLabel, valueHeader, fmt) {
  renderBreakdownBar(chartId, rows, {
    catFn: (r) => r.quant,
    seriesFn: (r) => r.settle,
    seriesLabel: (s) => SDB_SETTLE_LABEL[s] || s,
    seriesColor: (s) => (s === "compact" ? COLOR_SDB : COLOR_SDB_LIGHT),
    valueKey,
    xLabel: "Quantizer",
    yLabel,
  });
  renderTable(tableId, ["Quantizer", "Settle", "nlist", valueHeader],
    rows.map((r) => [r.quant, SDB_SETTLE_LABEL[r.settle] || r.settle, fmtInt(r.nlist), fmt(r[valueKey])]));
}

function renderQdrantBreakdown(chartId, tableId, rows, valueKey, yLabel, valueHeader, fmt, yMax, yStep) {
  renderBreakdownBar(chartId, rows, {
    catFn: (r) => `m=${r.m}, efc=${r.ef_construct}`,
    seriesFn: (r) => r.quant,
    seriesLabel: (s) => QDR_QUANT_LABEL[s] || s,
    seriesColor: (s) => (s === "scalar" ? COLOR_QDR : COLOR_QDR_LIGHT),
    valueKey,
    xLabel: "HNSW (m, ef_construct)",
    yLabel,
    yMax,
    yStep,
  });
  renderTable(tableId, ["m", "ef_construct", "Quant", valueHeader],
    rows.map((r) => [r.m, r.ef_construct, QDR_QUANT_LABEL[r.quant] || r.quant, fmt(r[valueKey])]));
}

// ---- boot -------------------------------------------------------------------

fetch("data/comparison.json")
  .then((r) => r.json())
  .then((data) => {
    setMetaLine(data.meta);
    renderRecallQps(data.recall_qps);

    // Shared y-axis scale (same max AND same gridline step) across both
    // build-time panels, so equal values land on the same horizontal line in
    // either chart -- matching ranges alone isn't enough if the two charts
    // pick different tick steps, the gridlines still wouldn't line up.
    const { max: buildTimeMax, step: buildTimeStep } = niceAxisScale(Math.max(
      ...data.build_time.serenedb_by_quant.map((r) => r.index_build_s + r.compact_s),
      ...data.build_time.qdrant_by_config.map((r) => r.build_s)));

    renderSerenedbBuildTimeStacked("chart-build-time-serenedb", "table-build-time-serenedb",
      data.build_time.serenedb_by_quant, buildTimeMax, buildTimeStep);
    renderQdrantBreakdown("chart-build-time-qdrant", "table-build-time-qdrant",
      data.build_time.qdrant_by_config, "build_s", "Build time (seconds)", "Build (s)", (v) => v.toFixed(1),
      buildTimeMax, buildTimeStep);

    renderSerenedbBreakdown("chart-index-size-serenedb", "table-index-size-serenedb",
      data.index_size.serenedb_by_quant, "index_mb", "Index size (MB)", "Size (MB)", (v) => fmtInt(v));
    renderQdrantBreakdown("chart-index-size-qdrant", "table-index-size-qdrant",
      data.index_size.qdrant_by_config, "index_mb", "Index size (MB)", "Size (MB)", (v) => fmtInt(v));
  })
  .catch((err) => {
    document.getElementById("meta-line").textContent = "Failed to load benchmark data: " + err;
  });
