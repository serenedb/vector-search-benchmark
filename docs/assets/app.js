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
      scales: {
        x: { title: { display: true, text: opts.xLabel }, grid: { display: false } },
        y: { title: { display: true, text: opts.yLabel }, grid: { color: COLOR_GRID }, beginAtZero: true },
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

function renderQdrantBreakdown(chartId, tableId, rows, valueKey, yLabel, valueHeader, fmt) {
  renderBreakdownBar(chartId, rows, {
    catFn: (r) => `m=${r.m}, efc=${r.ef_construct}`,
    seriesFn: (r) => r.quant,
    seriesLabel: (s) => QDR_QUANT_LABEL[s] || s,
    seriesColor: (s) => (s === "scalar" ? COLOR_QDR : COLOR_QDR_LIGHT),
    valueKey,
    xLabel: "HNSW (m, ef_construct)",
    yLabel,
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

    renderSerenedbBreakdown("chart-build-time-serenedb", "table-build-time-serenedb",
      data.build_time.serenedb_by_quant, "build_s", "Build time (seconds)", "Build (s)", (v) => v.toFixed(1));
    renderQdrantBreakdown("chart-build-time-qdrant", "table-build-time-qdrant",
      data.build_time.qdrant_by_config, "build_s", "Build time (seconds)", "Build (s)", (v) => v.toFixed(1));

    renderSerenedbBreakdown("chart-index-size-serenedb", "table-index-size-serenedb",
      data.index_size.serenedb_by_quant, "index_mb", "Index size (MB)", "Size (MB)", (v) => fmtInt(v));
    renderQdrantBreakdown("chart-index-size-qdrant", "table-index-size-qdrant",
      data.index_size.qdrant_by_config, "index_mb", "Index size (MB)", "Size (MB)", (v) => fmtInt(v));
  })
  .catch((err) => {
    document.getElementById("meta-line").textContent = "Failed to load benchmark data: " + err;
  });
