"""Export a saved Ontology and its Lake files to an offline interactive graph."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from urllib.parse import quote

from ..artifacts import WORKSPACE, read_json
from .html import escaped

NODE_TYPES = ("File", "Table", "Schema", "Field", "Concept")
COLORS = {"File": "#65748a", "Table": "#c27918", "Schema": "#3472c5", "Field": "#218373", "Concept": "#8054b9"}


def build_graph(ontology: dict, lake: Path, output: Path) -> dict:
    """Use persisted objects as facts; Claim payloads never become graph objects."""
    lake = lake.resolve()
    nodes, edges = {}, []

    def node(kind: str, object_id: str, data: dict | None = None) -> str:
        graph_id = f"{kind.lower()}:{object_id}"
        if graph_id not in nodes:
            nodes[graph_id] = {"id": graph_id, "type": kind, "object_id": object_id,
                               "label": (data or {}).get("name", object_id), "data": data or {}, "missing": data is None}
        return graph_id

    def file(reference: str, actual: bool = False) -> str:
        path = Path(reference)
        unsafe = ".." in path.parts or "\x00" in reference or (not actual and bool(re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", reference)))
        candidate = path if path.is_absolute() else lake / path
        if not unsafe:
            unsafe = not candidate.is_relative_to(lake) or not candidate.resolve().is_relative_to(lake)
        relative = candidate.relative_to(lake).as_posix() if not unsafe else reference
        graph_id = f"file:{relative}"
        exists = not unsafe and candidate.is_file()
        if graph_id not in nodes:
            data = {"path": relative, "exists_in_lake": exists, "referenced": not actual}
            if exists:
                data["bytes"] = candidate.stat().st_size
            if unsafe:
                data["warning"] = "Reference is outside the Lake or uses an unsafe path; no file link is offered."
            elif not exists:
                data["warning"] = "Referenced file is missing from the saved Lake."
            node("File", relative, data)
            nodes[graph_id].update(missing=not exists, actual_file=actual, unsafe=unsafe, href=None)
            if exists:
                relative_url = os.path.relpath(candidate, output.resolve().parent).replace(os.sep, "/")
                nodes[graph_id]["href"] = "./" + quote(relative_url, safe="/")
        elif not actual:
            nodes[graph_id]["data"]["referenced"] = True
        return graph_id

    def structural(source: str, target: str, label: str) -> None:
        edges.append({"id": f"structure:{len(edges)}", "source": source, "target": target,
                      "label": label, "group": "structural", "data": {"relationship": label}})

    for path in sorted(lake.rglob("*")):
        if path.is_file():
            file(path.relative_to(lake).as_posix(), actual=True)
    for object_id, concept in ontology.get("concepts", {}).items():
        node("Concept", object_id, concept)
    for object_id, schema in ontology.get("schemas", {}).items():
        schema_id = node("Schema", object_id, schema)
        for field in schema.get("fields", []):
            field_id = node("Field", field["id"], field)
            nodes[field_id]["schema"] = schema_id
            structural(schema_id, field_id, "contains")
        for reference in schema.get("documentation", []):
            structural(file(reference), schema_id, "defines")
    for object_id, table in ontology.get("tables", {}).items():
        table_id = node("Table", object_id, table)
        structural(table_id, node("Schema", table["schema_id"], ontology.get("schemas", {}).get(table["schema_id"])), "uses schema")
        structural(table_id, file(table["path"]), "stored in")
    source_types = {"field_link": "Field", "sub_concept_of": "Concept", "about": "Table"}
    objects_by_type = {"Field": {field["id"]: field for schema in ontology.get("schemas", {}).values() for field in schema.get("fields", [])},
                       "Concept": ontology.get("concepts", {}), "Table": ontology.get("tables", {})}
    for object_id, link in ontology.get("links", {}).items():
        kind, source, target = link["kind"], link["source"], link["target"]
        if kind == "mentions":
            source_id = file(source)
        else:
            source_type = source_types[kind]
            source_id = node(source_type, source, objects_by_type[source_type].get(source))
        target_id = node("Concept", target, ontology.get("concepts", {}).get(target))
        edges.append({"id": f"link:{object_id}", "source": source_id, "target": target_id,
                      "label": kind, "group": "persisted", "data": link})
    return {"nodes": list(nodes.values()), "edges": edges}


STYLE = """:root { color-scheme: light; font-family: system-ui, sans-serif; color: #26364c; background: #f3f6fb; }
* { box-sizing: border-box; } body { margin: 0; } [hidden] { display: none !important; }
header { padding: 22px 28px; background: #14243e; color: #f4f7ff; } h1 { margin: 0 0 7px; font-size: 25px; } h2 { margin: 0 0 12px; font-size: 18px; } p { line-height: 1.5; margin: 8px 0; }
.stats { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; } .stats span { background: #273b58; border-radius: 5px; padding: 6px 10px; font-size: 13px; }
.layout { display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 16px; padding: 18px; }
.card { background: white; border: 1px solid #d7dfec; border-radius: 9px; padding: 16px; min-width: 0; }
.toolbar { display: flex; flex-wrap: wrap; align-items: center; gap: 9px; margin-bottom: 12px; } button, input { font: inherit; }
button { border: 1px solid #becbdd; border-radius: 5px; padding: 7px 10px; background: white; color: inherit; cursor: pointer; } button:hover { background: #eaf1fc; } button:disabled { opacity: .4; cursor: default; }
.type-filter { font-size: 13px; white-space: nowrap; } .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin: 0 4px; }
#node-search { width: 100%; padding: 10px; border: 1px solid #b8c6da; border-radius: 5px; margin: 8px 0; } label { font-size: 13px; }
.hint, #visible-count, #selection-hint { font-size: 12px; color: #586c86; } .legend { display: flex; flex-wrap: wrap; gap: 16px; font-size: 12px; margin: 10px 0; }
.line { display: inline-block; width: 28px; border-top: 2px solid #566982; vertical-align: middle; margin-right: 5px; } .line.structural { border-top-style: dashed; } .line.overview { border-color: #9474c5; border-top-style: dotted; }
#graph { width: 100%; height: 72vh; min-height: 460px; max-height: 850px; background: #fbfcff; border: 1px solid #dce4ef; border-radius: 6px; touch-action: none; cursor: grab; }
#graph.dragging { cursor: grabbing; } .node { cursor: pointer; outline: none; } .node rect { stroke-width: 1.5; } .node.selected rect { stroke-width: 3.5; } .node.missing rect { stroke: #bc4037; stroke-dasharray: 5 3; }
.node text { fill: #26364c; pointer-events: none; font-size: 13px; } .node .node-type { font-size: 10px; font-weight: 650; letter-spacing: .04em; }
.edge { cursor: pointer; } .edge path { fill: none; stroke-width: 1.4; } .edge .edge-hit { stroke: transparent; stroke-width: 15; } .edge.structural .edge-line { stroke-dasharray: 6 4; } .edge.overview .edge-line { stroke-dasharray: 2 4; }
.edge text { fill: #53647e; font-size: 10px; paint-order: stroke; stroke: #fbfcff; stroke-width: 4; stroke-linejoin: round; opacity: 0; pointer-events: none; }
.show-labels .edge text, .edge:hover text, .edge.selected text { opacity: 1; } .edge:hover .edge-line, .edge.selected .edge-line { stroke-width: 3; }
#node-list { max-height: 230px; overflow-y: auto; display: grid; gap: 5px; margin: 10px 0 18px; } #node-list button { text-align: left; font-size: 12px; overflow-wrap: anywhere; } #node-list button.selected { background: #eaf1ff; border-color: #4678bf; }
#detail-title { overflow-wrap: anywhere; } #detail-data { white-space: pre-wrap; overflow-wrap: anywhere; background: #f5f7fb; border-radius: 5px; padding: 12px; font: 11.5px/1.6 ui-monospace, Consolas, monospace; max-height: 400px; overflow-y: auto; }
#detail-link { display: block; margin: 12px 0; color: #2865b4; overflow-wrap: anywhere; } #empty { padding: 22px; background: #f1f5fb; border-radius: 5px; }
@media (max-width: 1000px) { .layout { grid-template-columns: 1fr; } aside { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; } #graph { height: 65vh; } }
@media (max-width: 650px) { header { padding: 18px; } .layout { padding: 12px; } aside { display: block; } #graph { min-height: 400px; } }
"""

SCRIPT = r"""const graph = JSON.parse(document.getElementById('graph-data').textContent);
const nodes = new Map(graph.nodes.map(node => [node.id, node]));
const svg = document.getElementById('graph');
const world = document.getElementById('world');
const search = document.getElementById('node-search');
const filters = Array.from(document.querySelectorAll('.type-filter input'));
const list = document.getElementById('node-list');
const colors = {File: '#65748a', Table: '#c27918', Schema: '#3472c5', Field: '#218373', Concept: '#8054b9'};
const kinds = ['File', 'Table', 'Schema', 'Field', 'Concept'];
let positions = new Map(), shownNodes = [], shownEdges = [], selected = null, focus = null;
let tx = 20, ty = 20, scale = 1, worldWidth = 1, worldHeight = 1;
function element(name, attributes = {}, text = null) {
  const result = document.createElementNS('http://www.w3.org/2000/svg', name);
  Object.entries(attributes).forEach(([key, value]) => result.setAttribute(key, value));
  if (text !== null) result.textContent = text;
  return result;
}
function compact(value, max = 29) { return value.length > max ? value.slice(0, max - 1) + '…' : value; }
function transform() { world.setAttribute('transform', `translate(${tx} ${ty}) scale(${scale})`); }
function reset() {
  const size = svg.getBoundingClientRect();
  // Keep labels readable in a large graph: pan vertically or select a node in the list.
  scale = Math.max(.65, Math.min(1, (size.width - 32) / worldWidth, (size.height - 32) / worldHeight));
  tx = 16; ty = 16; transform();
}
function zoom(factor, x = svg.clientWidth / 2, y = svg.clientHeight / 2) {
  const nextScale = Math.max(.2, Math.min(3, scale * factor));
  tx = x - (x - tx) * nextScale / scale; ty = y - (y - ty) * nextScale / scale;
  scale = nextScale; transform();
}
function neighborhood(id) {
  const ids = new Set([id]);
  graph.edges.forEach(edge => { if (edge.source === id) ids.add(edge.target); if (edge.target === id) ids.add(edge.source); });
  return ids;
}
function overviewEdges(visible) {
  const summaries = new Map();
  graph.edges.filter(edge => edge.label === 'field_link').forEach(edge => {
    const schema = nodes.get(edge.source).schema;
    if (schema && visible.has(schema) && visible.has(edge.target) && !visible.has(edge.source)) {
      const id = schema + '→' + edge.target;
      if (!summaries.has(id)) summaries.set(id, {id: 'overview:' + id, source: schema, target: edge.target, group: 'overview', data: {explanation: 'Derived overview connection through hidden Fields; this is not a persisted Ontology Link.', link_ids: []}});
      summaries.get(id).data.link_ids.push(edge.data.id);
    }
  });
  return Array.from(summaries.values()).map(edge => ({...edge, label: `${edge.data.link_ids.length} field_link${edge.data.link_ids.length === 1 ? '' : 's'} via Fields`}));
}
function selectNode(node, center = false) {
  selected = node.id;
  document.getElementById('detail-title').textContent = `${node.type} · ${node.label}`;
  document.getElementById('detail-data').textContent = JSON.stringify({id: node.object_id, missing: node.missing, ...node.data}, null, 2);
  document.getElementById('selection-hint').textContent = node.missing ? 'Referenced object is missing from the saved Ontology or Lake.' : 'Select “Show connections” to inspect the immediate neighbors.';
  const link = document.getElementById('detail-link');
  link.hidden = !node.href;
  if (node.href) { link.setAttribute('href', node.href); link.textContent = 'Open Lake file · ' + node.data.path; } else link.removeAttribute('href');
  document.getElementById('focus-node').disabled = false;
  world.querySelectorAll('.node').forEach(item => item.classList.toggle('selected', item.dataset.node === selected));
  world.querySelectorAll('.edge').forEach(item => item.classList.remove('selected'));
  Array.from(list.children).forEach(item => item.classList.toggle('selected', item.dataset.node === selected));
  if (center && positions.has(selected)) {
    const point = positions.get(selected); scale = Math.max(scale, .85);
    tx = svg.clientWidth / 2 - (point.x + 105) * scale; ty = svg.clientHeight / 2 - (point.y + 29) * scale; transform();
  }
}
function selectEdge(edge, group) {
  selected = null;
  document.getElementById('detail-title').textContent = edge.label;
  document.getElementById('detail-data').textContent = JSON.stringify({graph_id: edge.id, relationship_type: edge.group, source: edge.source, target: edge.target, ...edge.data}, null, 2);
  document.getElementById('selection-hint').textContent = edge.group === 'persisted' ? 'An exact Link from ontology.json.' : edge.group === 'overview' ? 'Derived summary of persisted field_link Links through hidden Fields.' : 'Structural relationship derived from a saved Schema or Table.';
  document.getElementById('detail-link').hidden = true;
  document.getElementById('focus-node').disabled = true;
  world.querySelectorAll('.selected').forEach(item => item.classList.remove('selected'));
  group.classList.add('selected');
}
function draw() {
  const query = search.value.trim().toLowerCase();
  const enabled = new Set(filters.filter(input => input.checked).map(input => input.value));
  const matches = graph.nodes.filter(node => JSON.stringify(node).toLowerCase().includes(query));
  let visible = focus ? neighborhood(focus) : query ? new Set(matches.map(node => node.id)) : new Set(graph.nodes.filter(node => enabled.has(node.type)).map(node => node.id));
  shownNodes = graph.nodes.filter(node => visible.has(node.id));
  shownEdges = graph.edges.filter(edge => visible.has(edge.source) && visible.has(edge.target));
  if (!visible.size) shownEdges = [];
  else shownEdges.push(...overviewEdges(visible));
  positions = new Map(); world.replaceChildren();
  let x = 24; worldHeight = 1;
  kinds.forEach(kind => {
    const column = shownNodes.filter(node => node.type === kind).sort((a, b) => a.label.localeCompare(b.label));
    if (!column.length) return;
    const columns = kind === 'Concept' ? Math.ceil(column.length / 8) : kind === 'Field' ? Math.ceil(column.length / 18) : 1;
    const rows = Math.ceil(column.length / columns);
    world.append(element('text', {x, y: 15, fill: colors[kind], 'font-size': 13, 'font-weight': 650}, `${kind} · ${column.length}`));
    column.forEach((node, index) => positions.set(node.id, {x: x + Math.floor(index / rows) * 236, y: 34 + (index % rows) * 78}));
    x += columns * 236 + 52;
    worldHeight = Math.max(worldHeight, rows * 78 + 34);
  });
  worldWidth = Math.max(1, x - 52);
  const edgeLayer = element('g'); world.append(edgeLayer);
  shownEdges.forEach((edge, index) => {
    const from = positions.get(edge.source), to = positions.get(edge.target);
    const forward = to.x > from.x;
    const x1 = from.x + (forward ? 210 : 0), x2 = to.x + (forward ? 0 : 210);
    const y1 = from.y + 29, y2 = to.y + 29;
    let d, labelX, labelY;
    if (from.x === to.x) {
      const side = from.x + 228 + (index % 4) * 12;
      d = `M ${from.x + 210} ${y1} C ${side} ${y1}, ${side} ${y2}, ${to.x + 210} ${y2}`;
      labelX = side; labelY = (y1 + y2) / 2;
    } else {
      const middle = (x1 + x2) / 2;
      d = `M ${x1} ${y1} C ${middle} ${y1}, ${middle} ${y2}, ${x2} ${y2}`;
      labelX = middle; labelY = (y1 + y2) / 2 - 5;
    }
    const group = element('g', {class: 'edge ' + edge.group, tabindex: 0, role: 'button', 'aria-label': `${edge.label}: ${edge.source} to ${edge.target}`});
    group.append(element('title', {}, `${edge.label} · ${edge.group}\n${edge.source} → ${edge.target}`));
    group.append(element('path', {class: 'edge-hit', d}));
    group.append(element('path', {class: 'edge-line', d, stroke: edge.group === 'overview' ? '#9474c5' : '#8999af', 'marker-end': 'url(#arrow)'}));
    group.append(element('text', {x: labelX, y: labelY, 'text-anchor': 'middle'}, edge.label));
    group.addEventListener('click', () => selectEdge(edge, group));
    group.addEventListener('keydown', event => { if (event.key === 'Enter') selectEdge(edge, group); });
    edgeLayer.append(group);
  });
  shownNodes.forEach(node => {
    const point = positions.get(node.id), color = colors[node.type];
    const group = element('g', {class: 'node' + (node.missing ? ' missing' : '') + (selected === node.id ? ' selected' : ''), transform: `translate(${point.x} ${point.y})`, tabindex: 0, role: 'button', 'aria-label': `${node.type}: ${node.label}`});
    group.dataset.node = node.id;
    group.append(element('title', {}, `${node.type} · ${node.label}\n${node.object_id}${node.missing ? '\nMissing reference' : ''}`));
    group.append(element('rect', {width: 210, height: 58, rx: 7, fill: '#ffffff', stroke: color}));
    group.append(element('rect', {width: 5, height: 58, rx: 3, fill: color, stroke: 'none'}));
    group.append(element('text', {class: 'node-type', x: 13, y: 19, fill: color}, node.type.toUpperCase() + (node.missing ? ' · MISSING' : '')));
    group.append(element('text', {x: 13, y: 40}, compact(node.label)));
    group.addEventListener('click', () => selectNode(node));
    group.addEventListener('keydown', event => { if (event.key === 'Enter') selectNode(node); });
    world.append(group);
  });
  svg.classList.toggle('show-labels', document.getElementById('edge-labels').checked && shownEdges.length < 80);
  const summaries = shownEdges.filter(edge => edge.group === 'overview').length;
  document.getElementById('visible-count').textContent = `${shownNodes.length} / ${graph.nodes.length} nodes · ${shownEdges.length - summaries} edges visible${summaries ? ' + ' + summaries + ' overview connections' : ''}${focus ? ' · focused on selected node' : query ? ' · search includes all types' : ''}`;
  document.getElementById('overview-note').hidden = !graph.nodes.some(node => node.type === 'Field' && !visible.has(node.id));
  document.getElementById('empty').hidden = shownNodes.length > 0;
  document.getElementById('clear-focus').disabled = !focus;
  list.replaceChildren();
  const listed = query ? matches : shownNodes;
  listed.slice().sort((a, b) => kinds.indexOf(a.type) - kinds.indexOf(b.type) || a.label.localeCompare(b.label)).forEach(node => {
    const button = document.createElement('button'); button.type = 'button'; button.dataset.node = node.id;
    button.textContent = `${node.type} · ${node.label}${node.missing ? ' · missing' : ''}`;
    button.classList.toggle('selected', node.id === selected);
    button.addEventListener('click', () => { if (!positions.has(node.id)) { focus = node.id; draw(); } selectNode(node, true); });
    list.append(button);
  });
  reset();
}
filters.forEach(input => input.addEventListener('change', () => { focus = null; draw(); }));
search.addEventListener('input', () => { focus = null; draw(); });
document.getElementById('edge-labels').addEventListener('change', () => svg.classList.toggle('show-labels', document.getElementById('edge-labels').checked));
document.getElementById('zoom-in').addEventListener('click', () => zoom(1.25));
document.getElementById('zoom-out').addEventListener('click', () => zoom(.8));
document.getElementById('reset-view').addEventListener('click', reset);
document.getElementById('focus-node').addEventListener('click', () => { if (selected) { focus = selected; search.value = ''; draw(); selectNode(nodes.get(selected), true); } });
document.getElementById('clear-focus').addEventListener('click', () => { focus = null; draw(); });
svg.addEventListener('wheel', event => { event.preventDefault(); const rect = svg.getBoundingClientRect(); zoom(event.deltaY < 0 ? 1.12 : 1 / 1.12, event.clientX - rect.left, event.clientY - rect.top); }, {passive: false});
let drag = null;
svg.addEventListener('pointerdown', event => { if (event.target.closest('.node, .edge')) return; drag = {x: event.clientX, y: event.clientY, tx, ty}; svg.setPointerCapture(event.pointerId); svg.classList.add('dragging'); });
svg.addEventListener('pointermove', event => { if (!drag) return; tx = drag.tx + event.clientX - drag.x; ty = drag.ty + event.clientY - drag.y; transform(); });
svg.addEventListener('pointerup', () => { drag = null; svg.classList.remove('dragging'); });
svg.addEventListener('pointercancel', () => { drag = null; svg.classList.remove('dragging'); });
draw();
"""


def export_ontology_html(run: Path, output: Path) -> dict:
    ontology = read_json(run / "ontology" / "ontology.json")
    graph = build_graph(ontology, run / "ontology" / "lake", output)
    counts = Counter(node["type"] for node in graph["nodes"])
    edge_counts = Counter(edge["group"] for edge in graph["edges"])
    graph_json = json.dumps(graph, ensure_ascii=False).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    filters = "".join(f'<label class="type-filter"><input type="checkbox" value="{kind}" {"" if kind == "Field" and counts[kind] > 24 else "checked"}><span class="swatch" style="background:{COLORS[kind]}"></span>{kind} ({counts[kind]})</label>' for kind in NODE_TYPES)
    stats = [f"{counts[kind]} {kind} nodes" for kind in NODE_TYPES]
    stats += [f'{edge_counts["persisted"]} persisted Links', f'{edge_counts["structural"]} structural edges']
    actual = sum(node.get("actual_file", False) for node in graph["nodes"])
    missing = sum(node["missing"] for node in graph["nodes"])
    stats += [f"{actual} actual Lake files", f"{missing} missing references"]
    document = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Ontology graph · {escaped(run.name)}</title><style>{STYLE}</style></head>
<body><header><h1>Ontology graph · {escaped(ontology.get("lake_id", run.name))}</h1><p>{escaped(run.name)}</p><p>Persisted Concepts, Schemas, Fields, Tables and Links, alongside the actual files in the saved Lake. Select a node or arrow to inspect its details.</p><div class="stats">{"".join(f'<span>{escaped(stat)}</span>' for stat in stats)}</div></header>
<div class="layout"><main class="card"><div class="toolbar">{filters}</div><div class="toolbar"><button id="zoom-in" type="button" aria-label="Zoom in">+</button><button id="zoom-out" type="button" aria-label="Zoom out">−</button><button id="reset-view" type="button">Reset view</button><button id="clear-focus" type="button" disabled>Clear focus</button><label><input id="edge-labels" type="checkbox"> Edge labels</label><span class="hint">Drag to pan · scroll to zoom · select a listed node to center it</span></div>
<div class="legend"><span><i class="line"></i>Persisted Link</span><span><i class="line structural"></i>Structural relationship</span><span><i class="line overview"></i>Derived overview through hidden Fields</span></div><p id="overview-note" class="hint" hidden>Overview hides Fields to keep labels readable. Dotted Schema → Concept arrows summarize their field_link Links with counts. Enable Field or select a node and “Show connections” to inspect the individual Fields.</p><p id="visible-count" aria-live="polite"></p><div id="empty" hidden>{"No nodes match the current filters or search." if graph["nodes"] else "No persisted ontology objects or Lake files were found. Save an ontology or add files to the run's ontology/lake directory, then export again."}</div>
<svg id="graph" role="img" aria-label="Interactive Ontology graph"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#8999af"></path></marker></defs><g id="world"></g></svg></main>
<aside class="card"><div><h2>Find a node</h2><label for="node-search">Search names, descriptions, ids and paths</label><input id="node-search" type="search" placeholder="Search all node types" autocomplete="off"><div id="node-list"></div></div><div><h2 id="detail-title">Node and edge details</h2><p id="selection-hint" class="hint">Select a node or arrow in the graph, or choose a node from the list.</p><button id="focus-node" type="button" disabled>Show connections</button><a id="detail-link" hidden></a><pre id="detail-data">File links open only actual files within this run's saved Lake. Claims and Experience are not graph facts.</pre></div></aside></div>
<script id="graph-data" type="application/json">{graph_json}</script><script>{SCRIPT}</script></body></html>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return {"html": str(output.resolve()), "bytes": output.stat().st_size, "nodes": dict(counts), "edges": dict(edge_counts), "actual_files": actual, "missing_references": missing}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Saved run directory")
    parser.add_argument("--output", type=Path, help="Defaults to RUN/ontology/ontology.html")
    args = parser.parse_args()
    run = (WORKSPACE / args.run).resolve()
    output = (WORKSPACE / args.output).resolve() if args.output else run / "ontology" / "ontology.html"
    print(json.dumps(export_ontology_html(run, output), indent=2))


if __name__ == "__main__":
    main()
