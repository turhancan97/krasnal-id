/** Leaflet and its CSS are fetched only when a result needs a map. */
import { distanceMetres } from './retrieval.mjs';
const CDN = 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/';
const INTEGRITY = {
  css: 'sha512-Zcn6bjR/8RZbLEpLIeOwNtzREBAJnUKESxces60Mpoj+2okopSAcSUIUOseddDm0cxnGQzxIR7vJgsLZbdLE3w==',
  js: 'sha512-BwHfrr4c9kmRkLw6iXFdzcdWV/PGkVgiIyIWLLlTSXzWQzxuSg4DiQUCpauz/EWjgk5TYQqX/kvn9pG1NpYfqg==',
};
let loading, map, layers, version = 0;
function loadLeaflet() {
  if (loading) return loading;
  const tags = [];
  const load = (kind) => new Promise((resolve, reject) => {
    const tag = document.createElement(kind === 'css' ? 'link' : 'script');
    tags.push(tag);
    if (kind === 'css') { tag.rel = 'stylesheet'; tag.href = `${CDN}leaflet.css`; }
    else { tag.src = `${CDN}leaflet.js`; tag.async = true; }
    tag.integrity = INTEGRITY[kind]; tag.crossOrigin = 'anonymous';
    const timeout = setTimeout(() => reject(new Error('Map library timed out')), 15000);
    tag.onload = () => { clearTimeout(timeout); resolve(); };
    tag.onerror = () => { clearTimeout(timeout); reject(new Error('Map library unavailable')); };
    document.head.append(tag);
  });
  loading = Promise.all([load('css'), load('js')]).then(() => window.L).catch((error) => {
    tags.forEach((tag) => tag.remove()); loading = null; throw error;
  });
  return loading;
}
function label(text) { const node = document.createElement('span'); node.textContent = text; return node; }

export async function renderMatchesMap(hits, metadata, position) {
  const request = ++version;
  const status = document.getElementById('map-status');
  status.textContent = 'Loading map…';
  try {
    const L = await loadLeaflet();
    if (request !== version) return;
    if (!map) {
      map = L.map('matches-map', { scrollWheelZoom: false });
      L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a>',
      }).addTo(map);
      layers = L.layerGroup().addTo(map);
    }
    layers.clearLayers(); map.invalidateSize();
    const dwarfs = new Map(metadata.dwarfs.map((d) => [d.id, d]));
    const placed = (d) => Number.isFinite(d?.lat) && Number.isFinite(d?.lon);
    const points = [];
    hits.forEach((hit, index) => {
      const d = dwarfs.get(hit.image.dwarf);
      if (!placed(d)) return;
      points.push([d.lat, d.lon]);
      const estimated = d.coordinate_source === 'commons_camera';
      const icon = L.divIcon({
        className: 'match-marker', iconSize: [30, 30], iconAnchor: [15, 15],
        html: `<span class="match-pin ${estimated ? 'estimated' : 'stated'} ${index === 0 ? 'first' : ''}" data-rank="${index + 1}">${index + 1}</span>`,
      });
      L.marker([d.lat, d.lon], { icon, title: `${index + 1}. ${d.name}`, zIndexOffset: index === 0 ? 1000 : 100-index })
        .bindPopup(label(`${index + 1}. ${d.name} — ${estimated ? 'position estimated from photographs' : 'Wikidata position'}`)).addTo(layers);
    });
    const group = metadata.co_located_groups.find((g) => g.includes(hits[0]?.image.dwarf));
    if (group) {
      const members = group.map((id) => dwarfs.get(id)).filter(placed);
      if (members.length) {
        const centre = { lat: members.reduce((s,d)=>s+d.lat,0)/members.length, lon: members.reduce((s,d)=>s+d.lon,0)/members.length };
        const radius = Math.max(25, ...members.map((d)=>distanceMetres(centre,d)+5));
        L.circle([centre.lat, centre.lon], { radius, color:'#B87A12', fillColor:'#EAB34D', fillOpacity:0.18, weight:2, dashArray:'4 4' })
          .bindPopup(label(`Nearby installation: ${members.map((d)=>d.name).join(', ')}`))
          .bindTooltip(`${members.length} statues here`, { permanent:true, direction:'top', offset:[0,-22], className:'group-map-label' }).addTo(layers);
        const shown = new Set(hits.map((h)=>h.image.dwarf));
        for (const d of members) {
          points.push([d.lat,d.lon]);
          if (!shown.has(d.id)) L.circleMarker([d.lat,d.lon], { radius:5, color:'#57606A', fillOpacity:0.7, dashArray:d.coordinate_source==='commons_camera'?'2 2':null })
            .bindPopup(label(`${d.name} — same installation`)).addTo(layers);
        }
      }
    }
    if (position) {
      // GPS stays in memory: Leaflet only puts a local marker on the map. Tile
      // requests disclose the viewed tile area, as the adjacent privacy text says.
      L.circleMarker([position.lat, position.lon], { radius:7, color:'#fff', weight:2, fillColor:'#2578AF', fillOpacity:1, className:'user-position' })
        .bindPopup(label('Your position')).addTo(layers);
      points.push([position.lat,position.lon]);
    }
    if (points.length) map.fitBounds(points, { padding:[30,40], maxZoom:18 });
    else map.setView([51.11,17.03],12);
    const count=hits.filter((h)=>placed(dwarfs.get(h.image.dwarf))).length;
    status.textContent = count < hits.length ? `${count} of ${hits.length} matches have a known position.` : (group ? 'The shaded area holds statues that stand together; zoom in.' : '');
  } catch (error) {
    if (request === version) status.textContent = 'Map unavailable. The photograph matches above still work.';
  }
}
