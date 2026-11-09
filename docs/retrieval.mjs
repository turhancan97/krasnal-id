/** Vector-only retrieval shared by the browser and metadata calibration. */
export function rankStatues(query, vectors, images, excludeIndex = -1) {
  const best = new Map();
  for (let i = 0; i < vectors.length; i += 1) {
    if (i === excludeIndex) continue;
    const reference = vectors[i];
    let cosine = 0;
    for (let d = 0; d < query.length; d += 1) cosine += query[d] * reference[d];
    const image = images[i];
    const previous = best.get(image.dwarf);
    if (!previous || cosine > previous.cosine) best.set(image.dwarf, { cosine, score: cosine, image });
  }
  // Stable sort preserves the existing first-reference order for exact cosine ties.
  return [...best.values()].sort((a, b) => b.cosine - a.cosine);
}

export function distanceMetres(a, b) {
  const rad = (degrees) => degrees * Math.PI / 180;
  const dLat = rad(b.lat - a.lat);
  const dLon = rad(b.lon - a.lon);
  const inner = Math.sin(dLat / 2) ** 2 +
    Math.cos(rad(a.lat)) * Math.cos(rad(b.lat)) * Math.sin(dLon / 2) ** 2;
  return 2 * 6371008.8 * Math.asin(Math.sqrt(Math.min(1, inner)));
}

export function weightByLocation(ranked, dwarfs, position, prior) {
  if (!position || prior.weight === 0) return ranked.map((hit) => ({ ...hit, score: hit.cosine }));
  if (prior.kernel !== 'exponential' || !(prior.scale_m > 0) || !(prior.weight >= 0)) {
    throw new Error('Unsupported location prior');
  }
  return ranked.map((hit, order) => {
    const dwarf = dwarfs.get(hit.image.dwarf);
    const placed = Number.isFinite(dwarf?.lat) && Number.isFinite(dwarf?.lon);
    const bonus = placed ? prior.weight * Math.exp(-distanceMetres(position, dwarf) / prior.scale_m) : 0;
    return { ...hit, score: hit.cosine + bonus, order };
  }).sort((a, b) => b.score - a.score || a.order - b.order);
}
