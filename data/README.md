# Local data

This directory is reserved for downloaded Wikimedia images, validated manifests, and cached
embeddings. Generated contents are ignored by Git and are not managed with DVC.

Every image admitted to a manifest must include `source_url`, `author`, `license`, and
`license_url`. Do not place images with unknown or incompatible licensing here.

`category-review.json` is the one tracked data-control artifact. It records human approval,
rejection, optional category corrections, and notes for every discovered mapping. Never
auto-approve new or changed mappings.

Wikidata discovery owns the ignored `data/discovery/` directory:

- `wikidata-response.json` stores the complete cached SPARQL response.
- `wikidata-response.meta.json` stores endpoint, query-hash, and retrieval provenance.
- `dwarfs.json` stores deterministic normalized records for the next pipeline stage.
- `audit.json` stores deterministic exclusions and manual-review warnings.

Commons acquisition adds these ignored artifacts:

- `commons/<QID>.json` and `commons/<QID>.meta.json` cache complete paginated API data.
- `fetched-images.json` stores validated attribution, checksums, dimensions, and source
  revision provenance for images eligible for manifest construction.
- `fetch-audit.json` records policy exclusions, duplicates, warnings, and operational errors.
- `data/images/<QID>/` contains atomically downloaded, Pillow-verified research copies.
