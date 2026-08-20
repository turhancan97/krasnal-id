# Local data

This directory is reserved for downloaded Wikimedia images, validated manifests, and cached
embeddings. Generated contents are ignored by Git and are not managed with DVC.

Every image admitted to a manifest must include `source_url`, `author`, `license`, and
`license_url`. Do not place images with unknown or incompatible licensing here.

`category-review.json` is the tracked category-control artifact. It records human approval,
rejection, optional category corrections, display-name overrides, and notes for every
discovered mapping. Never auto-approve new or changed mappings.

`image-review.json` is the tracked image-control artifact. It records explicit retain or
exclude decisions for image-level exceptions, keyed by dwarf ID and Commons page ID. Its
staging hash must match the canonical `fetched-images.json` artifact before downstream code
uses the decisions. Unlisted images retain the acquisition pipeline's decision.

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

Manifest construction reads only the validated `fetched-images.json` records and writes the
ignored `data/manifest.json` artifact. It never scans the local image directory. The manifest
records the discovery query, fetched staging, and image-review provenance hashes.
