# Local data

This directory is reserved for downloaded Wikimedia images, validated manifests, and cached
embeddings. Generated contents are ignored by Git and are not managed with DVC.

Every image admitted to a manifest must include `source_url`, `author`, `license`, and
`license_url`. Do not place images with unknown or incompatible licensing here.

Wikidata discovery owns the ignored `data/discovery/` directory:

- `wikidata-response.json` stores the complete cached SPARQL response.
- `wikidata-response.meta.json` stores endpoint, query-hash, and retrieval provenance.
- `dwarfs.json` stores deterministic normalized records for the next pipeline stage.
- `audit.json` stores deterministic exclusions and manual-review warnings.
