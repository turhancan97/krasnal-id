# Local data

This directory is reserved for downloaded Wikimedia images, validated manifests, and cached
embeddings. Generated contents are ignored by Git and are not managed with DVC.

Every image admitted to a manifest must include `source_url`, `author`, `license`, and
`license_url`. Do not place images with unknown or incompatible licensing here.
