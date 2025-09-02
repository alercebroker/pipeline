This generator is designed to produce synthetic LSST alert-like objects for internal testing and development.

# Assumptions

- Field values are randomly generated, with some fields occasionally set to `None` to simulate missing data, as allowed by the schema.
- Unique IDs are generated using a custom hash function to avoid collisions and ensure reproducibility.
- Only the most recent N (30 or 50) sources/non-detections are retained per object to limit memory usage and match typical alert payloads.
- The generator does not enforce physical or astrophysical realism; values are chosen uniformly within plausible ranges.
- Filter bands are always one of ("u", "g", "r", "i", "z", "y"), matching schema expectations.
