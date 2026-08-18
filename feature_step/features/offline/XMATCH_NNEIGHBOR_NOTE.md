# Xmatch client drops AllWISE — Xwave `catalog` arg is the fix

> **STATUS (2026-07-14): FIXED.** `libs/xmatch_client` now forwards a `catalog`
> argument to `/v1/bulk-conesearch` (`conesearch[_with_metadata](..., catalog=...)`),
> so the server scopes per catalog *before* the KNN. The live `feature_step`
> (`step.get_xmatch_info`) issues one request per catalog in `XMATCH_CATALOGS`
> (default `["allwise","gaia"]`), and the offline path (`features/offline/xmatch.py
> ::compute_matches`) mirrors it — one request per catalog with `catalog=<name>`.
> The diagnosis below is retained for the record; the "Fix" section is now
> implemented, not proposed.

## Calling the client (minimum)

```python
from xmatch_client import XmatchClient
client = XmatchClient(base_url="http://quimal-db1.alerce.online:8081")
client.conesearch(ras=[75.825020], decs=[42.212812],
                  oids=["36028933559755080"], radius=1.5, catalogs=["allwise"])
```

**Result:** `[]` — no AllWISE.

## Why

`XmatchClient.conesearch` hardcodes `nneighbor: 1` and sends **no** `catalog`, so Xwave
returns the single **global** nearest across all catalogs:

```python
payload = {
    "oids": oid_batch,
    "ra": ra_batch,
    "dec": dec_batch,
    "radius": radius,
    "nneighbor": 1,
}
```

then filters that one result client-side, which can only remove, never restore:

```python
if catalogs_set and catalog_name not in catalogs_set:
    continue
```

At this position Gaia is nearer, so AllWISE never survives:

| catalog | id | dist |
|---|---|---|
| gaia | 201794094900626176 | 0.107" |
| allwise | 0748p424_ac51-023838 | 0.184" |

## Observation: Xwave exposes a `catalog` argument

`/v1/bulk-conesearch` accepts a `catalog` field:

```go
type BulkConesearchRequest struct {
    Ra        []float64 `json:"ra"`
    Dec       []float64 `json:"dec"`
    Radius    float64   `json:"radius"`
    Catalog   string    `json:"catalog"`
    Nneighbor int       `json:"nneighbor"`
}
```

and the server filters candidates by catalog **before** KNN, so `nneighbor` becomes
per-catalog:

```go
func (c *ConesearchService) getObjectsInRanges(pixelRanges []healpix.PixelRange, catalog string) ([]repository.Mastercat, error) {
    objects, err := c.store.FindObjectsInPixelRanges(c.ctx, pixelRanges)
    ...
    if catalog != "all" {
        return filterByCatalog(objects, catalog), nil
    }
    return objects, nil
}
```

Passing `catalog="allwise"` returns AllWISE's own nearest directly — no need to raise
`nneighbor` and post-filter:

```python
requests.post(".../v1/bulk-conesearch", json={
    "oids": ["36028933559755080"], "ra": [75.825020], "dec": [42.212812],
    "radius": 1.5, "nneighbor": 1, "catalog": "allwise"})
# -> allwise 0748p424_ac51-023838  dist=0.184"
```

**Fix:** have `XmatchClient.conesearch` forward `catalog` to the payload instead of
filtering client-side. Cleaner than `nneighbor>=N` + dedupe. Note: `catalog` is a single
string, so covering N catalogs (each with its own nearest) would mean N calls, one per
catalog.
