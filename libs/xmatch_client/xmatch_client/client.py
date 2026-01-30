from collections import defaultdict
from typing import Any, Literal, TypedDict

from requests import Response, Session

from xmatch_client.utils import batched


class Match(TypedDict):
    oid: str
    query_ra: float
    query_dec: float
    match_id: str
    catalog: str
    match_ra: float
    match_dec: float
    distance: float


Metadata = dict[str, Any]


class MatchWithMetadata(Match):
    metadata: Metadata


class XmatchClient:
    session: Session
    base_url: str
    batch_size: int

    def __init__(self, base_url: str, batch_size: int = 500):
        """Initialize the XmatchClient.

        Args:
            base_url: Base url of the xmatch service.
            batch_size: Number of queries per batch.
        """
        self.base_url = base_url
        self.batch_size = batch_size
        self.session = Session()

    def _request(
        self,
        method: Literal["GET", "POST"],
        endpoint: str,
        json: dict[str, Any],
        timeout: int = 30,
    ) -> Response:
        """Send an HTTP request to the xmatch service.

        Args:
            method: HTTP method ("GET" or "POST").
            endpoint: API endpoint.
            json: JSON payload.
            timeout: Request timeout in seconds.

        Returns:
            Response object from requests.
        """
        return self.session.request(
            method, f"{self.base_url}/{endpoint}", json=json, timeout=timeout
        )

    @staticmethod
    def _parse_matches(
        oid: str,
        query_ra: float,
        query_dec: float,
        catalog_name: str,
        matches: dict[str, Any],
    ) -> list[Match]:
        """Convert raw match data to a list of Match dicts.

        Args:
            oid: Object ID.
            query_ra: Query right ascension.
            query_dec: Query declination.
            catalog_name: Name of the catalog.
            matches: Raw match data.

        Returns:
            List of Match dictionaries.
        """
        return [
            Match(
                oid=oid,
                query_ra=query_ra,
                query_dec=query_dec,
                match_id=match["id"],
                catalog=catalog_name,
                match_ra=match["ra"],
                match_dec=match["dec"],
                distance=match["distance"],
            )
            for match in matches
        ]

    def conesearch(
        self,
        ras: list[float],
        decs: list[float],
        oids: list[str | int],
        radius: float = 1.5,
        catalogs: list[str] | None = None,
    ):
        """Perform a bulk cone search for multiple positions.

        Args:
            ras: List of right ascensions.
            decs: List of declinations.
            oids: List of object IDs.
            radius: Search radius in arcseconds.
            catalogs: List of catalog names to search.

        Returns:
            List of Match dictionaries.
        """
        if not len(ras) == len(decs) == len(oids):
            raise ValueError("oids, ras and decs must have the same length")

        catalogs_set = set(catalogs) if catalogs is not None else None

        matches: list[Match] = []
        for oid_batch, ra_batch, dec_batch in zip(
            batched(oids, self.batch_size),
            batched(ras, self.batch_size),
            batched(decs, self.batch_size),
        ):
            payload = {
                "oids": oid_batch,
                "ra": ra_batch,
                "dec": dec_batch,
                "radius": radius,
                "nneighbor": 1,
            }

            resp = self._request("POST", "v1/bulk-conesearch", json=payload)
            resp.raise_for_status()

            query_results = resp.json()
            for result in query_results:
                oid = result["Oid"]
                query_ra = result["QueryRA"]
                query_dec = result["QueryDec"]
                catalogs = result["Data"]

                for catalog in catalogs:
                    catalog_name = catalog["catalog"]
                    catalog_matches = catalog["data"]
                    if catalogs_set and catalog_name not in catalogs_set:
                        continue

                    matches += self._parse_matches(
                        oid, query_ra, query_dec, catalog_name, catalog_matches
                    )

        return matches

    def metadata(
        self, ids: list[str], catalogs: list[str]
    ) -> dict[str, dict[str, Metadata]]:
        """Fetch metadata for a list of IDs and catalogs.

        Args:
            ids: List of source IDs.
            catalogs: List of catalog names.

        Returns:
            Dictionary mapping catalog names to dictionaries of ID-metadata pairs.
        """
        if not len(ids) == len(catalogs):
            raise ValueError("ids and catalogs must have the same length")

        ids_by_cat = defaultdict(list)
        for id, catalog in zip(ids, catalogs):
            ids_by_cat[catalog].append(id)

        metadata_by_cat_by_id = {}
        for catalog, ids in ids_by_cat.items():
            for id_batch in batched(ids, self.batch_size):
                payload = {"ids": id_batch, "catalog": catalog}

                resp = self._request("POST", "v1/bulk-metadata", json=payload)
                resp.raise_for_status()

                metadatas = resp.json()
                metadata_by_cat_by_id[catalog] = {
                    metadata["id"]: metadata for metadata in metadatas
                }

        return metadata_by_cat_by_id

    def conesearch_with_metadata(
        self,
        ras: list[float],
        decs: list[float],
        oids: list[str | int],
        radius: float = 1.5,
        catalogs: list[str] | None = None,
    ):
        """Perform a cone search and fetch metadata for the matches.

        Args:
            ras: List of right ascensions.
            decs: List of declinations.
            oids: List of object IDs.
            radius: Search radius in arcseconds.
            catalogs: List of catalog names to search.

        Returns:
            List of MatchWithMetadata dictionaries.
        """
        matches = self.conesearch(ras, decs, oids, radius, catalogs)

        matches_id = []
        matches_catalog = []
        for match in matches:
            matches_id.append(match["match_id"])
            matches_catalog.append(match["catalog"])

        metadata_by_cat_by_id = self.metadata(ids=matches_id, catalogs=matches_catalog)

        matches_with_metadata = [
            MatchWithMetadata(
                **match,
                metadata=metadata_by_cat_by_id[match["catalog"]][match["match_id"]],
            )
            for match in matches
        ]

        return matches_with_metadata
