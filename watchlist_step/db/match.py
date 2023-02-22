from typing import List, Tuple


def format_values_for_query(coordinates: List[Tuple]) -> str:
    """
    Transforms a list of tuples into a string that can be
    concatenated as the VALUES of a SQL Query
    """
    return ",\n".join(map(str, coordinates))


def create_match_query(values: str, base_radius=30 / 3600):
    """
    Returns a SQL Query that selects all targets found within
    the coordinates provided as values
    """
    return (
        f"""
        WITH positions (ra, dec, oid, candid) AS (
                VALUES
                %s
        )
        SELECT
        positions.oid,
        positions.candid,
        watchlist_target.id
        FROM watchlist_target, positions
        WHERE q3c_join(positions.ra, positions.dec,watchlist_target.ra, \
            watchlist_target.dec, {base_radius})
        AND q3c_dist(positions.ra, positions.dec, watchlist_target.ra, \
            watchlist_target.dec) < watchlist_target.radius
    """
        % values
    )


def create_insertion_query(values):
    """
    Returns a SQL Query that inserts all targets provided in
    the values string into the watchlist matches
    """

    return f"""
    INSERT INTO watchlist_match (target_id, object_id, candid, date) VALUES {values};
    """
