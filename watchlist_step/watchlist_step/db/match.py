from psycopg2.sql import SQL, Literal


def create_match_query(len, base_radius=30 / 3600):
    """
    Returns a SQL Query that selects all targets found within
    the coordinates provided as values
    """

    return SQL(
        """
        SELECT
            positions.oid,
            positions.candid,
            watchlist_target.id,
            watchlist_target.filter
        FROM
            watchlist_target,
            (VALUES {}) AS positions(ra, dec, oid, candid)
        WHERE
            q3c_join(
                positions.ra,
                positions.dec,
                watchlist_target.ra,
                watchlist_target.dec,
                LEAST(watchlist_target.radius, {})
            )
        """
    ).format(
        SQL(", ").join(SQL("(%s, %s, %s, %s)") for _ in range(len)),
        Literal(base_radius),
    )


def create_insertion_query():
    """
    Returns a SQL Query that inserts all targets provided in
    the values string into the watchlist matches
    """

    return SQL(
        """
        INSERT INTO
            watchlist_match(target_id, object_id, candid, values, date, ready_to_notify)
        VALUES (%s, %s, %s, %s, %s, false)
        """
    )


def update_match_query():
    """
    Returns a SQL Query that updates the filter values from the provided match
    """

    return SQL(
        """
        UPDATE
            watchlist_match AS wl_match
        SET
            values = wl_match.values || %(values)s
        WHERE
            wl_match.object_id = %(oid)s
            AND wl_match.candid = %(candid)s
            AND wl_match.target_id = %(target_id)s
        RETURNING
            wl_match.object_id,
            wl_match.candid,
            wl_match.target_id,
            wl_match.values
        """
    )


def update_for_notification():
    """
    Returns a SQL Query that marks the provided match as ready to notify
    """

    return SQL(
        """
        UPDATE
            watchlist_match AS wl_match
        SET
            ready_to_notify = true
        WHERE
            wl_match.object_id = %(oid)s
            AND wl_match.candid = %(candid)s
            AND wl_match.target_id = %(target_id)s
        """
    )
