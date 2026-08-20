"""Shared plumbing for the offline batched upserts.

`feature_writer`, `probability_writer` and `xmatch.persist_matches` all do the
same thing to different tables: fold a unit's worth of rows into one statement
per page with `psycopg2.extras.execute_values`. The duplicate-key check below
lived in two of them as a copy-paste; adding a third user is what moved it here.
"""


# Rows per INSERT statement. execute_values folds this many tuples into one
# VALUES list, so the cost is one round trip per page instead of one per row.
PAGE_SIZE = 1000


def assert_no_duplicate_keys(records, key_fields, table) -> None:
    """Refuse a batch that carries the same primary key twice.

    execute_values puts many rows in ONE statement, and Postgres rejects an
    ON CONFLICT DO UPDATE that would touch the same row twice ("cannot affect
    row a second time"). Under the old per-row executemany this was impossible,
    so the batching introduces the failure mode -- and it always means the
    caller assembled the rows wrong, which is worth naming rather than passing
    to the driver.
    """
    seen = set()
    for r in records:
        key = tuple(r[f] for f in key_fields)
        if key in seen:
            raise ValueError(
                f"duplicate {table} key {dict(zip(key_fields, key))} in one write batch"
            )
        seen.add(key)
