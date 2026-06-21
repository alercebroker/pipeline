#!/usr/bin/env python
"""Compare offline BHRF probabilities vs. stored probabilities. DEFERRED skeleton.

Pending the stored-probability table definition. Once db.fetch_stored_probabilities
is implemented, this will mirror offline_compare_vs_alerce.py for probabilities:
run the offline pipeline on an oid, fetch the stored probabilities, and diff.
"""
import sys


def main():
    print(
        "offline_compare_probabilities is not implemented yet: pending the "
        "stored-probability table definition (see the offline classification design doc)."
    )
    sys.exit(2)


if __name__ == "__main__":
    main()
