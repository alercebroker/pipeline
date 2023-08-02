from math import ceil


class Pagination:
    """Paginate responses from the database."""

    def __init__(self, query, page, per_page, total, items):
        """Set attributes from args."""
        self.query = query
        self.page = page
        self.per_page = per_page
        self.total = total
        self.items = items

    @property
    def pages(self):
        """Get total number of pages."""
        if self.per_page == 0 or self.total is None:
            pages = 0
        else:
            pages = int(ceil(self.total / float(self.per_page)))
        return pages

    def prev(self):
        """Return a :class:`Pagination` object for the previous page."""
        assert (
            self.query is not None
        ), "a query object is required for this method to work"
        return self.query.paginate(self.page - 1, self.per_page)

    @property
    def prev_num(self):
        """Get number of the previous page."""
        if not self.has_prev:
            return None
        return self.page - 1

    @property
    def has_prev(self):
        """Check if a previous page exists."""
        return self.page > 1

    def next(self):
        """Return a :class:`Pagination` object for the next page."""
        assert (
            self.query is not None
        ), "a query object is required for this method to work"
        return self.query.paginate(self.page + 1, self.per_page)

    @property
    def has_next(self):
        """Check if a next page exists."""
        return self.page < self.pages

    @property
    def next_num(self):
        """Get number of the next page."""
        if not self.has_next:
            return None
        return self.page + 1


class PaginationNoCount(Pagination):
    def __init__(self, query, page, per_page, items, has_next):
        super().__init__(query, page, per_page, None, items)
        self._has_next = has_next

    @property
    def has_next(self):
        """Check if a previous page exists."""
        return self._has_next
