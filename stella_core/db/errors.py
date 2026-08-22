"""
Errors the database layer raises.

Both backends used to signal a missing row with a bare Exception whose message ended in
"not found". A caller could not tell that apart from a locked SQLite file or an
unreachable mongod, so every failure became a 404 and the view told the user the object
did not exist -- while putting the driver's error text in the response body.
"""


class NotFound(Exception):
    """The row does not exist. Anything else means the lookup itself failed."""
