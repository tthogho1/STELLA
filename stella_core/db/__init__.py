"""
Explicit database initialization.

stella_core no longer picks a backend on its own as a side effect of being imported.
Constructing one can mean opening a real MongoDB connection, and doing that automatically
at whatever moment something else happened to `import stella_core.db` was surprising and
made it hard to use any db-free piece of the runtime (or test one) without a database
available. The host application calls init_database() once, explicitly, at startup (see
app/server.py); nothing below chooses a backend on its own.

`db` is a proxy rather than the backend instance itself, so that every module which does
`from stella_core.db import db` at its own import time (Task, TaskManager, ChatQueue, the
Flask views) keeps working after init_database() runs -- or runs again, as tests do to
swap in a throwaway backend per test case -- without needing to be re-imported.
"""
import os

from stella_core.db.database_interface import DatabaseInterface


class DatabaseFactory:
    @staticmethod
    def get_database():
        db_type = os.getenv('DATABASE')

        # Imported here rather than at module scope, so that choosing "sqlite" does not
        # also require pymongo to be installed (and vice versa).
        if db_type == 'mongodb':
            from stella_core.db.databases.mongodb import MongoDB
            return MongoDB()
        elif db_type == 'sqlite':
            from stella_core.db.databases.sqlite import SQLite
            return SQLite()
        else:
            raise Exception(f"Unsupported database type: {db_type}")


class _DatabaseProxy:
    """Forwards every call to whichever backend init_database() last set."""

    def __getattr__(self, name):
        if _current is None:
            raise RuntimeError(
                "stella_core.db is not initialized -- call stella_core.db.init_database() "
                "before using it (app/server.py does this at startup)."
            )
        return getattr(_current, name)


_current = None
db = _DatabaseProxy()


def init_database(backend: DatabaseInterface = None) -> DatabaseInterface:
    """
    Selects the backend `db` forwards to.
    :param backend: Use this backend directly. Omit to build one from the DATABASE
                    environment variable via DatabaseFactory, STELLA's own default.
    :return: The backend now in effect.
    """
    global _current
    _current = backend if backend is not None else DatabaseFactory.get_database()
    return _current
