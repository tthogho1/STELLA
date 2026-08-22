"""
Shared helpers for Flask views.

Every REST view in app/views follows the same "load an owned object or fail" shape:
fetch something from the db, return 404 if it is missing, then return 403 if the caller
(the JWT identity) isn't the owner. That pattern was previously hand-copied into every
route (see git history of app/views/chat.py, workspace.py, user.py), which meant the
404/403 status codes and message wording drifted between copies. get_owned_or_404
centralizes it so there is exactly one place that decides those status codes.
"""
from typing import Callable, Optional, Tuple

from flask import jsonify

from stella_core.db.errors import NotFound


def get_owned_or_404(
    getter: Callable[[str], object],
    obj_id: str,
    user_id: str,
    owner_attr: str = "owner",
    not_owner_msg: str = "User does not have access",
) -> Tuple[Optional[object], Optional[tuple]]:
    """
    Fetches an object with `getter(obj_id)` and checks that `user_id` owns it.

    Returns (obj, None) on success, or (None, response) on failure -- where `response`
    is the (jsonify(...), status_code) tuple a Flask view can return directly. Callers
    look like:

        obj, err = get_owned_or_404(db.get_workspace, workspace_id, user_id)
        if err:
            return err
    """
    try:
        obj = getter(obj_id)
    except NotFound as e:
        return None, (jsonify({"msg": str(e)}), 404)
    except Exception as e:
        # A locked database or an unreachable server is not a missing row. Reporting it
        # as 404 told the user the workspace did not exist and put the driver's error
        # text in the response; log it and say nothing specific instead.
        print(f"[VIEW] !! Lookup failed for {obj_id} ({type(e).__name__}: {e})")
        return None, (jsonify({"msg": "Could not read from the database"}), 500)

    if getattr(obj, owner_attr) != user_id:
        return None, (jsonify({"msg": not_owner_msg}), 403)

    return obj, None
