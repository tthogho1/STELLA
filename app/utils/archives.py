"""
Reading zip archives that came from somewhere else.

Two endpoints unpack archives -- GET /agent/download, which fetches a package from the
package manager, and POST /workspace/<id>/source, which takes one straight from the user
-- and neither can trust what it is handed. The checks are the same for both, so they
live here rather than being written out twice with the limits quietly drifting apart.
"""
import os
import shutil


def safe_members(archive, max_bytes, max_files=None):
    """
    The entries of `archive` that are safe to extract, or ValueError explaining why not.

    ZipFile.extract() already strips leading slashes and ".." components, so an entry
    cannot escape the target directory. It does so silently though, which quietly moves
    a file somewhere the archive did not intend -- better to refuse the archive and say
    so. The size ceiling is the part that actually protects anything.

    Directory entries are dropped: extract() and extract_members() both create the
    parents they need, so keeping them only made the file count wrong.
    """
    members = []
    total = 0
    for info in archive.infolist():
        name = info.filename
        if "__MACOSX/" in name:
            continue
        if name.startswith(("/", "\\")) or ".." in name.replace("\\", "/").split("/"):
            raise ValueError(f"the archive contains an out-of-tree path ({name})")
        if info.is_dir():
            continue
        total += info.file_size
        if total > max_bytes:
            raise ValueError(
                f"the archive expands to more than {max_bytes // (1024 * 1024)} MB")
        members.append(name)
        if max_files is not None and len(members) > max_files:
            raise ValueError(f"the archive holds more than {max_files} files")
    return members


def extract_members(archive, members, destination):
    """
    Writes `members` under `destination`, keeping the archive's own layout.

    Nothing is stripped or rearranged. A zip made from a project directory carries a
    wrapper directory (`orders-main/...` from GitHub, the folder name from the Finder),
    and dropping it is tempting -- but a single top-level directory is also what an
    archive of one package looks like, and there is no way to tell the two apart. Guessing
    wrong turns the path the user names in chat into one that does not exist, which is a
    worse outcome than a longer path. The caller reports what the top level is instead.

    Written out rather than handed to ZipFile.extract() so the containment check is
    explicit: safe_members() already refused traversal, but this is the last point where
    the path that will actually be opened can be compared against the directory it has
    to stay in.
    """
    root = os.path.abspath(destination)
    written = 0
    for name in members:
        target = os.path.abspath(os.path.join(root, *name.split("/")))
        if not target.startswith(root + os.sep):
            raise ValueError(f"the archive contains an out-of-tree path ({name})")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with archive.open(name) as source, open(target, "wb") as handle:
            shutil.copyfileobj(source, handle)
        written += 1
    return written
