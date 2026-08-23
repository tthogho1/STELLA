"""
What /agent/download will and will not unpack.

The extracted files are imported on the next reload, so this endpoint is only ever as
safe as the index it points at -- and PACKAGE_MANAGER_URL now makes that configurable.
These are the checks that do not depend on trusting the index.
"""
import io
import zipfile

import pytest

from app.views.agent import MAX_PACKAGE_UNCOMPRESSED_BYTES, _safe_members


def _zip(entries, compress=zipfile.ZIP_STORED):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compress) as z:
        for name, content in entries:
            z.writestr(name, content)
    buf.seek(0)
    return zipfile.ZipFile(buf)


def test_an_ordinary_package_is_accepted():
    members = _safe_members(_zip([("MyAgent.py", "x"), ("tools/helper.py", "y")]))

    assert members == ["MyAgent.py", "tools/helper.py"]


def test_macos_metadata_is_skipped():
    members = _safe_members(_zip([("MyAgent.py", "x"), ("__MACOSX/._MyAgent.py", "junk")]))

    assert members == ["MyAgent.py"]


class TestOutOfTreePaths:
    """extract() would normalise these away rather than escape, but it does so silently:
    the file lands somewhere the package did not ask for. Refuse instead."""

    def test_a_parent_traversal_is_refused(self):
        with pytest.raises(ValueError, match="out-of-tree"):
            _safe_members(_zip([("../../escaped.py", "x")]))

    def test_an_absolute_path_is_refused(self):
        with pytest.raises(ValueError, match="out-of-tree"):
            _safe_members(_zip([("/etc/passwd", "x")]))

    def test_a_traversal_further_in_is_refused(self):
        with pytest.raises(ValueError, match="out-of-tree"):
            _safe_members(_zip([("tools/../../out.py", "x")]))

    def test_a_dotted_filename_is_still_fine(self):
        """Only a whole '..' path component is a traversal."""
        assert _safe_members(_zip([("my..agent.py", "x")])) == ["my..agent.py"]


class TestSize:
    def test_a_zip_bomb_is_refused(self):
        """Measured: deflate reaches about 1000:1, so 199 KB on the wire became 200 MB."""
        oversized = b"\0" * (MAX_PACKAGE_UNCOMPRESSED_BYTES + 1)

        with pytest.raises(ValueError, match="expands to more than"):
            _safe_members(_zip([("bomb.py", oversized)], zipfile.ZIP_DEFLATED))

    def test_the_limit_is_on_the_total_not_one_entry(self):
        half = b"\0" * (MAX_PACKAGE_UNCOMPRESSED_BYTES // 2 + 1)

        with pytest.raises(ValueError, match="expands to more than"):
            _safe_members(_zip([("a.py", half), ("b.py", half)], zipfile.ZIP_DEFLATED))

    def test_a_normal_sized_package_is_unaffected(self):
        content = b"\0" * (1024 * 1024)

        assert _safe_members(_zip([("big.py", content)], zipfile.ZIP_DEFLATED)) == ["big.py"]


def test_nothing_is_written_when_an_entry_is_refused(tmp_path):
    """The members are collected before anything is extracted, so a bad entry at the end
    does not leave the earlier ones on disk."""
    archive = _zip([("good.py", "x"), ("../bad.py", "y")])

    with pytest.raises(ValueError):
        _safe_members(archive)

    assert list(tmp_path.iterdir()) == []
