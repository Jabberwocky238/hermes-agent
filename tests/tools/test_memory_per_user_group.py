"""Tests for per-user / per-group memory bucketing.

The built-in MemoryStore wrote MEMORY.md/USER.md to a single profile-scoped
directory, so on multi-user messaging platforms every user shared one store —
a cross-user memory bleed. ``get_memory_dir`` now resolves the bucket from
the session identity (user_id / chat_type / chat_id). These tests pin that
resolution and the filename-safe slug that protects against path traversal.
"""

import hashlib
import re

import pytest

from tools.memory_tool import MemoryStore, _user_slug, get_memory_dir


# =========================================================================
# get_memory_dir resolution
# =========================================================================

class TestGetMemoryDir:
    def test_no_identity_is_root_layout(self, tmp_path, monkeypatch):
        """CLI / cron / bare scripts keep the historical root ``memories/`` layout."""
        monkeypatch.setattr("tools.memory_tool.get_hermes_home", lambda: tmp_path)
        assert get_memory_dir() == tmp_path / "memories"

    def test_dm_isolated_per_user(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.memory_tool.get_hermes_home", lambda: tmp_path)
        alice = get_memory_dir(user_id="alice", chat_type="dm", chat_id="x")
        bob = get_memory_dir(user_id="bob", chat_type="dm", chat_id="y")
        assert alice == tmp_path / "memories" / "alice"
        assert bob == tmp_path / "memories" / "bob"
        assert alice != bob

    def test_group_shared_by_chat_id(self, tmp_path, monkeypatch):
        """A group session buckets under groups/<chat_id>/ shared by all members."""
        monkeypatch.setattr("tools.memory_tool.get_hermes_home", lambda: tmp_path)
        # Two different users in the SAME group resolve to the SAME directory.
        from_alice = get_memory_dir(user_id="alice", chat_type="group", chat_id="grp1")
        from_bob = get_memory_dir(user_id="bob", chat_type="group", chat_id="grp1")
        assert from_alice == tmp_path / "memories" / "groups" / "grp1"
        assert from_alice == from_bob

    def test_different_groups_isolated(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.memory_tool.get_hermes_home", lambda: tmp_path)
        assert get_memory_dir(chat_type="group", chat_id="grp1") != get_memory_dir(
            chat_type="group", chat_id="grp2"
        )

    def test_group_without_chat_id_falls_back_to_user(self, tmp_path, monkeypatch):
        """A group chat with no chat_id degrades to per-user bucketing, not root."""
        monkeypatch.setattr("tools.memory_tool.get_hermes_home", lambda: tmp_path)
        got = get_memory_dir(user_id="alice", chat_type="group", chat_id=None)
        assert got == tmp_path / "memories" / "alice"

    def test_group_chat_type_is_case_insensitive(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.memory_tool.get_hermes_home", lambda: tmp_path)
        assert get_memory_dir(chat_type="Group", chat_id="grp1") == get_memory_dir(
            chat_type="group", chat_id="grp1"
        )


# =========================================================================
# Path-traversal hardening
# =========================================================================

class TestUserSlug:
    def test_empty_returns_empty(self):
        assert _user_slug(None) == ""
        assert _user_slug("") == ""

    def test_traversal_collapsed(self):
        # ``..`` and ``/`` cannot survive the slug — they collapse to ``_``.
        slug = _user_slug("../../etc/passwd")
        assert "/" not in slug
        assert ".." not in slug
        assert re.fullmatch(r"[A-Za-z0-9_.-]+", slug)

    def test_traversal_does_not_escape_root(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.memory_tool.get_hermes_home", lambda: tmp_path)
        got = get_memory_dir(user_id="../../etc/passwd")
        # Result must stay under the memories root.
        assert tmp_path / "memories" in got.parents
        assert str(got).startswith(str(tmp_path / "memories"))

    def test_long_user_id_truncated(self):
        slug = _user_slug("a" * 5000)
        assert len(slug) <= 128

    def test_empty_after_strip_uses_sha_fallback(self):
        # A value that strips to empty (only dots/underscores) falls back to a
        # stable sha prefix rather than the empty string (which would resolve
        # to the shared root and silently re-bleed users).
        slug = _user_slug("....")
        assert slug.startswith("u_")
        assert slug == "u_" + hashlib.sha256(b"....").hexdigest()[:16]


# =========================================================================
# MemoryStore plumbing — chat_type/chat_id reach the store
# =========================================================================

class TestMemoryStoreBucketsByIdentity:
    def test_store_group_uses_group_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.memory_tool.get_hermes_home", lambda: tmp_path)
        store = MemoryStore(chat_type="group", chat_id="grp1")
        # _path_for resolves through get_memory_dir with the store's identity.
        assert store._path_for("memory") == tmp_path / "memories" / "groups" / "grp1" / "MEMORY.md"
        assert store._path_for("user") == tmp_path / "memories" / "groups" / "grp1" / "USER.md"

    def test_store_dm_uses_user_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("tools.memory_tool.get_hermes_home", lambda: tmp_path)
        store = MemoryStore(user_id="alice")
        assert store._path_for("memory") == tmp_path / "memories" / "alice" / "MEMORY.md"

    def test_group_and_dm_do_not_collide(self, tmp_path, monkeypatch):
        """A user's DM memory and a group they're in must not share a file."""
        monkeypatch.setattr("tools.memory_tool.get_hermes_home", lambda: tmp_path)
        dm_store = MemoryStore(user_id="alice", chat_type="dm")
        grp_store = MemoryStore(user_id="alice", chat_type="group", chat_id="grp1")
        assert dm_store._path_for("memory") != grp_store._path_for("memory")


# =========================================================================
# Cross-user isolation end-to-end (the bug this fixes)
# =========================================================================

class TestCrossUserIsolation:
    def test_one_user_cannot_read_another(self, tmp_path, monkeypatch):
        """The bleed this fixes: alice writes, bob's fresh store must not see it."""
        monkeypatch.setattr("tools.memory_tool.get_hermes_home", lambda: tmp_path)

        alice = MemoryStore(memory_char_limit=500, user_char_limit=300, user_id="alice")
        alice.load_from_disk()
        alice.add("memory", "Alice's secret preference")
        alice.save_to_disk("memory")

        bob = MemoryStore(memory_char_limit=500, user_char_limit=300, user_id="bob")
        bob.load_from_disk()
        # Bob's frozen snapshot must NOT contain Alice's entry.
        assert "Alice's secret preference" not in bob._system_prompt_snapshot["memory"]
        assert "Alice's secret preference" not in "\n".join(bob._entries_for("memory"))

    def test_group_members_share(self, tmp_path, monkeypatch):
        """Conversely, two users in the same group session DO share memory."""
        monkeypatch.setattr("tools.memory_tool.get_hermes_home", lambda: tmp_path)

        alice = MemoryStore(
            memory_char_limit=500, user_char_limit=300,
            user_id="alice", chat_type="group", chat_id="grp1",
        )
        alice.load_from_disk()
        alice.add("memory", "team decision")
        alice.save_to_disk("memory")

        bob = MemoryStore(
            memory_char_limit=500, user_char_limit=300,
            user_id="bob", chat_type="group", chat_id="grp1",
        )
        bob.load_from_disk()
        assert "team decision" in "\n".join(bob._entries_for("memory"))
