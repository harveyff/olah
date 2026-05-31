import io
import json

import pytest

from olah.cache.bitset import Bitset
from olah import errors
from olah.mirror.meta import RepoMeta
from olah.proxy import files as proxy_files
from olah.proxy.files import RemoteInfo

pytest.importorskip("portalocker")

from olah.cache.olah_cache import CURRENT_OLAH_CACHE_VERSION, MAX_BLOCK_NUM, OlahCache, OlahCacheHeader


def test_bitset_can_set_clear_and_validate_bounds():
    bitset = Bitset(10)

    bitset.set(1)
    bitset.set(9)
    bitset.clear(1)

    assert bitset.test(1) is False
    assert bitset.test(9) is True
    assert str(bitset).startswith("0000000001")

    with pytest.raises(IndexError):
        bitset.set(10)


def test_olah_cache_header_round_trips_through_binary_stream():
    header = OlahCacheHeader(
        version=CURRENT_OLAH_CACHE_VERSION,
        block_size=1024,
        file_size=2049,
        compression_algo=2,
    )
    stream = io.BytesIO()

    header.write(stream)
    stream.seek(0)
    restored = OlahCacheHeader.read(stream)

    assert restored.version == CURRENT_OLAH_CACHE_VERSION
    assert restored.block_size == 1024
    assert restored.file_size == 2049
    assert restored.block_number == 3
    assert restored.compression_algo == 2


def test_olah_cache_header_rejects_invalid_magic_and_oversized_files():
    with pytest.raises(Exception, match="not a Olah cache file"):
        OlahCacheHeader.read(io.BytesIO(b"BAD!"))

    oversized = OlahCacheHeader(
        version=CURRENT_OLAH_CACHE_VERSION,
        block_size=1,
        file_size=MAX_BLOCK_NUM + 1,
    )
    with pytest.raises(Exception, match="out of the max capability"):
        oversized._valid_header()


@pytest.mark.asyncio
async def test_olah_cache_ignores_zero_length_block_placeholders(tmp_path):
    cache = OlahCache.create(str(tmp_path / "cache"))
    cache.resize(16)

    empty_block = tmp_path / "cache" / "blocks" / "block_00000000.bin"
    empty_block.write_bytes(b"")

    assert cache.has_block(0) is False

    payload = b"abcd" + b"\x00" * (cache._get_block_size() - 4)
    await cache.write_block(0, payload)

    assert cache.has_block(0) is True
    assert empty_block.stat().st_size > 0
    cache.close()


def test_error_responses_return_expected_status_and_headers():
    repo_missing = errors.error_repo_not_found()
    revision_missing = errors.error_revision_not_found("abc123")
    proxy_timeout = errors.error_proxy_timeout()

    assert repo_missing.status_code == 401
    assert repo_missing.headers["x-error-code"] == "RepoNotFound"
    assert json.loads(repo_missing.body) == {"error": "Repository not found"}
    assert revision_missing.status_code == 404
    assert json.loads(revision_missing.body) == {"error": "Invalid rev id: abc123"}
    assert proxy_timeout.status_code == 504
    assert proxy_timeout.headers["x-error-message"] == "Proxy Timeout"


@pytest.mark.asyncio
async def test_olah_cache_invalidates_wrong_decompressed_length(tmp_path):
    cache = OlahCache.create(str(tmp_path / "cache"))
    cache.resize(16)

    payload = b"abcd" + b"\x00" * (cache._get_block_size() - 4)
    await cache.write_block(0, payload)

    block_path = tmp_path / "cache" / "blocks" / "block_00000000.bin"
    assert block_path.exists()

    # Force a truncated gzip payload that still has a valid header.
    import gzip

    block_path.write_bytes(gzip.compress(b"ab"))

    result = await cache.read_block(0)
    assert result is None
    assert cache.has_block(0) is False
    cache.close()


@pytest.mark.asyncio
async def test_olah_cache_invalidates_corrupt_gzip_block(tmp_path):
    cache = OlahCache.create(str(tmp_path / "cache"))
    cache.resize(16)

    block_path = tmp_path / "cache" / "blocks" / "block_00000000.bin"
    block_path.write_bytes(b"not-valid-gzip")

    assert cache.has_block(0) is True
    result = await cache.read_block(0)
    assert result is None
    assert cache.has_block(0) is False
    cache.close()


@pytest.mark.asyncio
async def test_olah_cache_write_block_skips_existing(tmp_path):
    cache = OlahCache.create(str(tmp_path / "cache"))
    cache.resize(16)

    first = b"abcd" + b"\x00" * (cache._get_block_size() - 4)
    second = b"efgh" + b"\x00" * (cache._get_block_size() - 4)
    await cache.write_block(0, first)
    await cache.write_block(0, second)

    restored = await cache.read_block(0)
    assert restored[:4] == b"abcd"
    cache.close()


@pytest.mark.asyncio
async def test_get_file_range_from_cache_refetches_invalidated_block():
    class FakeCache:
        def _get_block_size(self):
            return 4

        def _get_file_size(self):
            return 8

        def has_block(self, idx):
            return idx in {0, 1}

        async def read_block(self, idx):
            if idx == 1:
                return None
            return [b"ABCD", b"EFGH"][idx]

    remote_payload = b"WXYZ"

    class FakeResponse:
        status_code = 206
        headers = {"content-length": str(len(remote_payload))}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def aiter_raw(self):
            yield remote_payload

    class FakeClient:
        def stream(self, **kwargs):
            return FakeResponse()

    class FakeCacheSize:
        def _get_file_size(self):
            return len(remote_payload)

    remote_info = RemoteInfo(
        method="GET",
        url="https://huggingface.co/file.bin",
        headers={},
    )

    chunks = [
        chunk
        async for chunk in proxy_files._get_file_range_from_cache(
            FakeCache(),
            start_pos=0,
            end_pos=8,
            client=FakeClient(),
            remote_info=remote_info,
        )
    ]

    assert chunks == [b"ABCD", remote_payload]


def test_repo_meta_to_dict_exposes_current_field_values():
    meta = RepoMeta()
    meta._id = "internal-id"
    meta.id = "team/demo"
    meta.author = "team"
    meta.tags = ["featured"]
    meta.likes = 7

    assert meta.to_dict()["id"] == "team/demo"
    assert meta.to_dict()["_id"] == "internal-id"
    assert meta.to_dict()["tags"] == ["featured"]
    assert meta.to_dict()["likes"] == 7
