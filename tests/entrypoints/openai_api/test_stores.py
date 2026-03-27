# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
from pytest_mock import MockerFixture
pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_store(tmp_path):
    from vllm_omni.entrypoints.openai.stores import VideoStore

    return VideoStore(directory=str(tmp_path / "video_store"))


def _make_video(video_id: str = "video_gen_abc123", model: str = "test-model"):
    from vllm_omni.entrypoints.openai.protocol.videos import VideoResponse

    return VideoResponse(id=video_id, model=model, prompt="a cat running")


# ---------------------------------------------------------------------------
# VideoStore
# ---------------------------------------------------------------------------



@pytest.mark.asyncio
async def test_upsert_and_get(tmp_path,mocker: MockerFixture) -> None:
    store = _make_store(tmp_path)
    video = _make_video()

    await store.upsert(video.id, video)
    result = await store.get(video.id)

    assert result is not None
    assert result.id == video.id
    assert result.model == video.model
    assert result.prompt == video.prompt




@pytest.mark.asyncio
async def test_get_missing_key_returns_none(tmp_path,mocker: MockerFixture) -> None:
    store = _make_store(tmp_path)
    result = await store.get("nonexistent")
    assert result is None




@pytest.mark.asyncio
async def test_pop_removes_and_returns(tmp_path,mocker: MockerFixture):
    store = _make_store(tmp_path)
    video = _make_video()

    await store.upsert(video.id, video)
    popped = await store.pop(video.id)
    assert popped is not None
    assert popped.id == video.id

    # should be gone now
    assert await store.get(video.id) is None




@pytest.mark.asyncio
async def test_pop_missing_key_returns_none(tmp_path):
    store = _make_store(tmp_path)
    assert await store.pop("nonexistent") is None




@pytest.mark.asyncio
async def test_update_fields(tmp_path):
    from vllm_omni.entrypoints.openai.protocol.videos import VideoGenerationStatus

    store = _make_store(tmp_path)
    video = _make_video()
    await store.upsert(video.id, video)

    updated = await store.update_fields(video.id, {"status": VideoGenerationStatus.IN_PROGRESS, "progress": 50})
    assert updated is not None
    assert updated.status == VideoGenerationStatus.IN_PROGRESS
    assert updated.progress == 50

    # persisted
    fetched = await store.get(video.id)
    assert fetched.status == VideoGenerationStatus.IN_PROGRESS
    assert fetched.progress == 50




@pytest.mark.asyncio
async def test_update_fields_missing_key_returns_none(tmp_path):
    store = _make_store(tmp_path)
    result = await store.update_fields("nonexistent", {"progress": 10})
    assert result is None




@pytest.mark.asyncio
async def test_list_values(tmp_path):
    store = _make_store(tmp_path)
    v1 = _make_video("id_1")
    v2 = _make_video("id_2")

    await store.upsert(v1.id, v1)
    await store.upsert(v2.id, v2)

    values = await store.list_values()
    ids = {v.id for v in values}
    assert ids == {"id_1", "id_2"}




@pytest.mark.asyncio
async def test_upsert_overwrites(tmp_path):
    store = _make_store(tmp_path)
    video = _make_video()
    await store.upsert(video.id, video)

    updated = video.model_copy(update={"prompt": "a dog running"})
    await store.upsert(video.id, updated)

    result = await store.get(video.id)
    assert result.prompt == "a dog running"




@pytest.mark.asyncio
async def test_persistence_across_instances(tmp_path):
    """Data written by one VideoStore instance is readable by another on the same directory."""
    directory = str(tmp_path / "video_store")
    from vllm_omni.entrypoints.openai.stores import VideoStore

    video = _make_video()

    store1 = VideoStore(directory=directory)
    await store1.upsert(video.id, video)
    store1.close()

    store2 = VideoStore(directory=directory)
    result = await store2.get(video.id)
    store2.close()

    assert result is not None
    assert result.id == video.id