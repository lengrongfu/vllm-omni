import asyncio
import os
from typing import Any

import diskcache

from vllm_omni.entrypoints.openai.protocol.videos import VideoResponse


class TaskRegistry:
    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> asyncio.Task[None] | None:
        async with self._lock:
            return self._tasks.get(key)

    async def pop(self, key: str) -> asyncio.Task[None] | None:
        async with self._lock:
            return self._tasks.pop(key, None)

    async def upsert(self, key: str, task: asyncio.Task[None]) -> None:
        def _cleanup(_: asyncio.Task[None]) -> None:
            asyncio.create_task(self.pop(key))

        task.add_done_callback(_cleanup)

        async with self._lock:
            self._tasks[key] = task


class VideoStore:
    """Persistent async-safe store for VideoResponse objects backed by diskcache."""

    def __init__(self, directory: str) -> None:
        self._cache = diskcache.Cache(directory)

    async def upsert(self, key: str, value: VideoResponse) -> None:
        await asyncio.to_thread(self._cache.set, key, value.model_dump_json())

    async def get(self, key: str) -> VideoResponse | None:
        raw = await asyncio.to_thread(self._cache.get, key)
        return VideoResponse.model_validate_json(raw) if raw is not None else None

    async def pop(self, key: str) -> VideoResponse | None:
        def _pop() -> VideoResponse | None:
            with self._cache.transact():
                raw = self._cache.get(key)
                if raw is None:
                    return None
                del self._cache[key]
                return VideoResponse.model_validate_json(raw)

        return await asyncio.to_thread(_pop)

    async def update_fields(self, key: str, updates: dict[str, Any]) -> VideoResponse | None:
        def _update() -> VideoResponse | None:
            with self._cache.transact():
                raw = self._cache.get(key)
                if raw is None:
                    return None
                new = VideoResponse.model_validate_json(raw).model_copy(update=updates)
                self._cache.set(key, new.model_dump_json())
                return new

        return await asyncio.to_thread(_update)

    async def list_values(self) -> list[VideoResponse]:
        def _list() -> list[VideoResponse]:
            return [VideoResponse.model_validate_json(self._cache[k]) for k in self._cache]

        return await asyncio.to_thread(_list)

    def close(self) -> None:
        self._cache.close()


metadata_path = os.path.join(os.getenv("VLLM_OMNI_STORAGE_PATH", "/tmp/storage"), "metadata")
VIDEO_STORE = VideoStore(directory=metadata_path)
VIDEO_TASKS: TaskRegistry = TaskRegistry()
