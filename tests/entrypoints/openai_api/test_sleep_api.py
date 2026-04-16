# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from argparse import Namespace
from http import HTTPStatus
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from vllm_omni.entrypoints.openai.api_server import router


class FakeEngine:
    """Full-featured stub that records calls and tracks sleep state."""

    def __init__(self):
        self._is_sleeping = False
        self._sleep_level: int | None = None
        self.sleep_calls: list[tuple] = []  # [(level, mode), ...]
        self.wake_up_calls: list[tuple] = []  # [(tags,), ...]

    async def sleep(self, level: int = 1, mode: str = "abort") -> None:
        self._is_sleeping = True
        self._sleep_level = level
        self.sleep_calls.append((level, mode))

    async def wake_up(self, tags: list[str] | None = None) -> None:
        self._is_sleeping = False
        self._sleep_level = None
        self.wake_up_calls.append((tags, ))

    async def is_sleeping(self) -> bool:
        return self._is_sleeping

    async def sleep_level(self) -> int:
        return self._sleep_level


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.state.engine_client = FakeEngine()
    app.state.args = Namespace(enable_sleep_mode=True)
    return TestClient(app, raise_server_exceptions=False)


def test_sleep(client: TestClient):
    assert client.post("/sleep").status_code == HTTPStatus.OK

    client.post("/sleep")
    engine: FakeEngine = client.app.state.engine_client
    level, model = engine.sleep_calls[-1]
    assert level == 1
    assert model == "abort"

    client.post("/sleep?level=2&mode=drain")
    engine: FakeEngine = client.app.state.engine_client
    level, model = engine.sleep_calls[-1]
    assert level == 2
    assert model == "abort"


def test_wakeup(client: TestClient):
    assert client.post("/wake_up").status_code == HTTPStatus.OK

    client.post("/wake_up")
    engine: FakeEngine = client.app.state.engine_client
    assert not engine._is_sleeping

    client.post("/wake_up?tags=")
    engine: FakeEngine = client.app.state.engine_client
    assert engine.wake_up_calls[-1] == ([''],)

    client.post("/wake_up?tags=weights")
    engine: FakeEngine = client.app.state.engine_client
    assert engine.wake_up_calls[-1] == (["weights"], )


def test_sleep_info(client: TestClient):
    client.post("/sleep?level=1")
    assert client.get("/sleep_info").status_code == HTTPStatus.OK

    client.post("/wake_up")
    assert client.get("/sleep_info").json() == {
        "sleep_level": None,
        "is_sleeping": False
    }

    client.post("/sleep?level=1")
    assert client.get("/sleep_info").json() == {
        "sleep_level": 1,
        "is_sleeping": True
    }

    client.post("/sleep?level=2")
    assert client.get("/sleep_info").json() == {
        "sleep_level": 2,
        "is_sleeping": True
    }