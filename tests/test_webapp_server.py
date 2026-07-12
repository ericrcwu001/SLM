import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
import time

from fastapi.testclient import TestClient
from PIL import Image

from webapp.server import app


IMAGE_PATH = Path("webapp/assets/references/city.jpg")


def _post(client: TestClient, prompt: str):
    with IMAGE_PATH.open("rb") as image:
        return client.post(
            "/api/generate",
            files={"image": ("sample.jpg", image, "image/jpeg")},
            data={"prompt": prompt},
        )


def test_stub_api_contract_and_artifact_serving():
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["ok"] is True and health.json()["stub"] is True
        glossary = client.get("/api/terms").json()
        assert len(glossary) == 54
        assert all({"term", "axis", "category", "definition", "example_usage", "grounded"} <= set(item) for item in glossary)

        grade = _post(client, "make it warmer with strong teal-orange contrast")
        assert grade.status_code == 200
        payload = grade.json()
        assert payload["route"] == "grade" and len(payload["previews"]) == 7
        assert payload["previews"][0]["name"] == "user_image"
        cube = client.get(payload["lut"]["cube_url"])
        assert cube.status_code == 200
        lines = cube.text.splitlines()
        assert "LUT_3D_SIZE 17" in lines and sum(line[:1].isdigit() for line in lines) == 4913
        for preview in payload["previews"]:
            assert client.get(preview["original_url"]).status_code == 200
            assert client.get(preview["graded_url"]).status_code == 200

        clarify = _post(client, "make it pop").json()
        assert clarify["route"] == "clarify" and clarify["lut"] is None and clarify["previews"] == []
        assert clarify["clarify_message"] and clarify["prompt_feedback"]["suggested_terms"]
        refuse = _post(client, "remove the person").json()
        assert refuse["route"] == "refuse" and refuse["refuse_reason"] == "out_of_scope"
        assert refuse["lut"] is None and refuse["previews"] == []


def test_bad_image_has_structured_error_and_server_recovers():
    with TestClient(app) as client:
        response = client.post(
            "/api/generate",
            files={"image": ("broken.png", b"this is not an image", "image/png")},
            data={"prompt": "make it warmer"},
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "bad_image"
        assert client.get("/api/health").json()["ok"] is True


def test_overlapping_requests_are_serialized(monkeypatch):
    active = 0
    max_active = 0
    guard = threading.Lock()

    with TestClient(app) as client:
        pipeline = client.app.extra.get("pipeline") if hasattr(client.app, "extra") else None
        from webapp import server

        original = server.STATE.pipeline.run

        def observed(*args, **kwargs):
            nonlocal active, max_active
            with guard:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.08)
            try:
                return original(*args, **kwargs)
            finally:
                with guard:
                    active -= 1

        monkeypatch.setattr(server.STATE.pipeline, "run", observed)
        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(pool.map(lambda _: _post(client, "make it warmer and strong"), range(2)))
        assert all(response.status_code == 200 for response in responses)
        assert max_active == 1


def test_timeout_keeps_inference_serialized_until_worker_exits(monkeypatch):
    active = 0
    max_active = 0
    guard = threading.Lock()

    with TestClient(app):
        from webapp import server

        original = server.STATE.pipeline.run

        def slow(*args, **kwargs):
            nonlocal active, max_active
            with guard:
                active += 1
                max_active = max(max_active, active)
            try:
                time.sleep(0.12)
                return original(*args, **kwargs)
            finally:
                with guard:
                    active -= 1

        monkeypatch.setattr(server.STATE.pipeline, "run", slow)

        async def exercise_timeout():
            # Keep this probe on one event loop, matching the one-worker uvicorn deployment.
            monkeypatch.setattr(server, "INFERENCE_LOCK", asyncio.Lock())
            image = Image.open(IMAGE_PATH).convert("RGB")

            async def invoke(label: str):
                try:
                    await server._run_pipeline("make it warmer and strong", image, Path("webapp/_runs") / label, 0.02)
                except asyncio.TimeoutError:
                    return "timeout"
                return "completed"

            first = asyncio.create_task(invoke("timeout_first"))
            await asyncio.sleep(0.04)  # response timed out, worker and lock must still be active
            assert server.INFERENCE_LOCK.locked()
            second = asyncio.create_task(invoke("timeout_second"))
            results = await asyncio.gather(first, second)
            while server.INFERENCE_LOCK.locked():
                await asyncio.sleep(0.01)
            return results

        assert asyncio.run(exercise_timeout()) == ["timeout", "timeout"]
        assert max_active == 1
