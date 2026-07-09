"""MMArt-PPR10K connector carries the authored user_want_* instruction through acquisition."""

from pathlib import Path

from data_pipeline.acquire.base import AcquireLimits
from data_pipeline.acquire.ppr10k_hf import PPR10KHFConnector


def test_ppr10k_carries_authored_instruction(tmp_path):
    sample = "data/0001"
    contents = {
        f"{sample}/before.jpg": b"\xff\xd8before",
        f"{sample}/processed.jpg": b"\xff\xd8processed",
        f"{sample}/config.xmp": b"<x:xmpmeta/>",
        f"{sample}/user_want_short.txt": "Warm it up and lift the shadows.",
        f"{sample}/user_want_long.txt": "Give the photo a cozy, warm, slightly faded editorial look.",
    }

    def fake_download(fn, root):
        p = Path(root) / fn.replace("/", "_")
        p.parent.mkdir(parents=True, exist_ok=True)
        data = contents[fn]
        p.write_bytes(data) if isinstance(data, bytes) else p.write_text(data, encoding="utf-8")
        return str(p)

    conn = PPR10KHFConnector(list_files_fn=lambda: list(contents), download_fn=fake_download)
    report = conn.acquire(tmp_path, AcquireLimits(max_items=10))

    assert report.acquired == 1
    art = report.artifacts[0]
    assert art.derivation_method == "pair_fit"
    assert art.authored_instruction == "Warm it up and lift the shadows."
    assert art.authored_instruction_natural == "Give the photo a cozy, warm, slightly faded editorial look."
    assert art.authored_instruction_source and "mmart_ppr10k" in art.authored_instruction_source
    # and it survives onto the provenance/registry row (which feeds Stage-9 SftRow)
    rr = art.to_registry_row()
    assert rr.authored_instruction == "Warm it up and lift the shadows."
    assert rr.authored_instruction_natural.startswith("Give the photo a cozy")


def test_ppr10k_without_instructions_still_acquires_pair(tmp_path):
    """No user_want_* files -> pair still acquired, authored fields None (teacher fills it later)."""
    sample = "data/0002"
    contents = {
        f"{sample}/before.jpg": b"\xff\xd8b",
        f"{sample}/processed.jpg": b"\xff\xd8p",
    }

    def fake_download(fn, root):
        p = Path(root) / fn.replace("/", "_")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(contents[fn])
        return str(p)

    conn = PPR10KHFConnector(list_files_fn=lambda: list(contents), download_fn=fake_download)
    report = conn.acquire(tmp_path, AcquireLimits(max_items=10))
    assert report.acquired == 1
    art = report.artifacts[0]
    assert art.authored_instruction is None
    assert art.authored_instruction_source is None
