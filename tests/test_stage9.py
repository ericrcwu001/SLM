"""Tests for Stage 9: gated interfaces, embeddings, selection, active/eval sets."""

import numpy as np
import pytest

from data_pipeline.active_dataset import AcceptanceChecker, SftRow, assemble_active
from data_pipeline.behavior_vector import measure_behavior
from data_pipeline.embeddings import build_pca, lut_behavior_embedding, tag_embedding
from data_pipeline.errors import RequiresTeacher, RequiresTokenizer
from data_pipeline.eval_sets import EvalCandidate, build_eval_sets
from data_pipeline.instruction_gen import TeacherClient, validate_tags_against_behavior
from data_pipeline.selection import SelectionCandidate, mmr_select, select_active
from data_pipeline.sources import procedural as proc
from data_pipeline.tokenize_targets import encode_residual_to_codes, is_available
from eval.cube_io import absolute_to_residual, identity_grid


# --- tokenizer interface (wired at Stage 8) ---
def test_tokenizer_enabled_encode():
    # The encoder is wired (delegates to the frozen VQVAE). Exercising it needs the frozen
    # weights, which are gitignored and only present after freeze/staging — skip cleanly when
    # they are absent (fresh clone / CI). When present, encode must yield 64 valid code ids.
    assert is_available() is True
    pytest.importorskip("torch")
    from tokenizer.frozen import frozen_final_dir, load_frozen_vqvae

    if not (frozen_final_dir() / "model.pt").is_file():
        pytest.skip("frozen tokenizer weights not staged (run tokenizer.freeze / slm_stage first)")
    codes = encode_residual_to_codes(np.zeros((17, 17, 17, 3), dtype=np.float64))
    assert len(codes) == 64
    assert all(isinstance(c, int) and 0 <= c < 256 for c in codes)
    # load_frozen_vqvae asserts loaded-weight hashes == manifest; confirm identity too.
    _model, manifest = load_frozen_vqvae()
    assert manifest["arch_version"] == "vq_v2_srgbres_17to4_cb256_t64"


def test_tokenizer_missing_weights_raises_requires_tokenizer(tmp_path, monkeypatch):
    # With the encoder enabled but no staged weights, callers must still see RequiresTokenizer
    # (degrade to pending), never a fabricated result.
    pytest.importorskip("torch")
    import tokenizer.frozen as frozen

    frozen.load_frozen_vqvae.cache_clear()
    monkeypatch.setenv("SLM_ARTIFACT_ROOT", str(tmp_path))  # empty -> no tokenizer/final/model.pt
    monkeypatch.chdir(tmp_path)                             # repo-relative fallback also empty
    with pytest.raises(RequiresTokenizer):
        encode_residual_to_codes(np.zeros((17, 17, 17, 3), dtype=np.float64))
    frozen.load_frozen_vqvae.cache_clear()


def test_teacher_gated_without_config(tmp_path):
    tc = TeacherClient(model_clients_path=tmp_path / "missing.yaml")
    assert tc.is_available() is False
    with pytest.raises(RequiresTeacher):
        tc.generate({"id": "x"})


def test_teacher_rejects_alias_model_id(tmp_path):
    cfg = tmp_path / "mc.yaml"
    cfg.write_text(
        "teacher_primary:\n  provider: openai\n  model_id: latest\n"
        "  endpoint_env: E\n  api_key_env: K\n  prompt_version: v1\n  batch_id: b1\n"
    )
    assert TeacherClient(model_clients_path=cfg).is_available() is False


def test_teacher_available_when_pinned(tmp_path):
    cfg = tmp_path / "mc.yaml"
    cfg.write_text(
        "teacher_primary:\n  provider: openai\n  model_id: gpt-x-2026-01\n"
        "  endpoint_env: SLM_TEACHER_BASE_URL\n  api_key_env: SLM_TEACHER_API_KEY\n"
        "  prompt_version: teacher_prompt_v1\n  batch_id: batch_001\n"
    )
    assert TeacherClient(model_clients_path=cfg).is_available() is True


def test_tag_behavior_validation():
    warm = measure_behavior(proc.generate_lut_tensor(
        next(s for s in proc.catalog() if s.lut_id == "proc_attr_warmer")))
    ok, issues = validate_tags_against_behavior(["warmer"], warm)
    assert ok, issues
    bad, bad_issues = validate_tags_against_behavior(["cooler"], warm)  # wrong direction
    assert not bad and any("tag_not_backed" in i for i in bad_issues)


# --- embeddings + selection ---
def test_tag_embedding_multihot():
    e = tag_embedding(["warmer", "matte"])
    assert e.sum() == 2.0


def test_lut_behavior_embedding_shape():
    luts = [proc.generate_lut_tensor(s) for s in proc.catalog()[:6]]
    residuals = [absolute_to_residual(l).reshape(-1) for l in luts]
    pca = build_pca(residuals, dim=5)
    emb = lut_behavior_embedding(residuals[0], measure_behavior(luts[0]), pca=pca)
    assert emb.shape[0] == 5 + 9  # pca dim + behavior keys


def test_mmr_selects_diverse_and_deterministic():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(30, 8))
    a = mmr_select(emb, 5, seed=0)
    b = mmr_select(emb, 5, seed=0)
    assert a == b and len(set(a)) == 5


def test_selection_enforces_family_caps():
    cands = []
    for i in range(80):
        cands.append(SelectionCandidate(id=f"p{i}", family="controlled_procedural",
                                        embedding=np.random.default_rng(i).normal(size=6), procedural=True))
    for i in range(40):
        cands.append(SelectionCandidate(id=f"f{i}", family="fresh_luts",
                                        embedding=np.random.default_rng(100 + i).normal(size=6)))
    rep = select_active(cands, target_size=100, seed=1234)
    proc_sel = rep.per_family.get("controlled_procedural", 0)
    assert proc_sel <= int(0.10 * 100)  # procedural cap 10%


# --- active dataset + acceptance ---
_FAMILIES = ["fresh_luts", "gmic_rawtherapee"]


def _candidate(i, family=None, tags=None, procedural=False):
    family = family or _FAMILIES[i % len(_FAMILIES)]
    spec = proc.catalog()[i % len(proc.catalog())]
    lut = proc.generate_lut_tensor(spec)
    return {
        "id": f"c{i}", "source_family": family, "source_lut_id": f"lut{i}",
        "gold_tags": tags if tags is not None else list(spec.gold_tags),
        "measured_behavior": measure_behavior(lut),
        "derived_lut_quality": {"representability_tier": "gold", "fit_deltaE00_mean": 0.0},
        "representability_tier": "gold", "split_unit_id": f"u{i}", "split": "train",
        "procedural_filler": procedural,
    }


def test_assemble_active_gated_fields():
    rows = assemble_active([_candidate(0), _candidate(1, procedural=True)])
    assert all(r.target_tokens is None and r.token_status == "pending_tokenizer" for r in rows)
    assert all(r.instruction is None and r.instruction_status == "pending_teacher" for r in rows)
    assert rows[1].procedural_filler and rows[1].headline_eligible is False


def test_acceptance_pending_for_gated_criteria():
    rows = assemble_active([_candidate(i) for i in range(20)])
    res = AcceptanceChecker(enforce_scale=False).check(rows, leakage_status="pass")
    assert res.criteria["scale"]["status"] == "pass"
    assert res.criteria["no_leakage"]["status"] == "pass"
    assert res.criteria["canonical_domain"]["status"] == "pass"
    # gated criteria are honestly pending, not fabricated pass
    assert res.criteria["representability_and_recon"]["status"] == "pending"
    assert res.criteria["unsupported_coverage"]["status"] == "pending"
    assert res.criteria["manifests_present"]["status"] == "pending"
    assert res.overall == "pending"


def test_acceptance_with_unsupported_corpus():
    # supported rows + a covering unsupported/refusal corpus (>=8 categories incl mixed).
    rows = assemble_active([_candidate(i) for i in range(10)])
    cats = ["local_region_edit", "semantic_object_recolor", "content_removal", "content_generation",
            "relighting", "texture_detail", "geometry", "reference_style_transfer"]
    for j, c in enumerate(cats):
        rows.append(SftRow(id=f"u{j}", is_supported=False, support_label="unsupported",
                           unsupported_category=c, split="train"))
    rows.append(SftRow(id="um", is_supported=False, support_label="unsupported",
                       unsupported_category="mixed_partial_supported_plus_local_edit",
                       mixed_prompt=True, split="train"))
    res = AcceptanceChecker(enforce_scale=False).check(rows, leakage_status="pass")
    # unsupported rows must NOT trip the measured-behavior check (they have no LUT)
    assert res.criteria["provenance_and_behavior"]["status"] == "pass"
    # coverage now satisfied
    assert res.criteria["unsupported_coverage"]["status"] == "pass"


def test_eval_sets_procedural_is_diagnostic_only():
    cands = [EvalCandidate(id=f"e{i}", split="diagnostic", procedural_filler=True) for i in range(5)]
    m = build_eval_sets(cands)
    assert m.diagnostic_only is True
    assert m.headline_eligible_count == 0
    assert len(m.slices["diagnostic"]) == 5


def test_eval_sets_headline_from_gold_nonprocedural():
    cands = [EvalCandidate(id=f"h{i}", split="eval", representability_tier="gold") for i in range(3)]
    m = build_eval_sets(cands)
    assert m.headline_eligible_count == 3
    assert not m.diagnostic_only


def test_eval_sets_headline_admits_well_fit_diagnostic():
    # #2: headline eligibility is a fidelity bar, not the tier label. A diagnostic row with a
    # faithful global fit (<= HEADLINE_FIT_MAX) is headline-eligible; a poorly-fit one is not.
    good = EvalCandidate(id="d_good", split="eval", representability_tier="diagnostic_only",
                         fit_deltaE00_mean=1.8)
    poor = EvalCandidate(id="d_poor", split="eval", representability_tier="diagnostic_only",
                         fit_deltaE00_mean=2.9)
    m = build_eval_sets([good, poor])
    assert "d_good" in m.slices["usage_weighted_headline_supported"]
    assert "d_poor" in m.slices["diagnostic"]
    assert m.headline_eligible_count == 1
