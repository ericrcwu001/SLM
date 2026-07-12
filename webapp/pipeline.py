"""Thin prompt -> route -> generated LUT orchestration for the local web demo."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from data_pipeline.attribute_spec import AttributeSpec, canonicalize, serialize, serialize_bucketed
from eval.cube_io import identity_grid
from eval.refuse_taxonomy import (
    REFUSE_OUT_OF_GAMUT,
    REFUSE_OUT_OF_SCOPE,
    ROUTE_CLARIFY,
    ROUTE_GRADE,
    ROUTE_REFUSE,
)
from webapp import terms
from webapp.models_config import WebappConfig, load_generator, load_interpreter, repo_path


@dataclass
class RouteResult:
    route: str
    refuse_reason: str | None
    spec_text: str | None
    clarify_message: str | None
    spec: AttributeSpec | None


@dataclass
class LutResult:
    codes: list[int]
    lut: np.ndarray
    record: dict[str, Any]
    fell_back_greedy: bool


class GeneratorRefused(RuntimeError):
    pass


class PromptToLutPipeline:
    """A load-once model pipeline with a complete, weight-free stub path."""

    def __init__(self, cfg: WebappConfig):
        self.cfg = cfg
        self.device = cfg.device
        self.terms = terms.TermsModule()
        self.vq_final_dir = cfg.vq_decoder.final_dir
        self.references_dir = repo_path(cfg.server.references_dir)
        self._references = self._load_reference_manifest()
        self.interp_model = self.interp_tok = self.interp_device = None
        self.gen_model = self.gen_processor = None
        self.load_error: str | None = None
        if not cfg.generator.stub:
            # A bad model path must surface via /api/health, not crash construction (which would take
            # the whole SPA down); self_check() reports the specific failure and generate() returns 503.
            try:
                self.interp_model, self.interp_tok, self.interp_device = load_interpreter(cfg)
                self.gen_model, self.gen_processor = load_generator(cfg)
            except Exception as exc:
                self.load_error = f"{type(exc).__name__}: {exc}"

    def is_ready(self) -> bool:
        """True when the pipeline can serve a grade (stub always; real needs both models loaded)."""
        if self.cfg.generator.stub:
            return True
        return self.interp_model is not None and self.gen_model is not None

    def _load_reference_manifest(self) -> list[dict[str, Any]]:
        manifest_path = self.references_dir / "references.json"
        if not manifest_path.is_file():
            return []
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = payload.get("references", []) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError("references.json must contain a list or a references list")
        normalized = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            filename = row.get("filename") or row.get("file") or row.get("path")
            name = row.get("name") or row.get("category") or row.get("label")
            if filename and name:
                normalized.append({**row, "filename": str(filename), "name": str(name)})
        return normalized

    def self_check(self) -> dict[str, Any]:
        issues: list[str] = []
        if self.load_error:
            issues.append(f"model load failed: {self.load_error}")
        if len(self._references) != 6:
            issues.append(f"expected 6 reference images, found {len(self._references)}")
        for row in self._references:
            if not (self.references_dir / row["filename"]).is_file():
                issues.append(f"missing reference: {row['filename']}")
        if not self.cfg.generator.stub:
            if self.interp_model is None:
                issues.append("interpreter not loaded")
            if self.gen_model is None:
                issues.append("generator not loaded")
            if self.vq_final_dir:
                final_dir = Path(self.vq_final_dir)
            else:
                from tokenizer.frozen import frozen_final_dir

                final_dir = frozen_final_dir()
            if not (final_dir / "model.pt").is_file():
                issues.append(f"missing frozen decoder: {final_dir / 'model.pt'}")
        ok = not issues
        return {
            "ok": ok,
            "loaded": True,
            "ready": ok,
            "device": self.device,
            "stub": self.cfg.generator.stub,
            "interpreter_ok": self.cfg.generator.stub or self.interp_model is not None,
            "generator_ok": self.cfg.generator.stub or self.gen_model is not None,
            "decoder_ok": self.cfg.generator.stub or not any("decoder" in issue for issue in issues),
            "references": len(self._references),
            "load_error": self.load_error,
            "issues": issues,
            "models": {
                "interpreter": "stub" if self.cfg.generator.stub else self.cfg.interpreter.model_path,
                "generator": "synthetic LUT stub" if self.cfg.generator.stub else self.cfg.generator.adapter_path,
                "vq_decoder": "not required in stub" if self.cfg.generator.stub else str(self.vq_final_dir or "auto"),
            },
        }

    @staticmethod
    def _clarify_message(_prompt: str) -> str:
        return "Which color or tonal direction do you want, and should the change be slight, moderate, strong, or extreme?"

    def _stub_route_and_spec(self, prompt: str) -> RouteResult:
        p = prompt.lower().strip()
        out_of_scope = (
            "remove ", "replace ", "add a ", "add the ", "crop", "retouch", "blur the",
            "sharpen the person", "change the background", "erase ", "move the ",
        )
        if any(phrase in p for phrase in out_of_scope):
            spec = AttributeSpec(route=ROUTE_REFUSE, refuse_reason=REFUSE_OUT_OF_SCOPE)
            return RouteResult(ROUTE_REFUSE, REFUSE_OUT_OF_SCOPE, None, None, spec)
        if any(phrase in p for phrase in ("impossible color", "outside gamut", "out of gamut")):
            spec = AttributeSpec(route=ROUTE_REFUSE, refuse_reason=REFUSE_OUT_OF_GAMUT)
            return RouteResult(ROUTE_REFUSE, REFUSE_OUT_OF_GAMUT, None, None, spec)
        vague = {"make it pop", "make it better", "make it look nicer", "look nicer", "fix it", "improve it", "nice", "better"}
        if not p or p in vague or (len(p.split()) <= 3 and any(word in p for word in ("pop", "nice", "better"))):
            spec = AttributeSpec(route=ROUTE_CLARIFY)
            return RouteResult(ROUTE_CLARIFY, None, None, self._clarify_message(prompt), spec)

        magnitude = 4.0 if re.search(r"\b(strong|strongly|very|intense)\b", p) else 2.3
        axes: dict[str, float] = {}
        if any(word in p for word in ("warm", "amber", "golden", "orange")):
            axes["temperature_delta_b"] = magnitude
        if any(word in p for word in ("cool", "blue", "teal")) and "teal-orange" not in p:
            axes["temperature_delta_b"] = -magnitude
        if "contrast" in p or "punch" in p:
            axes["contrast_l_spread_delta"] = magnitude
        if any(word in p for word in ("saturat", "vibrant", "vivid")):
            axes["chroma_delta"] = magnitude
        if any(word in p for word in ("muted", "desaturat")):
            axes["chroma_delta"] = -magnitude
        if any(word in p for word in ("matte", "faded", "filmic")):
            axes["matte_strength"] = magnitude
        if "teal-orange" in p or "teal and orange" in p:
            axes.update({"split_tone_strength": magnitude, "shadow_hue_deg": 200.0, "highlight_hue_deg": 55.0})
        if not axes:
            axes = {"temperature_delta_b": 2.3, "contrast_l_spread_delta": 2.0}
        spec = canonicalize(AttributeSpec(route=ROUTE_GRADE, axes=axes))
        spec_text = serialize(spec)
        return RouteResult(ROUTE_GRADE, None, spec_text, None, spec)

    def route_and_spec(self, prompt: str) -> RouteResult:
        if self.cfg.generator.stub:
            return self._stub_route_and_spec(prompt)

        import torch
        from interpreter.comparator import _safe_parse
        from interpreter.example import build_prompt_ids

        ids = build_prompt_ids(self.interp_tok, prompt)
        input_ids = torch.tensor([ids]).to(self.interp_device)
        with torch.no_grad():
            output = self.interp_model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=self.cfg.interpreter.max_new_tokens,
                do_sample=False,
                eos_token_id=self.interp_tok.eos_token_id,
                pad_token_id=self.interp_tok.pad_token_id,
            )
        pred_text = self.interp_tok.decode(output[0][len(ids):], skip_special_tokens=True)
        spec = _safe_parse(pred_text)
        if spec is None:
            return RouteResult(ROUTE_CLARIFY, None, None, self._clarify_message(prompt), None)
        spec = canonicalize(spec)
        if spec.route == ROUTE_REFUSE:
            return RouteResult(ROUTE_REFUSE, spec.refuse_reason or REFUSE_OUT_OF_SCOPE, None, None, spec)
        if spec.route == ROUTE_CLARIFY:
            return RouteResult(ROUTE_CLARIFY, None, None, self._clarify_message(prompt), spec)
        return RouteResult(ROUTE_GRADE, None, serialize(spec), None, spec)

    def _cond_text(self, prompt: str, spec: AttributeSpec) -> str:
        spec_text = (serialize_bucketed if self.cfg.generator.spec_bucketize else serialize)(spec)
        if self.cfg.generator.input_mode == "instruction":
            return prompt
        if self.cfg.generator.input_mode == "instruction_and_spec":
            return f"{prompt}\n{spec_text}"
        return spec_text

    @staticmethod
    def _stub_lut() -> np.ndarray:
        """Build a visible warm, cinematic split-tone grade directly in absolute LUT space."""
        lut = identity_grid(17).astype(np.float64)
        luma = lut[..., 0] * 0.2126 + lut[..., 1] * 0.7152 + lut[..., 2] * 0.0722
        contrast = (lut - 0.5) * 1.09 + 0.5
        shadows = np.clip((0.52 - luma) / 0.52, 0.0, 1.0)[..., None]
        highlights = np.clip((luma - 0.45) / 0.55, 0.0, 1.0)[..., None]
        contrast += shadows * np.array([-0.015, 0.012, 0.022])
        contrast += highlights * np.array([0.045, 0.018, -0.030])
        contrast += np.array([0.012, 0.004, -0.012])
        return np.clip(contrast, 0.0, 1.0)

    def generate_lut(self, cond_text: str, spec_text: str, image: Image.Image) -> LutResult:
        if self.cfg.generator.stub:
            return LutResult([], self._stub_lut(), {"behavioral_fidelity": 1.0, "collapsed": False, "stub": True}, False)

        from eval.behavioral_fidelity import decode_codes, score_from_lut
        from eval.best_of_n import best_of_n_codes
        from sft.generate import generate_codes

        g = self.cfg.generator
        best_codes, record = best_of_n_codes(
            self.gen_model,
            self.gen_processor,
            image=image,
            cond_text=cond_text,
            spec_text=spec_text,
            n=g.best_of_n_N,
            sampling=g.sampling,
            chunk=g.chunk,
            device=self.device,
        )
        fell_back = False
        if best_codes is None:
            fell_back = True
            best_codes = generate_codes(self.gen_model, self.gen_processor, image=image, text=cond_text, sampling=None, device=self.device)
            if best_codes is None:
                raise GeneratorRefused(cond_text)
            fallback_lut = decode_codes(best_codes, final_dir=self.vq_final_dir)
            record = score_from_lut(fallback_lut, spec_text, codes=best_codes)
        lut = decode_codes(best_codes, final_dir=self.vq_final_dir)
        return LutResult(list(best_codes), lut, record, fell_back)

    def _reference_images(self):
        for row in self._references:
            path = self.references_dir / row["filename"]
            with Image.open(path) as image:
                yield row["name"], image.convert("RGB")

    @staticmethod
    def _safe_stem(name: str) -> str:
        stem = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        return stem or "reference"

    def _preview(self, pil_image: Image.Image, lut: np.ndarray, run_dir: Path, name: str, run_id: str) -> dict[str, str]:
        from webapp.lut import apply_lut, save_image

        stem = self._safe_stem(name)
        original_name = f"{stem}_original.png"
        graded_name = f"{stem}_graded.png"
        original = np.asarray(pil_image.convert("RGB"), dtype=np.float64) / 255.0
        graded = apply_lut(original, lut)
        save_image(pil_image, run_dir / original_name)
        save_image(np.rint(graded * 255.0).astype(np.uint8), run_dir / graded_name)
        return {
            "name": name,
            "original_url": f"/runs/{run_id}/{original_name}",
            "graded_url": f"/runs/{run_id}/{graded_name}",
        }

    @staticmethod
    def _base_payload(route: str, feedback: dict[str, Any]) -> dict[str, Any]:
        return {
            "route": route,
            "refuse_reason": None,
            "clarify_message": None,
            "attribute_spec_text": None,
            "lut": None,
            "previews": [],
            "prompt_feedback": feedback,
            "quality": None,
        }

    def run(self, prompt: str, image: Image.Image, run_dir: str | Path) -> dict[str, Any]:
        route_result = self.route_and_spec(prompt)
        feedback = terms.suggest_terms(prompt, route_result.spec, route_result.route)
        payload = self._base_payload(route_result.route, feedback)
        if route_result.route == ROUTE_REFUSE:
            payload["refuse_reason"] = route_result.refuse_reason
            return payload
        if route_result.route == ROUTE_CLARIFY:
            payload["clarify_message"] = route_result.clarify_message
            return payload

        assert route_result.spec is not None and route_result.spec_text is not None
        try:
            output = self.generate_lut(self._cond_text(prompt, route_result.spec), route_result.spec_text, image)
        except GeneratorRefused:
            payload["route"] = ROUTE_CLARIFY
            payload["clarify_message"] = "The model could not produce a confident grade. Try naming a direction and intensity."
            payload["attribute_spec_text"] = route_result.spec_text
            return payload

        from webapp.lut import export_cube

        destination = Path(run_dir)
        destination.mkdir(parents=True, exist_ok=True)
        run_id = destination.name
        export_cube(output.lut, destination / "output.cube")
        previews = [self._preview(image, output.lut, destination, "user_image", run_id)]
        previews.extend(self._preview(ref, output.lut, destination, name, run_id) for name, ref in self._reference_images())
        payload.update(
            {
                "attribute_spec_text": route_result.spec_text,
                "lut": {"cube_url": f"/runs/{run_id}/output.cube"},
                "previews": previews,
                "quality": {
                    "behavioral_fidelity": output.record.get("behavioral_fidelity"),
                    "collapsed": output.record.get("collapsed"),
                    "fell_back_greedy": output.fell_back_greedy,
                },
            }
        )
        return payload
