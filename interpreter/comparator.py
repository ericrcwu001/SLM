"""Predicted-spec vs target-spec comparison for interpreter scoring.

No such comparator existed in-repo (``eval.behavioral_fidelity.behavioral_agreement`` compares a spec
to a *measured LUT*, not to another *spec*). We build one on top of the frozen seam, reusing
``parse``/``canonicalize`` and ``behavioral_agreement``'s sign+magnitude backing rule.

Two failure modes the adversarial review flagged are handled explicitly here:
  * ``behavioral_agreement`` returns ``fidelity=None`` whenever a spec asserts no measurable axis
    (extremely common early in training, and legal for a near-identity LUT). We therefore compute
    precision/recall from the raw ``axes_total``/``axes_backed`` counts, never the ``None`` fidelity.
  * ``attribute_spec.parse`` *raises* on malformed text and silently defaults grammar-less gibberish
    to ``route=grade``. We parse defensively: any malformation -> ``parse_ok=False`` and a hard miss
    (route wrong, F1 0), so garbage never scores as a correct grade.
"""

from __future__ import annotations

from typing import Optional

from data_pipeline.attribute_spec import AttributeSpec, canonicalize, parse
from eval.behavioral_fidelity import DEFAULT_TOL, behavioral_agreement
from eval.refuse_taxonomy import ROUTE_GRADE, ROUTE_REFUSE


def spec_as_mb(spec: AttributeSpec) -> dict:
    """Recast a spec's asserted axes as a measured-behavior-shaped dict, so ``behavioral_agreement``
    can back the OTHER spec against it (turning spec-vs-spec into two spec-vs-"measurement" checks)."""
    mb = dict(spec.axes)
    mb["per_hue_saturation"] = dict(spec.sat)
    return mb


def _safe_parse(text: str) -> Optional[AttributeSpec]:
    """Parse the first spec line; return None on ANY malformation (never a silent grade default).

    Two guards: (1) the line must START with ``route=`` — ``parse`` otherwise defaults the route to
    ``grade`` and drops unrecognized tokens, so grammar-less gibberish would masquerade as an empty
    grade spec; (2) ``parse`` itself raises on a bad route/refuse-kind/float, which we swallow to a
    miss. Our training targets always begin with ``route=``, so a well-formed generation passes.
    """
    if not text or not text.strip():
        return None
    first = text.strip().splitlines()[0].strip()
    if not first.startswith("route="):
        return None
    try:
        return parse(first)
    except Exception:  # noqa: BLE001  (parse raises ValueError on bad route/kind/float)
        return None


def _backed_fraction(asserting: AttributeSpec, reference_mb: dict, tol: float) -> float:
    """``axes_backed / axes_total`` of ``asserting`` against ``reference_mb``; 1.0 (vacuous) when
    ``asserting`` claims nothing — no false positives to penalize."""
    res = behavioral_agreement(asserting, reference_mb, tol=tol)
    n = res["axes_total"]
    return 1.0 if n == 0 else res["axes_backed"] / n


def _combined_axes(spec: AttributeSpec) -> dict:
    """Non-hue axes + sat sectors as one signed dict (hue-angle axes have no meaningful sign)."""
    d = {f: v for f, v in spec.axes.items() if not f.endswith("_hue_deg")}
    d.update({f"sat_{s}": v for s, v in spec.sat.items()})
    return d


def _sign_fraction(asserting: AttributeSpec, reference: AttributeSpec) -> float:
    """Fraction of ``asserting``'s axes whose SIGN matches ``reference`` (magnitude ignored); 1.0
    (vacuous) when ``asserting`` claims nothing. This isolates 'did it get the DIRECTION right'
    from magnitude calibration, which the tol-based ``attribute_f1`` conflates."""
    ca, cr = _combined_axes(asserting), _combined_axes(reference)
    if not ca:
        return 1.0
    match = sum(1 for f, v in ca.items() if cr.get(f, 0.0) != 0.0 and (v > 0) == (cr[f] > 0))
    return match / len(ca)


def compare_specs(pred_text: str, gold_text: str, *, tol: float = DEFAULT_TOL) -> dict:
    """Compare a predicted ``attribute_spec_text`` to the gold target.

    Returns route match, refuse-kind match (for refuse gold), and a set-level ``attribute_f1`` over
    asserted axes (grade↔grade only; refuse/clarify assert no axes so F1 is ``None`` there).
    """
    gold = parse(gold_text)  # gold is our own canonical target — trusted to parse
    pred = _safe_parse(pred_text)
    if pred is None:
        return {"parse_ok": False, "route_correct": False, "refuse_kind_correct": False,
                "attribute_f1": 0.0, "direction_f1": 0.0, "precision": None, "recall": None,
                "route_pred": None, "route_gold": gold.route}

    pred = canonicalize(pred)
    route_correct = pred.route == gold.route
    refuse_kind_correct = (pred.refuse_reason == gold.refuse_reason
                           if gold.route == ROUTE_REFUSE else True)

    if gold.route == ROUTE_GRADE and pred.route == ROUTE_GRADE:
        precision = _backed_fraction(pred, spec_as_mb(gold), tol)  # pred's claims backed by gold
        recall = _backed_fraction(gold, spec_as_mb(pred), tol)     # gold's claims recovered by pred
        f1 = 0.0 if (precision + recall) == 0.0 else 2 * precision * recall / (precision + recall)
        # sign-only counterpart: isolates direction correctness from magnitude calibration.
        dp, dr = _sign_fraction(pred, gold), _sign_fraction(gold, pred)
        direction_f1 = 0.0 if (dp + dr) == 0.0 else 2 * dp * dr / (dp + dr)
    else:
        precision = recall = f1 = direction_f1 = None  # not grade↔grade -> attribute F1 undefined

    return {"parse_ok": True, "route_correct": route_correct,
            "refuse_kind_correct": refuse_kind_correct, "attribute_f1": f1,
            "direction_f1": direction_f1, "precision": precision, "recall": recall,
            "route_pred": pred.route, "route_gold": gold.route}


def joint_score(cmp: dict) -> float:
    """Single per-row score in [0,1] that rewards correct routing AND (for grade) correct axes.

    ``route_correct * (attribute_f1 if grade-gold else 1.0)``. A grade row predicted as refuse -> 0;
    a refuse/clarify row routed correctly -> 1.0 (there are no axes to grade). Refuse-kind is
    reported separately (not folded here) so it stays visible rather than diluting the headline.
    """
    if not cmp["route_correct"]:
        return 0.0
    if cmp["attribute_f1"] is None:  # refuse/clarify gold: route match is the whole score
        return 1.0
    return float(cmp["attribute_f1"])
