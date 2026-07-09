"""LUT VQ-tokenizer configuration + pinned domain/geometry constants (Stages 7-8).

Pure and torch-free so it is import-safe and testable without the ``ml`` extras.
Everything the encoder/decoder/VQ, the training loop, and the frozen manifest must
agree on lives here so the whole system shares one identity.

Cross-references:
  * model_architecture.md "LUT Tokenizer" (geometry, codebook, manifest fields);
  * training_plan_colab.md "Stage 1: LUT Tokenizer Training" (hyperparameters, gate);
  * eval/cube_io.py (tensor convention ``lut[r,g,b,c]``, identity grid, .cube order);
  * eval/vocab.py (token suffix <-> codebook index == identity).

Changing any pinned value here changes the tokenizer identity and requires a new
manifest + regenerated targets (Canonical LUT Contract).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# Pinned domain identity — imported (not re-declared) so there is exactly one source.
from eval.cube_io import GRID_SIZE  # 17
from eval.color_pipeline import COLOR_PIPELINE_VERSION
from eval.schemas import CANONICAL_DOMAIN_ID, CUBE_SERIALIZATION_VERSION

# --- architecture identity ----------------------------------------------------------
# v2 (audit remediation): decoder upsampling switched from stride-2 ConvTranspose3d to
# resize(trilinear)+Conv3d (kills the Odena checkerboard artifact), convs use
# `replicate` boundary padding (fixes corner/edge ΔE bias), and the VQ nearest-code
# search runs in float64 with lowest-index tie-break (cross-hardware token stability).
# The 17->4->17 / cb256 / t64 token grammar is unchanged; only the decoder internals,
# padding, and VQ arithmetic differ, so v1 checkpoints are not weight-compatible.
TOKENIZER_ARCH_VERSION = "vq_v2_srgbres_17to4_cb256_t64"

GRID = GRID_SIZE          # 17-node LUT grid
LATENT_GRID = 4           # 4x4x4 latent
TOKEN_COUNT = 64          # 4**3 code positions -> 64 tokens per supported output
CODEBOOK_SIZE = 256       # <lut_000>..<lut_255>

assert LATENT_GRID ** 3 == TOKEN_COUNT, "token_count must equal latent_grid**3"

# --- pinned tensor / flatten orders (recorded verbatim in the frozen manifest) ------
# On-disk residual arrays are indexed ``[r, g, b, channel]`` (eval/cube_io.py, audit A4);
# node i -> input value i/(N-1). For Conv3d we move channels first and map the three
# spatial axes as (X=r, Y=g, Z=b):  [r,g,b,c]  <->  [N, C=3, X=r, Y=g, Z=b].
TENSOR_AXIS_ORDER = "residual_rgbc_to_ncxyz__x=r,y=g,z=b"

# The 4x4x4 latent grid is flattened to 64 tokens in C-order over (X, Y, Z) with Z
# fastest, i.e. ``token = x*16 + y*4 + z``. This is the pinned latent_flatten_order.
LATENT_FLATTEN_ORDER = "xyz_c_order_z_fastest__token=x*16+y*4+z"

# Codebook index k <-> token suffix "kkk" is the identity map (eval/vocab.py).
TOKEN_SUFFIX_TO_CODEBOOK_INDEX = "identity"

# .cube table order (R fastest) — recorded for the manifest; the walk lives in cube_io.
CUBE_TABLE_ORDER = "rgb_r_fastest__b_outer_g_mid_r_inner"


@dataclass(frozen=True)
class TokenizerConfig:
    """Immutable tokenizer + training configuration.

    Geometry defaults are pinned to the Canonical LUT Contract and must not drift.
    Training defaults follow training_plan_colab.md "Stage 1" starting values.
    """

    # -- geometry (pinned) --
    grid: int = GRID
    latent_grid: int = LATENT_GRID
    token_count: int = TOKEN_COUNT
    codebook_size: int = CODEBOOK_SIZE

    # -- model capacity --
    code_dim: int = 64                 # codebook embedding dimension = latent channels at VQ
    enc_channels: tuple[int, int] = (64, 128)   # hidden widths for 17->9 and 9->5
    dec_channels: tuple[int, int] = (128, 64)    # hidden widths for 4->5 and 5->9
    norm_groups: int = 8               # GroupNorm groups (must divide each hidden width)

    # -- vector quantizer (EMA) --
    ema_decay: float = 0.99
    ema_eps: float = 1.0e-5
    commit_beta: float = 0.25          # commitment loss weight (training_plan_colab.md)
    dead_code_threshold: float = 1.0   # revive a code whose EMA cluster size drops below this
    dead_code_revival: bool = True

    # -- loss weights (L_commit is scaled by commit_beta inside the VQ) --
    # Rebalanced from v1 after a gradient-balance audit (see losses.py): residuals are
    # O(0.05) so raw L_recon is tiny and its gradient was drowned by the perceptual terms,
    # which structurally starved the PSNR>=35 gate (that gate is driven ONLY by L_recon).
    # w_recon is raised ~1.5 orders so reconstruction/PSNR and ΔE reach their gates
    # together; w_neutral is cut because L_neutral is now a small target-relative term
    # (was the dominant gradient when it pushed neutral chroma toward absolute zero).
    # v2.1 (bilevel-loop tuned, run tok_hp_v1): w_recon 25->35 and w_deltaE 0.10->0.25 were the
    # loop's winning move (proxy meanΔE 2.51->2.36); w_tail 0.05->0.18 + tail_frac 0.05->0.10 was
    # the lowest-proxy observation (t=5, 2.28) and directly strengthens the p95/p99 tail gate.
    # Tuned against a 1200-step MLX proxy of the Stage-1 gate on the frozen split; other weights
    # left at the audited v2 starting point.
    # v2.3 (run 3): the 40k tail-weighted run cleared p99/max/scraped_web-mean but was left with
    # two marginal worst-LUT fails — p5-PSNR 29.29 (need >=30) and scraped_web p95 5.20 (need <=5.0).
    # v2.4 (run 4): run 3's heavier reweight (w_recon 50, w_tail 0.30, tail_frac 0.15) REGRESSED both
    # binding metrics (p5-PSNR 29.29->29.10, scraped_web p95 5.20->5.28) — loss reweighting has
    # saturated, and p5-PSNR vs scraped_web-p95 are anti-correlated across checkpoints (a Pareto
    # tension: no single checkpoint clears both). So revert to run 2's better-balanced weights
    # (w_recon 35, w_tail 0.25, tail_frac 0.12) and switch levers to the documented first-line tail
    # remedy — neutral-preserving scale-jitter augmentation (ADR-0017), enabled at the CLI via
    # --augment --scale-jitter 0.05 (a mechanism orthogonal to reweighting).
    w_recon: float = 35.0              # LUT-grid reconstruction (MSE on residual) — drives PSNR
    w_deltaE: float = 0.25             # perceptual CIEDE2000 mean over all grid nodes
    w_tail: float = 0.25               # tail-aware: mean of the worst `tail_frac` node ΔE per LUT
    w_smooth: float = 0.01             # 3D Laplacian smoothness on the reconstructed residual
    w_clip: float = 1.0                # penalty for absolute LUT values outside [0,1]
    w_neutral: float = 0.05            # neutral-axis (r=g=b) target-relative chroma penalty
    w_commit: float = 1.0              # multiplier on the VQ commitment/codebook loss
    tail_frac: float = 0.12            # fraction of worst nodes per LUT used by the tail term

    # -- optimization (training_plan_colab.md Stage 1 starting values) --
    lr: float = 3.0e-4
    lr_min: float = 3.0e-5             # cosine-decay floor (0.1*lr); reached at max_steps
    lr_decay: bool = True              # warmup -> cosine decay to lr_min (polishes the tail)
    weight_decay: float = 1.0e-4
    grad_clip: float = 1.0
    batch_size: int = 16              # MLX Metal 3D-conv throughput (128 ~1s/step; 16 ~0.14s);
                                      # small batch also aids VQ codebook usage. Raise on CUDA/Colab.
    max_steps: int = 20000
    warmup_steps: int = 500
    seed: int = 0

    # -- train-only data augmentation (neutral-preserving residual scale jitter) --
    # Off by default; the documented first-line remedy when codebook usage or tail ΔE
    # fail the gate (model_architecture.md Stage 1, ADR-0017). Applied only to the
    # training ResidualDataset — the dev-holdout gate never augments.
    augment: bool = False
    scale_jitter: float = 0.0          # e.g. 0.05 => residual *= 1 +/- U(0,0.05)

    # -- checkpointing --
    ckpt_every: int = 1000
    eval_every: int = 1000
    keep_last: int = 3

    # -- provenance ids (recorded in checkpoint + frozen manifest) --
    arch_version: str = TOKENIZER_ARCH_VERSION
    canonical_domain_id: str = CANONICAL_DOMAIN_ID
    color_pipeline_version: str = COLOR_PIPELINE_VERSION
    cube_serialization_version: str = CUBE_SERIALIZATION_VERSION
    tensor_axis_order: str = TENSOR_AXIS_ORDER
    latent_flatten_order: str = LATENT_FLATTEN_ORDER
    token_suffix_to_codebook_index: str = TOKEN_SUFFIX_TO_CODEBOOK_INDEX

    def __post_init__(self) -> None:
        # Fail loudly on any geometry drift from the pinned contract.
        if self.grid != GRID:
            raise ValueError(f"grid must be {GRID} (Canonical LUT Contract), got {self.grid}")
        if self.latent_grid ** 3 != self.token_count:
            raise ValueError("latent_grid**3 must equal token_count")
        if self.codebook_size != CODEBOOK_SIZE:
            raise ValueError(f"codebook_size must be {CODEBOOK_SIZE}, got {self.codebook_size}")
        for w in (*self.enc_channels, *self.dec_channels, self.code_dim):
            if w % self.norm_groups != 0:
                raise ValueError(
                    f"norm_groups={self.norm_groups} must divide every channel width "
                    f"(offending width {w})"
                )

    def to_dict(self) -> dict:
        return asdict(self)


# The default v1 configuration.
DEFAULT_CONFIG = TokenizerConfig()
