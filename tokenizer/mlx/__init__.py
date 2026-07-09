"""MLX (Apple-GPU) implementation of the LUT VQ-tokenizer.

A second, separate implementation used only for local GPU-accelerated training on Apple
Silicon (PyTorch/MPS cannot run this model's 3D transposed convs). The trained weights are
converted back into the torch ``tokenizer.model.VQVAE`` (see ``convert.py``) so freezing,
eval, the CLI, and Colab all stay torch-native. The torch implementation remains the
source of truth and portable format.

Import-safe: importing this package does not import ``mlx`` or run anything. Use the
submodules directly (``from tokenizer.mlx.model_mlx import VQVAEmlx``) or the CLI
``python -m tokenizer.mlx.train_mlx``.
"""
