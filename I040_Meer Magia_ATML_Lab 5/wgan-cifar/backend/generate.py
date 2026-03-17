from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

if __package__ is None or __package__ == "":
    # Allow: python backend/generate.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from .models import build_critic, build_generator
from .utils import ProjectPaths, critic_scores, ensure_dirs, generate_images, save_image_grid


def load_or_build_model(path: Path, builder):
    if path.exists():
        return keras.models.load_model(path, compile=False)
    return builder()


def main() -> None:
    p = argparse.ArgumentParser(description="Generate WGAN samples for demo/inspection.")
    p.add_argument("--num-images", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--latent-dim", type=int, default=100)
    args = p.parse_args()

    paths = ProjectPaths.from_any_file(__file__)
    ensure_dirs(paths)

    gen_path = paths.models_dir / "generator.h5"
    critic_path = paths.models_dir / "critic.h5"

    generator = load_or_build_model(gen_path, lambda: build_generator(args.latent_dim))
    critic = load_or_build_model(critic_path, build_critic)

    images = generate_images(generator, args.num_images, seed=args.seed, latent_dim=args.latent_dim)
    scores = critic_scores(critic, images)

    out_path = paths.samples_dir / f"generated_seed_{int(args.seed)}.png"
    save_image_grid(images, out_path, title=f"Seed {args.seed}")

    print(f"Saved: {out_path}")
    print("Critic scores (first 10):", np.asarray(scores[:10]))


if __name__ == "__main__":
    main()

