from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

if __package__ is None or __package__ == "":
    # Allow: python backend/train.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from .models import build_critic, build_generator
from .utils import ProjectPaths, ensure_dirs, generate_images, make_dataset, save_image_grid, save_loss_history


@dataclass
class TrainConfig:
    latent_dim: int = 100
    batch_size: int = 64
    epochs: int = 2
    steps_per_epoch: int = 300  # keep demo training light/fast
    n_critic: int = 5
    lr: float = 5e-5
    clip_value: float = 0.01
    sample_every_epochs: int = 1
    sample_count: int = 16
    seed: int = 123


class WGANTrainer:
    def __init__(self, generator: keras.Model, critic: keras.Model, cfg: TrainConfig):
        self.generator = generator
        self.critic = critic
        self.cfg = cfg
        self.g_opt = keras.optimizers.RMSprop(learning_rate=cfg.lr)
        self.c_opt = keras.optimizers.RMSprop(learning_rate=cfg.lr)

    @tf.function
    def critic_train_step(self, real_images: tf.Tensor) -> tf.Tensor:
        z = tf.random.normal([self.cfg.batch_size, self.cfg.latent_dim])
        with tf.GradientTape() as tape:
            fake_images = self.generator(z, training=True)
            real_scores = self.critic(real_images, training=True)
            fake_scores = self.critic(fake_images, training=True)
            c_loss = tf.reduce_mean(fake_scores) - tf.reduce_mean(real_scores)
        grads = tape.gradient(c_loss, self.critic.trainable_variables)
        self.c_opt.apply_gradients(zip(grads, self.critic.trainable_variables))
        for w in self.critic.trainable_variables:
            w.assign(tf.clip_by_value(w, -self.cfg.clip_value, self.cfg.clip_value))
        return c_loss

    @tf.function
    def generator_train_step(self) -> tf.Tensor:
        z = tf.random.normal([self.cfg.batch_size, self.cfg.latent_dim])
        with tf.GradientTape() as tape:
            fake_images = self.generator(z, training=True)
            fake_scores = self.critic(fake_images, training=True)
            g_loss = -tf.reduce_mean(fake_scores)
        grads = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(grads, self.generator.trainable_variables))
        return g_loss

    def train_step(self, real_images: tf.Tensor) -> tuple[float, float]:
        c_losses = []
        for _ in range(self.cfg.n_critic):
            c_loss = self.critic_train_step(real_images)
            c_losses.append(float(c_loss.numpy()))
        g_loss = float(self.generator_train_step().numpy())
        return float(np.mean(c_losses)), g_loss


def train(cfg: TrainConfig) -> dict:
    paths = ProjectPaths.from_any_file(__file__)
    ensure_dirs(paths)

    ds = make_dataset(batch_size=cfg.batch_size)
    it = iter(ds.repeat())

    generator = build_generator(cfg.latent_dim)
    critic = build_critic()
    trainer = WGANTrainer(generator, critic, cfg)

    history = {"critic_loss": [], "generator_loss": [], "epoch": []}

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        c_epoch = []
        g_epoch = []
        for _ in range(cfg.steps_per_epoch):
            real_images = next(it)
            c_loss, g_loss = trainer.train_step(real_images)
            c_epoch.append(c_loss)
            g_epoch.append(g_loss)

        history["epoch"].append(epoch)
        history["critic_loss"].append(float(np.mean(c_epoch)))
        history["generator_loss"].append(float(np.mean(g_epoch)))

        if epoch % cfg.sample_every_epochs == 0:
            samples = generate_images(generator, cfg.sample_count, seed=cfg.seed + epoch, latent_dim=cfg.latent_dim)
            save_image_grid(samples, paths.samples_dir / f"samples_epoch_{epoch:03d}.png", title=f"Epoch {epoch}")

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch}/{cfg.epochs} | "
            f"critic_loss={history['critic_loss'][-1]:.4f} | "
            f"gen_loss={history['generator_loss'][-1]:.4f} | "
            f"{elapsed:.1f}s"
        )

    generator_path = paths.models_dir / "generator.h5"
    generator.save(generator_path, include_optimizer=False)

    # Optional: useful for debugging / critic score demo.
    critic_path = paths.models_dir / "critic.h5"
    critic.save(critic_path, include_optimizer=False)

    losses_path = paths.logs_dir / "loss_history.json"
    save_loss_history(losses_path, history)
    return history


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train a minimal WGAN on CIFAR-10 (demo-friendly).")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--steps-per-epoch", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--latent-dim", type=int, default=100)
    p.add_argument("--n-critic", type=int, default=5)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--clip-value", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()
    return TrainConfig(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        n_critic=args.n_critic,
        lr=args.lr,
        clip_value=args.clip_value,
        seed=args.seed,
    )


def main() -> None:
    cfg = parse_args()
    train(cfg)


if __name__ == "__main__":
    main()

