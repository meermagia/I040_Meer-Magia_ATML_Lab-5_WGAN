from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras


CIFAR10_CLASSES = [
    "Airplane",
    "Car",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck",
]


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    outputs: Path
    models_dir: Path
    samples_dir: Path
    logs_dir: Path

    @staticmethod
    def from_any_file(current_file: str | Path) -> "ProjectPaths":
        # backend/utils.py -> backend -> wgan-cifar
        root = Path(current_file).resolve().parents[1]
        outputs = root / "outputs"
        return ProjectPaths(
            root=root,
            outputs=outputs,
            models_dir=outputs / "models",
            samples_dir=outputs / "samples",
            logs_dir=outputs / "logs",
        )


def ensure_dirs(paths: ProjectPaths) -> None:
    paths.models_dir.mkdir(parents=True, exist_ok=True)
    paths.samples_dir.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)


def normalize_to_neg1_pos1(images: np.ndarray) -> np.ndarray:
    images = images.astype("float32")
    return images / 127.5 - 1.0


def denormalize_to_0_1(images: np.ndarray) -> np.ndarray:
    return np.clip((images + 1.0) / 2.0, 0.0, 1.0)


def load_cifar10_normalized() -> Tuple[np.ndarray, np.ndarray]:
    (x_train, y_train), _ = keras.datasets.cifar10.load_data()
    x_train = normalize_to_neg1_pos1(x_train)
    y_train = y_train.squeeze().astype("int64")
    return x_train, y_train


def make_dataset(batch_size: int = 64, shuffle_buffer: int = 50_000) -> tf.data.Dataset:
    x_train, _ = load_cifar10_normalized()
    ds = tf.data.Dataset.from_tensor_slices(x_train)
    ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def save_loss_history(path: Path, history: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def load_loss_history(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_image_grid(images_0_1: np.ndarray, out_path: Path, title: str | None = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    images_0_1 = np.asarray(images_0_1)
    n = int(images_0_1.shape[0])
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.0))
    axes = np.array(axes).reshape(rows, cols)
    for ax in axes.ravel():
        ax.axis("off")

    for i in range(n):
        r, c = divmod(i, cols)
        axes[r, c].imshow(images_0_1[i])

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_images(
    generator: keras.Model,
    num_images: int,
    seed: int,
    latent_dim: int = 100,
) -> np.ndarray:
    tf.random.set_seed(int(seed))
    z = tf.random.normal([int(num_images), int(latent_dim)])
    fake = generator(z, training=False)
    fake = fake.numpy()
    return denormalize_to_0_1(fake)


def critic_scores(critic: keras.Model, images_0_1: np.ndarray) -> np.ndarray:
    images = normalize_to_neg1_pos1(np.asarray(images_0_1) * 255.0)
    scores = critic(images, training=False).numpy().reshape(-1)
    return scores


def get_real_samples(num_images: int, seed: int = 0) -> np.ndarray:
    (x_train, _), _ = keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255.0
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(x_train), size=int(num_images), replace=False)
    return x_train[idx]

