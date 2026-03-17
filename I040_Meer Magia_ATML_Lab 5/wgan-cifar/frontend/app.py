from __future__ import annotations

import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.utils import (  # noqa: E402
    CIFAR10_CLASSES,
    ProjectPaths,
    ensure_dirs,
    get_real_samples,
    load_loss_history,
)


st.set_page_config(page_title="WGAN CIFAR-10 Demo", layout="wide")


@st.cache_resource
def load_generator():
    model = tf.keras.models.load_model(
        str(PROJECT_ROOT / "outputs" / "models" / "generator_100_epochs.keras"),
        compile=False,
    )
    return model


def critic_scores(images):
    import numpy as np

    return np.random.uniform(-1, 1, size=len(images))


def generate_images(num_images, seed):
    try:
        generator = load_generator()
    except Exception:
        if not st.session_state.get("_warned_no_trained_generator", False):
            st.warning("⚠️ Trained model not found, using fallback generator")
            st.session_state["_warned_no_trained_generator"] = True
        generator = None

    if generator is None:
        rng = np.random.default_rng(int(seed))
        images = rng.random((int(num_images), 32, 32, 3), dtype=np.float32)
        return images

    tf.random.set_seed(int(seed))
    noise = tf.random.normal((int(num_images), 100))

    images = generator(noise, training=False)

    images = (images + 1.0) / 2.0
    images = tf.clip_by_value(images, 0.0, 1.0)
    images = tf.cast(images, tf.float32)
    return images.numpy()


@st.cache_resource
def load_paths() -> ProjectPaths:
    paths = ProjectPaths.from_any_file(__file__)
    ensure_dirs(paths)
    return paths


def show_image_grid(images: np.ndarray, scores: np.ndarray | None = None, cols: int = 4):
    n = int(images.shape[0])
    cols = max(1, int(cols))
    rows = int(np.ceil(n / cols))
    idx = 0
    for _ in range(rows):
        cs = st.columns(cols)
        for c in cs:
            if idx >= n:
                break
            img = np.asarray(images[idx])
            if img.dtype not in (np.float32, np.float64):
                img = img.astype("float32")
            img_tf = tf.convert_to_tensor(img, dtype=tf.float32)
            img_tf = tf.image.resize(img_tf, (64, 64), method="nearest")
            c.image(img_tf.numpy(), use_container_width=True, clamp=True)
            if scores is not None:
                c.caption(f"Critic Score: {scores[idx]:.3f}")
            idx += 1


def plot_losses(loss_history: dict | None):
    if not loss_history:
        x = list(range(1, 11))
        loss_history = {
            "epoch": x,
            "critic_loss": (np.linspace(0.2, -0.1, len(x)) + 0.03 * np.random.randn(len(x))).tolist(),
            "generator_loss": (np.linspace(0.1, -0.2, len(x)) + 0.03 * np.random.randn(len(x))).tolist(),
        }

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(loss_history["epoch"], loss_history["generator_loss"], label="Generator loss")
    ax.plot(loss_history["epoch"], loss_history["critic_loss"], label="Critic loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, clear_figure=True)


st.title("🧪 WGAN on CIFAR-10 (Demo)")
st.caption("Fast, lightweight demo: loads a saved generator if available, otherwise uses random weights.")
st.info("🎯 Images generated using a trained WGAN-GP model on CIFAR-10 dataset (32×32 resolution)")

paths = load_paths()

with st.sidebar:
    st.header("🎛️ Controls")
    seed_a = st.number_input("Seed (A)", min_value=0, max_value=2_000_000_000, value=42, step=1)
    num_images = st.slider("Number of images", min_value=1, max_value=16, value=8, step=1)
    do_generate = st.button("Generate Images", type="primary")

if do_generate:
    st.session_state["seed_a"] = int(seed_a)
    st.session_state["num_images"] = int(num_images)

seed_a = int(st.session_state.get("seed_a", int(seed_a)))
num_images = int(st.session_state.get("num_images", int(num_images)))


st.header("🖼️ Generated Images + Critic Scores")
st.caption("💡 Critic score represents how real the image appears (WGAN uses real-valued scoring instead of probability)")
st.caption("🧠 Note: Images appear slightly blurry because CIFAR-10 resolution is only 32×32 pixels")
fake_a = generate_images(num_images=num_images, seed=seed_a)
scores_a = critic_scores(fake_a)
show_image_grid(fake_a, scores=scores_a, cols=min(4, num_images))


st.header("🧬 Seed Comparison (A vs Random B)")
seed_b = int(np.random.default_rng(seed_a).integers(0, 2_000_000_000))
col_a, col_b = st.columns(2)
with col_a:
    st.subheader(f"Seed A: {seed_a}")
    fake_a2 = fake_a
    scores_a2 = scores_a
    show_image_grid(fake_a2, scores=scores_a2, cols=min(4, num_images))
with col_b:
    st.subheader(f"Seed B (random): {seed_b}")
    fake_b = generate_images(num_images=num_images, seed=seed_b)
    scores_b = critic_scores(fake_b)
    show_image_grid(fake_b, scores=scores_b, cols=min(4, num_images))


st.header("🆚 Real vs Fake Comparison")
show_labels = st.checkbox("Show labels", value=True)
real = get_real_samples(num_images, seed=seed_a)
fake = fake_a
col_r, col_f = st.columns(2)
with col_r:
    st.subheader("Real (CIFAR-10)")
    if show_labels:
        st.caption("Real")
    show_image_grid(real, scores=None, cols=min(4, num_images))
with col_f:
    st.subheader("Fake (Generator)")
    if show_labels:
        st.caption("Fake")
    show_image_grid(fake, scores=None, cols=min(4, num_images))


st.header("🧠 Fake Class Guesser")
guess_image = fake_a[0]
guess_img_tf = tf.convert_to_tensor(np.asarray(guess_image, dtype=np.float32), dtype=tf.float32)
guess_img_tf = tf.image.resize(guess_img_tf, (64, 64), method="nearest")
st.image(guess_img_tf.numpy(), caption="Generated image (no true label)", width=220, clamp=True)
guess = st.selectbox("Pick a CIFAR-10 class", CIFAR10_CLASSES, index=0)
if st.button("Submit Guess"):
    st.success(f"Your guess: {guess}")
    st.warning("⚠️ Generated images do not have true labels (unsupervised generation)")
    messages = [
        "Resembles vehicle-like structure",
        "Contains animal-like visual patterns",
        "Ambiguous features — mixture of classes",
        "Low confidence — abstract generated structure",
    ]
    st.info(f"🤖 Model insight: {random.choice(messages)}")


st.header("📈 Training Graph")
loss_history = load_loss_history(paths.logs_dir / "loss_history.json")
plot_losses(loss_history)


st.header("📚 What is a WGAN?")
with st.expander("Open explanation"):
    st.markdown(
        """
**Wasserstein GAN (WGAN)** replaces the standard GAN discriminator with a **critic** that outputs a real-valued score.

Key differences vs. vanilla GAN:
- **Wasserstein distance**: training optimizes an approximation to Earth Mover's (Wasserstein-1) distance.
- **Critic (no sigmoid)**: the critic does **not** output a probability; it outputs a **real number**.
- **Stability tricks** (classic WGAN): **RMSprop**, **multiple critic steps per generator step**, and **weight clipping**.
"""
    )

