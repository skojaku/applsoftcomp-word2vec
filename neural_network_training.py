# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "marimo>=0.9.0",
#   "numpy>=1.24",
#   "matplotlib>=3.7",
#   "Pillow>=10.0",
#   "plotly>=5.0",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium", css_file="marimo_lecture_note_theme.css")


@app.cell(hide_code=True)
def _():
    import matplotlib as _mpl

    _mpl.use("Agg")
    # Match lecture-note sans stack (applied-soft-comp/docs/lecture-note/scss/style.scss)
    _mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "-apple-system",
                "BlinkMacSystemFont",
                "Segoe UI",
                "Helvetica",
                "Arial",
                "DejaVu Sans",
                "sans-serif",
            ],
            "font.size": 13,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part I: Training a tiny neural network: cat vs dog **faces** (AFHQ)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(
        mo.image(
            src="./figs/frank-rosenblatt.png",
            alt="Frank Rosenblatt's perceptron (source: Wikimedia Commons)",
            width="100%",
            rounded=True,
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What we want to do

    We will run a toy Rosenblatt's neural network that classifies an image into a cat or dog face.

    ### How we do it

    The images are **grayscale**, represented as a marix of pixel values. For each pixel, we multiply a weight, one weight per pixel. We then compute the output by summing up the weighted pixel values:

    \[
    \text{output} = \mathbf{w}^\top \mathbf{x} + b.
    \]

    where b is the offset. Here \(\mathbf{x}\) is the grayscale face flattened (pixels in about \([0,1]\)). Think of this as a weighted average of pixel values.

    We consider that an image is a dog if output>0, otherwise cat.

    ### Training

    We want to find the weights \(\mathbf{w}\) that correctly classify all training images. We will do this by looking at the first training image that the model gets wrong, then clicking **Add** or **Subtract** to fix it.

    More specifically, when the classifier makes incorrect classification, the weights are updated by:

    \[

    w \leftarrow w \pm \eta x

    \]

    where $\eta$ is the learning rate, controlling the update size. $\pm$ depends on whether the misclassification is made for a dog face (should be +) or a cat face (should be -).

    ### Try yourself

    Let's learn how it works. We will give you misclassified images. You click whether to add to or subtract the pixel values from the weights. Click *Add (+)* if a dog face is misclassified. Click *Subtract* if a cat face.

    /// Admonition | Data

    Faces are from **[AFHQ](https://github.com/clovaai/stargan-v2)** (prepared via
    [Hugging Face `huggan/AFHQ`](https://huggingface.co/datasets/huggan/AFHQ)). A helper script (./scripts/prepare_afhq_subset.py) resizes to
    a square grid (default **\(64\times64\)**), converts to **grayscale**, and saves **PNG files** under
    `data/afhq_catdog/` that this notebook loads directly.

    ///
    """)
    return


@app.cell(hide_code=True)
def _(
    CLASS_NAMES,
    add_btn,
    click_count,
    compute_output,
    lr_slider,
    mo,
    randomize_btn,
    subtract_btn,
    train_images,
    train_labels,
    training_mistake_index,
    weights,
):
    import io as _ioc
    import matplotlib.pyplot as _plt_ctrl
    import numpy as _np

    _FIG_W = 6.0

    _w = weights()
    mist_idx = training_mistake_index(_w)
    _eta = float(lr_slider.value)

    if mist_idx is None:
        compare_widget = mo.md("*No figure — all training faces are classified correctly.*")
        status_line = (
            f"**All training images correct.** 100% on this demo set. "
            f"Learning rate **η = {_eta:g}** · training clicks **{int(click_count())}** — try **Randomize** to practice again."
        )
    else:
        idx = int(mist_idx)
        img = train_images[idx]
        flat = img.reshape(-1).astype(_np.float64, copy=False)
        score = compute_output(_w, flat)
        pred = 1 if score > 0 else 0
        true_label = int(train_labels[idx])

        _ih, _iw = int(img.shape[0]), int(img.shape[1])
        W_mat = _w.reshape(_ih, _iw)
        _lim_side = float(_np.max(_np.abs(W_mat))) + 1e-9

        _fig_pair, (_ax_im, _ax_wm) = _plt_ctrl.subplots(1, 2, figsize=(_FIG_W, _FIG_W * 0.42))
        _ax_im.imshow(
            _np.clip(img, 0.0, 1.0),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        _ax_im.set_title("Face")
        _ax_im.axis("off")
        _wm = _ax_wm.imshow(
            W_mat,
            cmap="RdBu_r",
            vmin=-_lim_side,
            vmax=_lim_side,
            interpolation="nearest",
        )
        _ax_wm.set_title("Weights W")
        _ax_wm.axis("off")
        _fig_pair.colorbar(_wm, ax=_ax_wm, fraction=0.046, pad=0.06)
        _plt_ctrl.tight_layout()
        _png_cmp = _ioc.BytesIO()
        _fig_pair.savefig(
            _png_cmp,
            format="png",
            dpi=110,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        _plt_ctrl.close(_fig_pair)
        compare_widget = mo.image(_png_cmp.getvalue(), width=640)

        pred_name = CLASS_NAMES[pred]
        true_name = CLASS_NAMES[true_label]
        status_line = (
            f"First mistake **#{idx}** in train order · True **{true_name}** · Pred **{pred_name}** "
            f"· score **{score:.3f}** · η **{_eta:g}** · clicks **{int(click_count())}**"
        )

    buttons = mo.hstack(
        [
            randomize_btn,
            add_btn,
            subtract_btn,
        ],
        justify="start",
        gap=1,
    )

    mo.vstack(
        [
            mo.md("## Interactive training"),
            lr_slider,
            compare_widget,
            buttons,
            mo.md(status_line),
        ],
        gap=1.25,
    )
    return


@app.cell(hide_code=True)
def _(
    accuracy_history,
    eval_images,
    eval_labels,
    evaluate_accuracy,
    mo,
    weights,
):
    import io as _ioa
    import matplotlib.pyplot as _plt_acc

    hist = accuracy_history()
    _w = weights()
    current_acc = float(evaluate_accuracy(_w, eval_images, eval_labels))

    if not hist:
        _acc_panel = mo.vstack(
            [
                mo.md("## Accuracy vs training clicks *(same images as training — demo only)*"),
                mo.callout(
                    mo.md(
                        f"**Current accuracy:** **{current_acc * 100:.1f}%**  \n"
                        "*No training clicks recorded yet — use **Add** or **Subtract** to start a curve.*"
                    ),
                    kind="neutral",
                ),
            ]
        )
    else:
        xs = [h[0] for h in hist]
        ys = [h[1] * 100.0 for h in hist]

        _fig, _ax = _plt_acc.subplots(figsize=(6.4, 3.4))
        _ax.plot(xs, ys, marker="o", color="#1f77b4", label="Accuracy (train = eval)")
        _ax.axhline(50.0, color="#9e9e9e", linestyle="--", linewidth=1.5, label="50% baseline")
        _ax.set_xlabel("Training click (each Add or Subtract)")
        _ax.set_ylabel("Accuracy (%)")
        _ax.set_title("Accuracy on demo set (same as training)")
        _ax.set_ylim(0, 100)
        _ax.grid(True, alpha=0.3)
        _ax.legend(loc="best")
        _plt_acc.tight_layout()

        _png_acc = _ioa.BytesIO()
        _fig.savefig(
            _png_acc,
            format="png",
            dpi=110,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        _plt_acc.close(_fig)
        _acc_fig_widget = mo.image(_png_acc.getvalue(), width=640)

        _acc_panel = mo.vstack(
            [
                mo.md("## Accuracy vs training clicks *(same images as training — demo only)*"),
                mo.callout(
                    mo.md(f"**Current accuracy:** **{current_acc * 100:.1f}%**"),
                    kind="info" if current_acc >= 0.55 else "neutral",
                ),
                _acc_fig_widget,
            ]
        )

    _acc_panel
    return


@app.cell(hide_code=True)
def _(mo):
    import json as _json
    from pathlib import Path as _Path

    from PIL import Image as _PilImage

    import numpy as _np

    _roots: list[_Path] = []
    try:
        _roots.append(_Path(__file__).resolve().parent)
    except NameError:
        pass
    _roots.append(_Path.cwd())

    _manifest_path: _Path | None = None
    for _r in _roots:
        _cand = _r / "data" / "afhq_catdog" / "manifest.json"
        if _cand.is_file():
            _manifest_path = _cand
            break

    if _manifest_path is None:
        mo.callout(
            mo.md(
                "**Missing `data/afhq_catdog/manifest.json`.** From the project root run:\n\n"
                "```\nuv run scripts/prepare_afhq_subset.py\n```\n\n"
                "That downloads AFHQ once (Hugging Face), then saves **220 PNGs** here. "
                "Re-open the notebook after it finishes."
            ),
            kind="danger",
        )
        raise FileNotFoundError("data/afhq_catdog/manifest.json not found (run scripts/prepare_afhq_subset.py)")

    _base = _manifest_path.parent
    _man = _json.loads(_manifest_path.read_text(encoding="utf-8"))
    _grid = int(_man["grid_size"])
    CLASS_NAMES = list(_man.get("class_names", ["Cat", "Dog"]))


    def _load_split(entries: list) -> tuple[_np.ndarray, _np.ndarray]:
        _imgs = []
        _labs = []
        for _e in entries:
            _p = _base / str(_e["file"])
            if not _p.is_file():
                raise FileNotFoundError(f"Missing image file: {_p}")
            _arr = _np.asarray(_PilImage.open(_p).convert("L"), dtype=_np.float32) / 255.0
            _imgs.append(_arr)
            _labs.append(int(_e["label"]))
        _x = _np.stack(_imgs, axis=0)
        _y = _np.array(_labs, dtype=_np.int64)
        return _x, _y


    train_images, train_labels = _load_split(_man["train"])
    # Demo: measure accuracy on the same faces as training (no holdout split in the notebook).
    eval_images, eval_labels = train_images, train_labels

    if train_images.shape[1] != _grid or train_images.shape[2] != _grid:
        raise ValueError(
            f"manifest says grid_size={_grid} but train images are {train_images.shape[1]}×{train_images.shape[2]}"
        )

    _n_w = _grid * _grid
    try:
        _path_show = str(_base.resolve().relative_to(_Path.cwd().resolve()))
    except ValueError:
        _path_show = str(_base.resolve())
    return CLASS_NAMES, eval_images, eval_labels, train_images, train_labels


@app.cell(hide_code=True)
def _():
    import numpy as _np


    def compute_output(weights, image_flat) -> float:
        return float(_np.dot(weights, image_flat))


    def evaluate_accuracy(weights, images, labels) -> float:
        w = _np.asarray(weights, dtype=_np.float64).reshape(-1)
        flat = images.reshape(len(images), -1).astype(_np.float64, copy=False)
        scores = flat @ w
        preds = (scores > 0).astype(_np.int64)
        return float(_np.mean(preds == labels))

    return compute_output, evaluate_accuracy


@app.cell(hide_code=True)
def _(
    eval_images,
    eval_labels,
    evaluate_accuracy,
    mo,
    train_images,
    train_labels,
):
    import numpy as _np

    FLAT_SIZE = int(train_images.shape[1] * train_images.shape[2])

    weights, set_weights = mo.state(_np.zeros(FLAT_SIZE, dtype=_np.float32))
    click_count, set_click_count = mo.state(0)
    accuracy_history, set_accuracy_history = mo.state([])


    def training_mistake_index(w_vec) -> int | None:
        _w64 = _np.asarray(w_vec, dtype=_np.float64).reshape(-1)
        for _i in range(int(train_images.shape[0])):
            _x = train_images[_i].reshape(-1).astype(_np.float64, copy=False)
            _s = float(_np.dot(_w64, _x))
            _pred = 1 if _s > 0 else 0
            if _pred != int(train_labels[_i]):
                return int(_i)
        return None


    def on_randomize(_value=None) -> None:
        set_weights((_np.random.randn(FLAT_SIZE).astype(_np.float32) * 0.01))
        set_click_count(0)
        set_accuracy_history([])


    def on_add(_value=None) -> None:
        _eta = float(lr_slider.value)
        if _eta <= 0.0:
            return
        _w0 = weights()
        idx = training_mistake_index(_w0)
        if idx is None:
            return
        _x = train_images[idx].reshape(-1).astype(_np.float32, copy=False)
        _w = _w0.astype(_np.float32, copy=True)
        _w += _np.float32(_eta) * _x
        set_weights(_np.asarray(_w, dtype=_np.float32, copy=True))
        c = int(click_count()) + 1
        set_click_count(c)
        acc = evaluate_accuracy(_w, eval_images, eval_labels)
        hist = list(accuracy_history())
        hist.append((c, acc))
        set_accuracy_history(hist)


    def on_subtract(_value=None) -> None:
        _eta = float(lr_slider.value)
        if _eta <= 0.0:
            return
        _w0 = weights()
        idx = training_mistake_index(_w0)
        if idx is None:
            return
        if int(train_labels[idx]) != 0:
            return
        _x = train_images[idx].reshape(-1).astype(_np.float32, copy=False)
        _w = _w0.astype(_np.float32, copy=True)
        _w -= _np.float32(_eta) * _x
        set_weights(_np.asarray(_w, dtype=_np.float32, copy=True))
        c = int(click_count()) + 1
        set_click_count(c)
        acc = evaluate_accuracy(_w, eval_images, eval_labels)
        hist = list(accuracy_history())
        hist.append((c, acc))
        set_accuracy_history(hist)


    lr_slider = mo.ui.slider(
        start=0.05,
        stop=2.5,
        step=0.05,
        value=1.0,
        label="Learning rate η",
        show_value=True,
        include_input=True,
        full_width=True,
    )
    randomize_btn = mo.ui.button(label="Randomize", on_click=on_randomize, kind="neutral")
    add_btn = mo.ui.button(
        label="Add (+)",
        on_click=on_add,
        kind="warn",
        tooltip="w ← w + η·x on the first misclassified training face (in order).",
    )
    subtract_btn = mo.ui.button(
        label="Subtract (−) cat",
        on_click=on_subtract,
        kind="danger",
        tooltip="w ← w − η·x when that face is truly cat; no-op if dog or no mistake.",
    )
    return (
        accuracy_history,
        add_btn,
        click_count,
        lr_slider,
        randomize_btn,
        subtract_btn,
        training_mistake_index,
        weights,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Part II — From Pixels to Words: Word Embeddings
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(
        mo.image(
            src="./figs/word-embedding.png",
            alt="Rosenblatt's perceptron for word embeddings",
            width="100%",
            rounded=True,
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### One-hot encoding: words as vectors

    The idea of Rosenblatt's neural networks can be extended to natural language processing. To simplify the story, let's focus on a unit of language, i.e., *words*.

    Words are not numerical unlike images. But we can assign a numerical ID to each number. For example, word "cat" has ID 0, and word "cat" has ID 1. For a vocabulary of $V$ words, these IDs can be equivalently expressed as *one-hot vector*  $\mathbf{x}_i \in \{0,1\}^V$ with all entries being zeros except a single 1 at the position of the ID.

    \[
    \text{cat} \to [1,0,0,\ldots,0], \\
    \text{dog} \to [0,1,0,\ldots,0],  \\
    \text{bird} \to [0,0,1,\ldots,0], \\
    \vdots
    \]

    Now plug this into the perceptron from Part I. Let's think about the case of word "dog".


    $$
    \text{output} = \underbrace{[w_0, w_1, \ldots, w_V]}_{\text{\normalsize weights}} \cdot \underbrace{[0, 1, 0, \ldots, 0]}_{\text{\normalsize dog vector}} = w_1
    $$

    The dot product simply selects the 2nd weight. So the weight $w_2$ is the score for word "dog". The same applies to other words, i.e., the dot product with the one-hot vector for the $i$th word simply selects the $i$th weight as that word's score.

    Goemetrically, the neural network maps words into a single horizontal line. This could be useful for representing a spectrum of sentiments of words (e.g., "happy" vs "unhappy").

    ### Multiple output neurons

    A single score can represent a spectrum of bipolar concepts. But it fails when there are more than two groups. A solution is to create another output neuron to represent a word as a vector, instead of a single scalar. The vectors are called *word embeddings*.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    import io as _io_1d
    import matplotlib.pyplot as _plt_1d

    # Hand-crafted 1-D positions that show overlap
    _CAT_1D = {
        "animal": (["cat", "dog", "bird"], [-0.75, -0.20, 0.50], "#1f77b4"),
        "action": (["run", "jump", "walk"], [-0.45, 0.30, 0.65], "#ff7f0e"),
        "emotion": (["happy", "sad", "calm"], [0.10, -0.40, 0.40], "#9467bd"),
    }
    # Ideal 2-D positions (manually separated clusters)
    _POS_2D = {
        "cat": (-1.2, -0.9),
        "dog": (-0.9, -0.5),
        "bird": (-1.1, -0.2),
        "run": (0.9, -0.8),
        "jump": (1.1, -0.3),
        "walk": (0.7, -0.6),
        "happy": (-0.1, 1.1),
        "sad": (0.4, 0.8),
        "calm": (-0.4, 0.9),
    }

    _fig1d, (_ax1, _ax2) = _plt_1d.subplots(1, 2, figsize=(11, 3.2))

    # Left: 1-D number line
    for _cat, (_words, _xs, _col) in _CAT_1D.items():
        _ax1.scatter(_xs, [0] * 3, color=_col, s=130, zorder=3, label=_cat)
        for _w, _x in zip(_words, _xs):
            _ax1.annotate(
                _w,
                (_x, 0),
                xytext=(0, 12),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
    _ax1.axhline(0, color="#aaa", linewidth=1.5)
    _ax1.set_xlim(-1.3, 1.1)
    _ax1.set_ylim(-0.3, 0.35)
    _ax1.set_yticks([])
    _ax1.set_xlabel("Single score  $w_i$", fontsize=11)
    _ax1.set_title("1 output neuron.\n(3 groups can't be separated cleanly)", fontsize=11)
    _ax1.legend(loc="lower right", fontsize=9)

    # Right: ideal 2-D clusters
    for _cat, (_words, _, _col) in _CAT_1D.items():
        _xs2 = [_POS_2D[_w][0] for _w in _words]
        _ys2 = [_POS_2D[_w][1] for _w in _words]
        _ax2.scatter(_xs2, _ys2, color=_col, s=130, zorder=3, label=_cat)
        for _w in _words:
            _ax2.annotate(_w, _POS_2D[_w], xytext=(6, 4), textcoords="offset points", fontsize=9)
    _ax2.set_xlabel("Dimension 1", fontsize=11)
    _ax2.set_ylabel("Dimension 2", fontsize=11)
    _ax2.set_title("2 output neurons. 2-D plane\n(clusters emerge!)", fontsize=11)
    _ax2.legend(loc="upper right", fontsize=9)
    _ax2.grid(True, alpha=0.3)

    _plt_1d.tight_layout()
    _buf_1d = _io_1d.BytesIO()
    _fig1d.savefig(_buf_1d, format="png", dpi=110, bbox_inches="tight", facecolor="white")
    _plt_1d.close(_fig1d)

    mo.vstack(
        [
            mo.image(_buf_1d.getvalue(), width=740),
        ],
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The weight is now a matrix $\textbf{W} \in \mathbb{R}^{2 \times V}$, not a vector, i.e.,

    \[
    \textbf{W} =
    \begin{bmatrix}
    w_{0,1} & w_{0,2} & w_{0,3} & \ldots & w_{0,V} \\
    w_{1,1} & w_{1,2} & w_{1,3} & \ldots & w_{1,V}
    \end{bmatrix}
    \]

    For word "dog" (ID: 1):

    \[
    \begin{align}
    \text{output} &= \begin{bmatrix}
    w_{0,0} & w_{0,1} & w_{0,2} & \ldots & w_{0,V} \\
    w_{1,0} & w_{1,1} & w_{1,2} & \ldots & w_{1,V}
    \end{bmatrix}
    \cdot
    \begin{bmatrix}
    0 \\
    1 \\
    0 \\
    \vdots
    \end{bmatrix}
    \\
    &=\begin{bmatrix}
    w_{0,1} \\
    w_{1,1}
    \end{bmatrix}.
    \end{align}
    \]

    As is the case for the 1d case, the one-hot vector *selects* a column of $\textbf{W}$. A dog is represented as 2d vector. In general, we can extend to $K$ output neurons to get $K$-dimensional embeddings. Here, we focus on $K=2$ to simplify the story.

    ### Training

    How can we learn the word embeddings? A simple idea is to present two words to the model and tell it whether they are related in some sense or not. We then updates the weights such that the related words are closer and unrelated words are farther.

    Below are 15 words in three groups, i.e.,  animals, actions, emotions (5 each).
    The embeddings start *random*. Your goal is to train until each group forms its own cluster.

    - Click *Closer ↔* or *Farther ↔* to update one pair at a time.
    - Click *Train 50 steps ▶* to run 50 correct updates automatically and watch the clusters form.
    - Click *Randomize* to reset and start fresh.
    """)
    return


@app.cell(hide_code=True)
def _():
    # Three clean semantic clusters (5 words each = 15 total)
    VOCAB = [
        "cat",
        "dog",
        "bird",
        "fish",
        "horse",
        "run",
        "walk",
        "jump",
        "swim",
        "fly",
        "happy",
        "sad",
        "angry",
        "calm",
        "joyful",
    ]
    WORD_INDEX = {w: i for i, w in enumerate(VOCAB)}

    CATEGORIES = {
        "animal": ["cat", "dog", "bird", "fish", "horse"],
        "action": ["run", "walk", "jump", "swim", "fly"],
        "emotion": ["happy", "sad", "angry", "calm", "joyful"],
    }
    CATEGORY_COLORS = {
        "animal": "#1f77b4",
        "action": "#ff7f0e",
        "emotion": "#9467bd",
    }

    # Positive pairs: within each category
    POS_PAIRS = [
        ("cat", "dog"),
        ("cat", "bird"),
        ("dog", "fish"),
        ("fish", "horse"),
        ("bird", "horse"),
        ("run", "walk"),
        ("run", "jump"),
        ("walk", "swim"),
        ("jump", "fly"),
        ("swim", "fly"),
        ("happy", "calm"),
        ("happy", "joyful"),
        ("sad", "angry"),
        ("calm", "sad"),
        ("joyful", "angry"),
    ]
    # Negative pairs: across categories
    NEG_PAIRS = [
        ("cat", "run"),
        ("dog", "happy"),
        ("bird", "sad"),
        ("fish", "jump"),
        ("horse", "calm"),
        ("run", "happy"),
        ("walk", "sad"),
        ("jump", "angry"),
        ("swim", "joyful"),
        ("fly", "calm"),
        ("cat", "happy"),
        ("dog", "run"),
    ]

    import random as _rng

    _queue = [(a, b, True) for a, b in POS_PAIRS] + [(a, b, False) for a, b in NEG_PAIRS]
    _rng.Random(42).shuffle(_queue)
    PAIR_QUEUE = _queue
    return (
        CATEGORIES,
        CATEGORY_COLORS,
        NEG_PAIRS,
        PAIR_QUEUE,
        POS_PAIRS,
        VOCAB,
        WORD_INDEX,
    )


@app.cell(hide_code=True)
def _(NEG_PAIRS, POS_PAIRS, WORD_INDEX):
    import numpy as _np_sim


    def compute_similarity(emb, word_a, word_b) -> float:
        ea = emb[WORD_INDEX[word_a]].astype(_np_sim.float64)
        eb = emb[WORD_INDEX[word_b]].astype(_np_sim.float64)
        na = float(_np_sim.linalg.norm(ea))
        nb = float(_np_sim.linalg.norm(eb))
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float(_np_sim.dot(ea, eb) / (na * nb))


    def evaluate_pair_accuracy(emb):
        pos_ok = sum(1 for a, b in POS_PAIRS if compute_similarity(emb, a, b) > 0)
        neg_ok = sum(1 for a, b in NEG_PAIRS if compute_similarity(emb, a, b) < 0)
        pos_acc = pos_ok / max(1, len(POS_PAIRS))
        neg_acc = neg_ok / max(1, len(NEG_PAIRS))
        overall = (pos_ok + neg_ok) / max(1, len(POS_PAIRS) + len(NEG_PAIRS))
        return pos_acc, neg_acc, overall

    return (evaluate_pair_accuracy,)


@app.cell(hide_code=True)
def _(PAIR_QUEUE, VOCAB, WORD_INDEX, evaluate_pair_accuracy, mo):
    import numpy as _np_emb

    _V = len(VOCAB)

    emb, set_emb = mo.state(_np_emb.random.randn(_V, 2).astype(_np_emb.float32) * 0.1)
    prev_pair_indices, set_prev_pair_indices = mo.state(None)  # (ia, ib) of last trained pair
    emb_click_count, set_emb_click_count = mo.state(0)
    emb_pair_idx, set_emb_pair_idx = mo.state(0)
    emb_accuracy_history, set_emb_accuracy_history = mo.state([])

    # Slider defined before callbacks so closures can reference it
    emb_lr_slider = mo.ui.slider(
        start=0.05,
        stop=0.5,
        step=0.05,
        value=0.1,
        label="Learning rate η",
        show_value=True,
        include_input=True,
        full_width=True,
    )


    def _step(e, pi_raw, eta, is_closer):
        """Apply one update step; returns updated embedding array."""
        _pi = pi_raw % len(PAIR_QUEUE)
        _wa, _wb, _ = PAIR_QUEUE[_pi]
        _ia, _ib = WORD_INDEX[_wa], WORD_INDEX[_wb]
        _ea = e[_ia].astype(_np_emb.float64, copy=True)
        _eb = e[_ib].astype(_np_emb.float64, copy=True)
        _d = _eb - _ea
        _ne = e.astype(_np_emb.float32, copy=True)
        if is_closer:
            _ne[_ia] = (_ea + eta * _d).astype(_np_emb.float32)
            _ne[_ib] = (_eb - eta * _d).astype(_np_emb.float32)
        else:
            _ne[_ia] = (_ea - eta * _d).astype(_np_emb.float32)
            _ne[_ib] = (_eb + eta * _d).astype(_np_emb.float32)
        return _ne


    def _apply(is_closer):
        _eta = float(emb_lr_slider.value)
        _pi_raw = int(emb_pair_idx())
        _pi = _pi_raw % len(PAIR_QUEUE)
        _wa, _wb, _ = PAIR_QUEUE[_pi]
        _ia, _ib = WORD_INDEX[_wa], WORD_INDEX[_wb]
        _e_curr = emb()
        # Remember which pair was just trained (shown at medium opacity after advance)
        set_prev_pair_indices((_ia, _ib))
        _ne = _step(_e_curr, _pi_raw, _eta, is_closer)
        set_emb(_ne)
        _c = int(emb_click_count()) + 1
        set_emb_click_count(_c)
        set_emb_pair_idx(_pi_raw + 1)
        _hist = list(emb_accuracy_history())
        _hist.append((_c, evaluate_pair_accuracy(_ne)))
        set_emb_accuracy_history(_hist)


    def _on_closer(_v=None):
        _apply(True)


    def _on_farther(_v=None):
        _apply(False)


    def _on_auto_train(_v=None):
        """Run 20 steps automatically, always choosing the correct direction."""
        _eta = float(emb_lr_slider.value)
        _e = emb()
        _pi_raw = int(emb_pair_idx())
        _c = int(emb_click_count())
        _hist = list(emb_accuracy_history())
        for _ in range(50):
            _pi = _pi_raw % len(PAIR_QUEUE)
            _, _, _is_pos = PAIR_QUEUE[_pi]
            _e = _step(_e, _pi_raw, _eta, _is_pos)
            _pi_raw += 1
            _c += 1
            _hist.append((_c, evaluate_pair_accuracy(_e)))
        set_emb(_e)
        set_emb_click_count(_c)
        set_emb_pair_idx(_pi_raw)
        set_emb_accuracy_history(_hist)


    def _on_randomize(_v=None):
        set_emb(_np_emb.random.randn(_V, 2).astype(_np_emb.float32) * 0.1)
        set_prev_pair_indices(None)
        set_emb_click_count(0)
        set_emb_pair_idx(0)
        set_emb_accuracy_history([])


    closer_btn = mo.ui.button(label="Closer ↔", on_click=_on_closer, kind="warn")
    farther_btn = mo.ui.button(label="Farther ↔", on_click=_on_farther, kind="danger")
    auto_train_btn = mo.ui.button(
        label="Train 50 steps",
        on_click=_on_auto_train,
        kind="success",
        tooltip="Run 50 correct updates automatically.",
    )
    randomize_emb_btn = mo.ui.button(label="Randomize", on_click=_on_randomize, kind="neutral")
    return (
        auto_train_btn,
        closer_btn,
        emb,
        emb_click_count,
        emb_lr_slider,
        emb_pair_idx,
        farther_btn,
        prev_pair_indices,
        randomize_emb_btn,
    )


@app.cell(hide_code=True)
def _(
    CATEGORIES,
    CATEGORY_COLORS,
    PAIR_QUEUE,
    WORD_INDEX,
    auto_train_btn,
    closer_btn,
    emb,
    emb_click_count,
    emb_lr_slider,
    emb_pair_idx,
    farther_btn,
    mo,
    prev_pair_indices,
    randomize_emb_btn,
):
    import plotly.graph_objects as _go

    _e = emb()
    _pidx = int(emb_pair_idx()) % len(PAIR_QUEUE)
    _pair_a, _pair_b, _is_pos = PAIR_QUEUE[_pidx]
    _ia = WORD_INDEX[_pair_a]
    _ib = WORD_INDEX[_pair_b]

    # Rings always follow the current emb positions of the focused pair.
    # The 1-second timer delay controls *which* pair gets the ring (pair index),
    # not where the ring sits — so rings move with their points immediately.
    _ring_x = [float(_e[_ia, 0]), float(_e[_ib, 0])]
    _ring_y = [float(_e[_ia, 1]), float(_e[_ib, 1])]

    _focused = {_ia, _ib}
    _prev = prev_pair_indices()
    _prev_set = set(_prev) if _prev is not None else set()
    _traces = []
    for _cat, _words in CATEGORIES.items():
        _xs = [float(_e[WORD_INDEX[w], 0]) for w in _words]
        _ys = [float(_e[WORD_INDEX[w], 1]) for w in _words]
        _ops = [1.0 if WORD_INDEX[w] in _focused else 0.75 if WORD_INDEX[w] in _prev_set else 0.4 for w in _words]
        _tcols = [
            CATEGORY_COLORS[_cat]
            if WORD_INDEX[w] in _focused
            else CATEGORY_COLORS[_cat]
            if WORD_INDEX[w] in _prev_set
            else "rgba(120,120,120,0.3)"
            for w in _words
        ]
        _traces.append(
            _go.Scatter(
                x=_xs,
                y=_ys,
                mode="markers+text",
                name=_cat,
                text=_words,
                textposition="top right",
                marker=dict(color=CATEGORY_COLORS[_cat], size=12, opacity=_ops),
                textfont=dict(color=_tcols),
            )
        )

    # Dashed line connecting the ring positions
    _traces.append(
        _go.Scatter(
            x=_ring_x,
            y=_ring_y,
            mode="lines",
            showlegend=False,
            line=dict(color="#555", dash="dash", width=1.5),
        )
    )
    # Ring highlights at the resolved positions
    _traces.append(
        _go.Scatter(
            x=_ring_x,
            y=_ring_y,
            mode="markers",
            showlegend=False,
            marker=dict(color="rgba(0,0,0,0)", size=22, line=dict(color="#333", width=2.5)),
            hovertemplate="%{text}<extra>current pair</extra>",
            text=[_pair_a, _pair_b],
        )
    )

    # Viewport centred on actual data bounding box, with 30% margin.
    import numpy as _np_emb_ui

    _xc = float((_e[:, 0].min() + _e[:, 0].max()) / 2)
    _yc = float((_e[:, 1].min() + _e[:, 1].max()) / 2)
    _half = (
        float(
            max(
                (_e[:, 0].max() - _e[:, 0].min()) / 2,
                (_e[:, 1].max() - _e[:, 1].min()) / 2,
                0.3,
            )
        )
        * 1.3
    )

    _fig_emb = _go.Figure(data=_traces)
    _fig_emb.update_layout(
        xaxis=dict(
            title="Dimension 1",
            range=[_xc - _half, _xc + _half],
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        yaxis=dict(
            title="Dimension 2",
            range=[_yc - _half, _yc + _half],
            scaleanchor="x",
            scaleratio=1,
            showgrid=False,
            zeroline=False,
            showticklabels=False,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=480,
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        transition={"duration": 400, "easing": "cubic-in-out"},
    )

    _ptype = "related ✓ (same category)" if _is_pos else "unrelated ✗ (different category)"
    _pair_md = mo.md(
        f"**Current pair:** *{_pair_a}* & *{_pair_b}* — {_ptype}  \n"
        f"Click **Closer** to pull together, **Farther** to push apart, "
        f"or **Train 20 steps** to run automatically."
    )
    _status = mo.md(
        f"Clicks: **{int(emb_click_count())}** · "
        f"Pair {_pidx + 1}/{len(PAIR_QUEUE)} · η = **{float(emb_lr_slider.value):g}**"
    )
    _btns = mo.hstack(
        [randomize_emb_btn, closer_btn, farther_btn, auto_train_btn],
        justify="start",
        gap=1,
    )
    mo.vstack([emb_lr_slider, _pair_md, _fig_emb, _btns, _status], gap=1.25)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### From Hand-labelled Pairs to Self-supervised Learning

    So far we labelled every pair by hand. For a vocabulary of 50,000 words that would require billions of labels. This is not clearly feasible.

    Let's step back and think about what defines the meaning of a word, for example "bird." You might say that a "bird" is an animal that can fly, with feathers. But some birds don't have feathers (like baby chickens), others can't fly (like penguins), and a bat flies without feathers yet is not a bird.

    Words derive their meaning not from intrinsic properties but from their relationships to other words. "Bird" means what it means because it contrasts with "fish," "insect," and "mammal," and "hot" only means what it means because "cold" exists.

    This leads to the idea of **distributional hypothesis**. Linguist J.R. Firth argued that a word is characterized by the company it keeps. Words "greese" and "swans" both appear near "pond" and "wings," while neither appears near "fins" or "gills."

    Now, going back to the problem of generating training data for word embedding models. Based on the distributional hypothesis, we consider that a word is defined by the other words it appears with. Operationally, we slide a fixed window of say ±2 words over every word, and all words inside the window form **positive pairs** while randomly sampled words as **negative pairs**. A model trained to distinguish the two is implicitly learning meaning from raw text.

    This is the core idea behind **Word2Vec**. Meaning is never labelled directly but emerges from the structure of language itself.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.center(
        mo.image(
            src="./figs/sliding-window-word-embedding.png",
            alt="Generating the data for self-suprvised training",
            width="70%",
            rounded=True,
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Query and Key — Two Roles of a Word

    Look at the sliding window again. In the sentence *"the quick brown fox jumps over the lazy dog"* with window size ±2, when the window is centred on **"fox"**, "fox" is the *focal word* and {"brown", "quick", "jumps", "over"} are *context words*. But when we slide the window to **"jumps"**, "jumps" becomes focal and "fox" becomes context.

    The same word plays **two different roles** depending on where the window sits:

    | Role | Also called | Example |
    |------|-------------|---------|
    | **Focal** (centre of the window) | *query* | "fox" when the window is centred on "fox" |
    | **Context** (neighbour) | *key* | "fox" when the window is centred on "jumps" |

    Because these roles are fundamentally different, we give each its own neural network (i.e., its own weight matrix):

    $$W_Q \in \mathbb{R}^{K \times V} \quad \text{(query matrix — for the focal role)}$$
    $$W_K \in \mathbb{R}^{K \times V} \quad \text{(key matrix — for the context role)}$$

    where $V$ is the vocabulary size and $K$ is the embedding dimension. Each matrix maps a one-hot vector $\mathbf{x}_i \in \{0,1\}^V$ to a $K$-dimensional vector:

    $$\mathbf{q}_i = W_Q\,\mathbf{x}_i \qquad \text{(query vector of word } i\text{)}$$
    $$\mathbf{k}_i = W_K\,\mathbf{x}_i \qquad \text{(key vector of word } i\text{)}$$

    Every word now has **two** embedding vectors — one for each role. The weights are **not shared**; they are learned independently.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training — From Dot Products to Gradient Descent

    ### 1. Alignment score

    Given a focal word $i$ and a context word $j$ from the sliding window, we measure how well they "fit" together with the dot product of their embeddings:

    $$s = \mathbf{q}_i \cdot \mathbf{k}_j$$

    A large positive $s$ means the model thinks $j$ is a likely context for $i$; a large negative $s$ means unlikely.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    dot_angle_slider = mo.ui.slider(
        start=0, stop=360, step=1, value=45, label="Angle of **k** (degrees)"
    )
    dot_len_q_slider = mo.ui.slider(
        start=0.1, stop=2.0, step=0.05, value=1.0, label="Length of **q**"
    )
    dot_len_k_slider = mo.ui.slider(
        start=0.1, stop=2.0, step=0.05, value=1.0, label="Length of **k**"
    )
    return dot_angle_slider, dot_len_k_slider, dot_len_q_slider


@app.cell(hide_code=True)
def _(dot_angle_slider, dot_len_k_slider, dot_len_q_slider, mo):
    import numpy as _np_dot
    import plotly.graph_objects as _go_dot
    from plotly.subplots import make_subplots as _make_subplots

    _angle_deg = dot_angle_slider.value
    _len_q = dot_len_q_slider.value
    _len_k = dot_len_k_slider.value

    # q is fixed along x-axis
    _qx, _qy = _len_q, 0.0
    _angle_rad = _np_dot.radians(_angle_deg)
    _kx = _len_k * _np_dot.cos(_angle_rad)
    _ky = _len_k * _np_dot.sin(_angle_rad)

    _dot = _qx * _kx + _qy * _ky  # q · k

    # --- Build figure with subplots: vectors on left, bar on right ---
    _fig = _make_subplots(
        rows=1, cols=2, column_widths=[0.75, 0.25],
        horizontal_spacing=0.08,
        subplot_titles=("Vectors", "Dot product"),
    )

    # Arrow helper: line + arrowhead annotation
    def _add_arrow(fig, x, y, color, name, row, col):
        fig.add_trace(
            _go_dot.Scatter(
                x=[0, x], y=[0, y], mode="lines",
                line=dict(color=color, width=3), name=name,
                showlegend=True,
            ),
            row=row, col=col,
        )
        fig.add_annotation(
            x=x, y=y, ax=0, ay=0,
            xref=f"x{'' if col == 1 else col}", yref=f"y{'' if col == 1 else col}",
            axref=f"x{'' if col == 1 else col}", ayref=f"y{'' if col == 1 else col}",
            showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=3,
            arrowcolor=color,
        )

    _add_arrow(_fig, _qx, _qy, "#1f77b4", "q (query)", 1, 1)
    _add_arrow(_fig, _kx, _ky, "#d62728", "k (key)", 1, 1)

    # Draw arc showing angle
    _arc_r = min(_len_q, _len_k, 0.4)
    _arc_t = _np_dot.linspace(0, _angle_rad, 40)
    _fig.add_trace(
        _go_dot.Scatter(
            x=(_arc_r * _np_dot.cos(_arc_t)).tolist(),
            y=(_arc_r * _np_dot.sin(_arc_t)).tolist(),
            mode="lines", line=dict(color="#888", width=1.5, dash="dot"),
            showlegend=False,
        ),
        row=1, col=1,
    )

    # Angle label
    _mid_angle = _angle_rad / 2
    _fig.add_annotation(
        x=float((_arc_r + 0.15) * _np_dot.cos(_mid_angle)),
        y=float((_arc_r + 0.15) * _np_dot.sin(_mid_angle)),
        text=f"{_angle_deg}°", showarrow=False,
        font=dict(size=12, color="#555"),
        xref="x", yref="y",
    )

    # Dot product bar
    _bar_color = "#2ca02c" if _dot >= 0 else "#d62728"
    _fig.add_trace(
        _go_dot.Bar(
            x=["q · k"], y=[_dot],
            marker_color=_bar_color, showlegend=False,
            text=[f"{_dot:.2f}"], textposition="outside",
            textfont=dict(size=16, color=_bar_color),
        ),
        row=1, col=2,
    )

    # Layout
    _lim = max(_len_q, _len_k) + 0.3
    _bar_lim = max(abs(_dot) + 0.5, 2.5)
    _fig.update_xaxes(range=[-_lim, _lim], zeroline=True, scaleanchor="y", row=1, col=1)
    _fig.update_yaxes(range=[-_lim, _lim], zeroline=True, row=1, col=1)
    _fig.update_xaxes(showticklabels=False, row=1, col=2)
    _fig.update_yaxes(range=[-_bar_lim, _bar_lim], zeroline=True, row=1, col=2)

    _fig.update_layout(
        height=400, margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(x=0.01, y=0.99),
        plot_bgcolor="white",
    )

    mo.vstack([
        mo.hstack([dot_angle_slider, dot_len_q_slider, dot_len_k_slider], justify="center"),
        _fig,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2. The problem: turning a score into a probability

    The score $s$ can be any real number $(-\infty, +\infty)$, but we need a probability $\hat{y} \in (0,1)$: *"how likely is this pair to be a real co-occurrence?"*. We also need the function to be smooth (differentiable) so we can use gradient descent.

    ### 3. Logistic (sigmoid) function

    The **sigmoid function** does exactly this:

    $$\sigma(s) = \frac{1}{1 + e^{-s}}$$

    You can think of the sigmoid as a **smooth version of a step function**: a step function snaps from 0 to 1 at $s=0$, but the sigmoid makes that transition gradual and differentiable — which is exactly what gradient descent needs.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    import numpy as _np_sig
    import plotly.graph_objects as _go_sig

    _s = _np_sig.linspace(-8, 8, 300)
    _sigmoid = 1 / (1 + _np_sig.exp(-_s))
    _step = (_s >= 0).astype(float)

    _fig_sig = _go_sig.Figure()

    # Step function
    _fig_sig.add_trace(_go_sig.Scatter(
        x=_s.tolist(), y=_step.tolist(),
        mode="lines", name="Step function",
        line=dict(color="#aaa", width=2, dash="dash"),
    ))

    # Sigmoid
    _fig_sig.add_trace(_go_sig.Scatter(
        x=_s.tolist(), y=_sigmoid.tolist(),
        mode="lines", name="σ(s) = 1 / (1 + e⁻ˢ)",
        line=dict(color="#1f77b4", width=3),
    ))

    # Reference lines
    _fig_sig.add_hline(y=0.5, line_dash="dot", line_color="#ccc", line_width=1)
    _fig_sig.add_vline(x=0, line_dash="dot", line_color="#ccc", line_width=1)

    # Annotations
    _fig_sig.add_annotation(x=6, y=0.95, text="→ 1 (likely pair)", showarrow=False,
                            font=dict(size=12, color="#2ca02c"))
    _fig_sig.add_annotation(x=-6, y=0.05, text="→ 0 (unlikely pair)", showarrow=False,
                            font=dict(size=12, color="#d62728"))
    _fig_sig.add_annotation(x=0.8, y=0.55, text="σ(0) = 0.5", showarrow=False,
                            font=dict(size=11, color="#555"))

    _fig_sig.update_layout(
        xaxis_title="s = q · k",
        yaxis_title="σ(s)",
        height=350,
        margin=dict(l=50, r=30, t=30, b=50),
        legend=dict(x=0.02, y=0.98),
        plot_bgcolor="white",
        yaxis=dict(range=[-0.08, 1.12]),
    )

    mo.center(_fig_sig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We interpret $\hat{y} = \sigma(s) = \sigma(\mathbf{q}_i \cdot \mathbf{k}_j)$ as the predicted probability that the pair $(i, j)$ is a genuine co-occurrence.

    ### 4. Loss function (binary cross-entropy)

    Let $y = 1$ for a real (positive) pair and $y = 0$ for a fake (negative) pair. The loss for one pair is:

    $$L = -\bigl[\,y \ln \sigma(s) \;+\; (1-y) \ln\bigl(1-\sigma(s)\bigr)\,\bigr]$$

    This penalises the model when its prediction $\sigma(s)$ is far from the true label $y$.

    ### 5. Gradient derivation (step by step)

    We need $\frac{\partial L}{\partial \mathbf{q}_i}$ and $\frac{\partial L}{\partial \mathbf{k}_j}$ to update the embeddings.

    **Step 1 — Derivative of the sigmoid:**
    $$\sigma'(s) = \sigma(s)\bigl(1-\sigma(s)\bigr)$$

    **Step 2 — Derivative of the loss w.r.t. the score $s$:**
    $$\frac{\partial L}{\partial s} = \sigma(s) - y$$

    (This clean result follows from combining the log derivatives with the sigmoid identity above.)

    **Step 3 — Derivative of the score w.r.t. the embeddings:**
    $$\frac{\partial s}{\partial \mathbf{q}_i} = \mathbf{k}_j, \qquad \frac{\partial s}{\partial \mathbf{k}_j} = \mathbf{q}_i$$

    **Step 4 — Chain rule:**
    $$\frac{\partial L}{\partial \mathbf{q}_i} = \bigl(\sigma(s)-y\bigr)\,\mathbf{k}_j, \qquad \frac{\partial L}{\partial \mathbf{k}_j} = \bigl(\sigma(s)-y\bigr)\,\mathbf{q}_i$$

    ### 6. Update rule (gradient descent)

    With learning rate $\eta$:

    $$\mathbf{q}_i \;\leftarrow\; \mathbf{q}_i \;-\; \eta\bigl(\sigma(s)-y\bigr)\,\mathbf{k}_j$$
    $$\mathbf{k}_j \;\leftarrow\; \mathbf{k}_j \;-\; \eta\bigl(\sigma(s)-y\bigr)\,\mathbf{q}_i$$

    **Intuition:**
    - For a **positive pair** ($y=1$): $\sigma(s)-1 < 0$, so both vectors are nudged *toward* each other — increasing their dot product.
    - For a **negative pair** ($y=0$): $\sigma(s)-0 > 0$, so both vectors are pushed *apart* — decreasing their dot product.

    ### 7. Scaling to real corpora — Word2Vec

    This is exactly the **skip-gram with negative sampling** algorithm (Mikolov et al., 2013). Applied to billions of $(focal, context)$ pairs from a large text corpus, with $K = 300$ dimensions, it produces embeddings where geometry encodes meaning:

    $$\mathbf{q}_{\text{king}} - \mathbf{q}_{\text{man}} + \mathbf{q}_{\text{woman}} \approx \mathbf{q}_{\text{queen}}$$

    No human labels — just co-occurrence statistics and gradient descent.
    """)
    return


if __name__ == "__main__":
    app.run()
