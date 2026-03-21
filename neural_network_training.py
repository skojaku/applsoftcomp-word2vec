# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "marimo>=0.9.0",
#   "numpy>=1.24",
#   "matplotlib>=3.7",
#   "Pillow>=10.0",
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
    # Training a tiny neural network: cat vs dog **faces** (AFHQ)
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
    ## What we want to do

    We will run a toy Rosenblatt's neural network that classifies an image into a cat or dog face.

    ## How we do it

    The images are **grayscale**, represented as a marix of pixel values. For each pixel, we multiply a weight, one weight per pixel. We then compute the output by summing up the weighted pixel values:

    \[
    \text{output} = \mathbf{w}^\top \mathbf{x} + b.
    \]

    where b is the offset. Here \(\mathbf{x}\) is the grayscale face flattened (pixels in about \([0,1]\)). Think of this as a weighted average of pixel values.

    We consider that an image is a dog if output>0, otherwise cat.

    ## Training

    We want to find the weights \(\mathbf{w}\) that correctly classify all training images. We will do this by looking at the first training image that the model gets wrong, then clicking **Add** or **Subtract** to fix it.

    More specifically, when the classifier makes incorrect classification, the weights are updated by:

    \[

    w \leftarrow w \pm \eta x
    \]

    where $\eta$ is the learning rate, controlling the update size. $\pm$ depends on whether the misclassification is made for a dog face (should be +) or a cat face (should be -).

    ## Try yourself

    Let's learn how it works. We will give you misclassified images. You click whether to add to or subtract the pixel values from the weights. Click *Add (+)* if a dog face is misclassified. Click *Subtract* if a cat face.

    **Data:** faces are from **[AFHQ](https://github.com/clovaai/stargan-v2)** (prepared via
    [Hugging Face `huggan/AFHQ`](https://huggingface.co/datasets/huggan/AFHQ)). A helper script resizes to
    a square grid (default **\(64\times64\)**), converts to **grayscale**, and saves **PNG files** under
    `data/afhq_catdog/` that this notebook loads directly — no dataset download when you open Marimo.
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
        _ax_im.imshow(_np.clip(img, 0.0, 1.0), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        _ax_im.set_title("Face (first mistake in train order)")
        _ax_im.axis("off")
        _wm = _ax_wm.imshow(W_mat, cmap="RdBu_r", vmin=-_lim_side, vmax=_lim_side, interpolation="nearest")
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
        _ax.plot(xs, ys, marker="o", color="#1976d2", label="Accuracy (train = eval)")
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
                    kind="success" if current_acc >= 0.55 else "info",
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
    mo.callout(
        mo.md(
            f"Loaded **{len(train_images)}** faces from `{_path_show}` "
            f"({_grid}×{_grid} gray, **{_n_w}** weights). "
            f"*Demo:* training and the accuracy plot use **this same set**. "
            f"*Sharper grid:* `uv run scripts/prepare_afhq_subset.py --grid 96` then reload."
        ),
        kind="success",
    )
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


# ── Part II: Word Embeddings ────────────────────────────────────────────────


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # Part II — From Pixels to Words

    ## One-hot encoding

    A vocabulary of $V$ words can be represented as vectors where word $i$ is a binary vector
    $\mathbf{x}_i \in \{0,1\}^V$ with a 1 only at position $i$ (all others 0).

    When we feed $\mathbf{x}_i$ into the perceptron:

    \[
    \text{output} = \mathbf{w}^\top \mathbf{x}_i = w_i
    \]

    The one-hot vector simply **selects the $i$-th weight**. So the weight $w_i$ is exactly
    the "score" the model assigns to word $i$.

    For example, a sentiment classifier might learn $w_\text{happy} > 0$ (positive word)
    and $w_\text{sad} < 0$ (negative word). The output $> 0$ means positive sentiment.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## From One Output to Many — Word Embeddings

    Instead of a single output neuron, use **two** output neurons. The weight matrix is
    now $W \in \mathbb{R}^{V \times 2}$:

    \[
    \mathbf{o}_i = W^\top \mathbf{x}_i = \text{row}_i(W) \in \mathbb{R}^2
    \]

    Row $i$ of $W$ is the **2-dimensional embedding** $\mathbf{e}_i$ of word $i$ — a
    coordinate in 2D space.

    **Geometric interpretation:** words with similar meaning end up **close** to each other
    in this 2D space. Distance encodes semantic similarity. This is the core idea behind
    word2vec, GloVe, and modern language model embeddings.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Training with Pairs: Positives and Negatives

    We train by showing the model **pairs** of words and clicking whether they should be
    **closer** or **farther** in the embedding space.

    - **Positive pairs** (semantically related): *cat & dog*, *happy & calm* → click **Closer**
    - **Negative pairs** (unrelated, randomly drawn): *cat & car*, *happy & tree* → click **Farther**

    The update rule uses $\boldsymbol{\delta} = \mathbf{e}_j - \mathbf{e}_i$ computed from
    the **original** values before any update:

    | Click | Rule |
    |-------|------|
    | Closer | $\mathbf{e}_i \mathrel{+}= \eta\,\boldsymbol{\delta}$ ; $\mathbf{e}_j \mathrel{-}= \eta\,\boldsymbol{\delta}$ |
    | Farther | $\mathbf{e}_i \mathrel{-}= \eta\,\boldsymbol{\delta}$ ; $\mathbf{e}_j \mathrel{+}= \eta\,\boldsymbol{\delta}$ |

    Clicking **Closer** moves both embeddings toward their midpoint.
    Clicking **Farther** pushes them apart.
    """)
    return


@app.cell(hide_code=True)
def _():
    VOCAB = [
        "cat", "dog", "bird", "fish",
        "red", "blue",
        "happy", "sad", "angry", "calm", "joyful",
        "run", "walk", "jump", "sleep",
        "car", "tree", "house", "book",
    ]
    WORD_INDEX = {w: i for i, w in enumerate(VOCAB)}

    CATEGORIES = {
        "animal":  ["cat", "dog", "bird", "fish"],
        "color":   ["red", "blue"],
        "emotion": ["happy", "sad", "angry", "calm", "joyful"],
        "action":  ["run", "walk", "jump", "sleep"],
        "object":  ["car", "tree", "house", "book"],
    }
    CATEGORY_COLORS = {
        "animal":  "#e67e22",
        "color":   "#9b59b6",
        "emotion": "#e74c3c",
        "action":  "#27ae60",
        "object":  "#2980b9",
    }

    POS_PAIRS = [
        ("cat",   "dog"),
        ("cat",   "bird"),
        ("dog",   "fish"),
        ("happy", "calm"),
        ("happy", "joyful"),
        ("sad",   "angry"),
        ("run",   "walk"),
        ("run",   "jump"),
        ("walk",  "sleep"),
        ("red",   "blue"),
    ]
    NEG_PAIRS = [
        ("cat",   "car"),
        ("cat",   "happy"),
        ("dog",   "run"),
        ("dog",   "tree"),
        ("bird",  "house"),
        ("fish",  "book"),
        ("happy", "car"),
        ("sad",   "tree"),
        ("run",   "red"),
        ("walk",  "blue"),
        ("jump",  "house"),
        ("sleep", "book"),
    ]

    PAIR_QUEUE = [(a, b, True) for a, b in POS_PAIRS] + [(a, b, False) for a, b in NEG_PAIRS]

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
def _(VOCAB, WORD_INDEX, POS_PAIRS, NEG_PAIRS):
    import numpy as _np_sim


    def compute_similarity(emb, word_a, word_b) -> float:
        ea = emb[WORD_INDEX[word_a]].astype(_np_sim.float64)
        eb = emb[WORD_INDEX[word_b]].astype(_np_sim.float64)
        norm_a = float(_np_sim.linalg.norm(ea))
        norm_b = float(_np_sim.linalg.norm(eb))
        if norm_a < 1e-12 or norm_b < 1e-12:
            return 0.0
        return float(_np_sim.dot(ea, eb) / (norm_a * norm_b))


    def evaluate_pair_accuracy(emb):
        pos_correct = sum(
            1 for a, b in POS_PAIRS if compute_similarity(emb, a, b) > 0
        )
        neg_correct = sum(
            1 for a, b in NEG_PAIRS if compute_similarity(emb, a, b) < 0
        )
        pos_acc = pos_correct / len(POS_PAIRS) if POS_PAIRS else 0.0
        neg_acc = neg_correct / len(NEG_PAIRS) if NEG_PAIRS else 0.0
        overall = (pos_correct + neg_correct) / (len(POS_PAIRS) + len(NEG_PAIRS))
        return pos_acc, neg_acc, overall

    return compute_similarity, evaluate_pair_accuracy


@app.cell(hide_code=True)
def _(mo, VOCAB, WORD_INDEX, PAIR_QUEUE, evaluate_pair_accuracy):
    import numpy as _np_emb

    _V = len(VOCAB)

    emb, set_emb = mo.state(
        _np_emb.random.randn(_V, 2).astype(_np_emb.float32) * 0.1
    )
    emb_click_count, set_emb_click_count = mo.state(0)
    emb_pair_idx, set_emb_pair_idx = mo.state(0)
    emb_accuracy_history, set_emb_accuracy_history = mo.state([])

    def _apply_update(is_closer: bool) -> None:
        _eta = float(emb_lr_slider.value)
        _e0 = emb()
        _pidx = int(emb_pair_idx()) % len(PAIR_QUEUE)
        _word_a, _word_b, _ = PAIR_QUEUE[_pidx]
        _ia = WORD_INDEX[_word_a]
        _ib = WORD_INDEX[_word_b]
        _ea = _e0[_ia].astype(_np_emb.float64, copy=True)
        _eb = _e0[_ib].astype(_np_emb.float64, copy=True)
        _delta = _eb - _ea
        _new_e = _e0.astype(_np_emb.float32, copy=True)
        if is_closer:
            _new_e[_ia] = (_ea + _eta * _delta).astype(_np_emb.float32)
            _new_e[_ib] = (_eb - _eta * _delta).astype(_np_emb.float32)
        else:
            _new_e[_ia] = (_ea - _eta * _delta).astype(_np_emb.float32)
            _new_e[_ib] = (_eb + _eta * _delta).astype(_np_emb.float32)
        set_emb(_new_e)
        _c = int(emb_click_count()) + 1
        set_emb_click_count(_c)
        set_emb_pair_idx(_pidx + 1)
        _acc = evaluate_pair_accuracy(_new_e)
        _hist = list(emb_accuracy_history())
        _hist.append((_c, _acc))
        set_emb_accuracy_history(_hist)

    def _on_closer(_v=None):
        _apply_update(is_closer=True)

    def _on_farther(_v=None):
        _apply_update(is_closer=False)

    def _on_randomize_emb(_v=None):
        set_emb(_np_emb.random.randn(_V, 2).astype(_np_emb.float32) * 0.1)
        set_emb_click_count(0)
        set_emb_pair_idx(0)
        set_emb_accuracy_history([])

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
    closer_btn = mo.ui.button(
        label="Closer ↔",
        on_click=_on_closer,
        kind="success",
        tooltip="Move the two words closer in embedding space.",
    )
    farther_btn = mo.ui.button(
        label="Farther ↔",
        on_click=_on_farther,
        kind="danger",
        tooltip="Push the two words farther apart in embedding space.",
    )
    randomize_emb_btn = mo.ui.button(
        label="Randomize",
        on_click=_on_randomize_emb,
        kind="neutral",
        tooltip="Reset embeddings to random values.",
    )

    return (
        closer_btn,
        emb,
        emb_accuracy_history,
        emb_click_count,
        emb_lr_slider,
        emb_pair_idx,
        farther_btn,
        randomize_emb_btn,
    )


@app.cell(hide_code=True)
def _(
    mo,
    VOCAB,
    WORD_INDEX,
    CATEGORIES,
    CATEGORY_COLORS,
    PAIR_QUEUE,
    emb,
    emb_click_count,
    emb_pair_idx,
    closer_btn,
    farther_btn,
    randomize_emb_btn,
    emb_lr_slider,
):
    import io as _io_emb
    import matplotlib.pyplot as _plt_emb
    import numpy as _np_ui

    _e = emb()
    _pidx = int(emb_pair_idx()) % len(PAIR_QUEUE)
    _pair_a, _pair_b, _is_pos = PAIR_QUEUE[_pidx]
    _ia = WORD_INDEX[_pair_a]
    _ib = WORD_INDEX[_pair_b]

    _fig, _ax = _plt_emb.subplots(figsize=(7, 6))

    for _cat_name, _words in CATEGORIES.items():
        _xs = [float(_e[WORD_INDEX[w], 0]) for w in _words]
        _ys = [float(_e[WORD_INDEX[w], 1]) for w in _words]
        _ax.scatter(_xs, _ys, color=CATEGORY_COLORS[_cat_name], s=80, zorder=3, label=_cat_name)
        for _w, _x, _y in zip(_words, _xs, _ys):
            _ax.annotate(_w, (_x, _y), xytext=(6, 4), textcoords="offset points", fontsize=10)

    # Highlight current pair
    _ax.scatter(
        [float(_e[_ia, 0]), float(_e[_ib, 0])],
        [float(_e[_ia, 1]), float(_e[_ib, 1])],
        color="orange", s=180, zorder=4, edgecolors="black", linewidths=1.5,
    )
    _ax.plot(
        [float(_e[_ia, 0]), float(_e[_ib, 0])],
        [float(_e[_ia, 1]), float(_e[_ib, 1])],
        color="orange", linestyle="--", linewidth=1.5, zorder=3,
    )

    _ax.set_title("Word Embeddings (2D)")
    _ax.set_xlabel("Dimension 1")
    _ax.set_ylabel("Dimension 2")
    _ax.legend(loc="best", fontsize=10)
    _ax.grid(True, alpha=0.3)
    _plt_emb.tight_layout()

    _buf = _io_emb.BytesIO()
    _fig.savefig(_buf, format="png", dpi=110, bbox_inches="tight", facecolor="white", edgecolor="none")
    _plt_emb.close(_fig)
    _scatter_img = mo.image(_buf.getvalue(), width=640)

    _pair_type = "positive (related)" if _is_pos else "negative (unrelated)"
    _pair_md = mo.md(
        f"**Current pair:** *{_pair_a}* & *{_pair_b}* — {_pair_type}  \n"
        f"Click **Closer** if they should be similar, **Farther** if not."
    )
    _status_md = mo.md(
        f"Clicks: **{int(emb_click_count())}** · Pair {_pidx + 1}/{len(PAIR_QUEUE)} · η = **{float(emb_lr_slider.value):g}**"
    )
    _buttons = mo.hstack([randomize_emb_btn, closer_btn, farther_btn], justify="start", gap=1)

    mo.vstack(
        [
            mo.md("## Interactive Word Embedding Training"),
            emb_lr_slider,
            _pair_md,
            _scatter_img,
            _buttons,
            _status_md,
        ],
        gap=1.25,
    )
    return


@app.cell(hide_code=True)
def _(mo, emb_accuracy_history, emb, evaluate_pair_accuracy):
    import io as _io_chart
    import matplotlib.pyplot as _plt_chart

    _hist = emb_accuracy_history()
    _cur_pos, _cur_neg, _cur_all = evaluate_pair_accuracy(emb())

    if not _hist:
        _chart_panel = mo.vstack(
            [
                mo.md("## Pair Accuracy vs Training Clicks"),
                mo.callout(
                    mo.md(
                        f"**Current accuracy:** pos {_cur_pos*100:.1f}% · neg {_cur_neg*100:.1f}% · overall {_cur_all*100:.1f}%  \n"
                        "*No training clicks recorded yet — click **Closer** or **Farther** to start.*"
                    ),
                    kind="neutral",
                ),
            ]
        )
    else:
        _xs = [h[0] for h in _hist]
        _pos_ys = [h[1][0] * 100.0 for h in _hist]
        _neg_ys = [h[1][1] * 100.0 for h in _hist]
        _all_ys = [h[1][2] * 100.0 for h in _hist]

        _fig2, _ax2 = _plt_chart.subplots(figsize=(6.4, 3.4))
        _ax2.plot(_xs, _pos_ys, marker="o", color="#27ae60", label="Positive pair acc")
        _ax2.plot(_xs, _neg_ys, marker="s", color="#e74c3c", label="Negative pair acc")
        _ax2.plot(_xs, _all_ys, marker="^", color="#1976d2", label="Overall acc")
        _ax2.axhline(50.0, color="#9e9e9e", linestyle="--", linewidth=1.5, label="50% baseline")
        _ax2.set_xlabel("Training click")
        _ax2.set_ylabel("Accuracy (%)")
        _ax2.set_title("Word embedding pair accuracy")
        _ax2.set_ylim(0, 100)
        _ax2.grid(True, alpha=0.3)
        _ax2.legend(loc="best")
        _plt_chart.tight_layout()

        _buf2 = _io_chart.BytesIO()
        _fig2.savefig(_buf2, format="png", dpi=110, bbox_inches="tight", facecolor="white", edgecolor="none")
        _plt_chart.close(_fig2)

        _chart_panel = mo.vstack(
            [
                mo.md("## Pair Accuracy vs Training Clicks"),
                mo.callout(
                    mo.md(
                        f"**Current accuracy:** pos {_cur_pos*100:.1f}% · neg {_cur_neg*100:.1f}% · overall {_cur_all*100:.1f}%"
                    ),
                    kind="success" if _cur_all >= 0.55 else "info",
                ),
                mo.image(_buf2.getvalue(), width=640),
            ]
        )

    _chart_panel
    return


if __name__ == "__main__":
    app.run()
