# HW1 - Rethinking Generalization

Reproduce the core experiments of
*Understanding Deep Learning Requires Rethinking Generalization*, Zhang et al., ICLR 2017
([arXiv:1611.03530](https://arxiv.org/abs/1611.03530)). Read Sections 1-4 before you start.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/):

    curl -LsSf https://astral.sh/uv/install.sh | sh

Then, from this directory:

    uv sync                      # creates .venv/ and installs the dependencies
    uv run jupyter lab hw1.ipynb

`uv sync` reads `pyproject.toml`; you do not need to activate the venv yourself when you launch
through `uv run`. If you prefer your own Jupyter, activate `.venv` and select it as the kernel.

If `torch.cuda.is_available()` prints `False` but you do have an NVIDIA GPU, install the build
that matches your driver, e.g.

    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

## Files

    hw1.ipynb      the assignment - fill in every TODO, answer every Q
    hw1_utils.py   data loading, augmentation and plotting helpers (do not modify)
    data/          CIFAR-10, downloaded automatically on first use (~170 MB)
    results/       one JSON per training run, written automatically

## Notes

- The notebook is 24 short training runs, ~15 minutes total on a recent GPU and proportionally
  more on an older one. Runs are cached by configuration under `results/`; delete a file there to
  force a re-run.
- On CPU this is several hours. If you have no GPU, use Colab.
- Do the parts in order. Parts 4-6 depend on the functions you write in Parts 1-3.

## Submit

`hw1.ipynb` with all cells executed and all questions answered, together with `results/`.
