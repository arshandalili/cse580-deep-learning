# HW1 - Memorization and Generalization

In this assignment we train a small convolutional network and a multilayer perceptron on
CIFAR-10, and then on versions of CIFAR-10 whose structure has been removed, in order to see
which properties of a model are responsible for its ability to generalize.

## Setup

### Local

Install [uv](https://docs.astral.sh/uv/getting-started/installation/):

    curl -LsSf https://astral.sh/uv/install.sh | sh

Then, from this directory:

    uv sync                      # creates .venv/ and installs the dependencies
    uv run jupyter lab hw1.ipynb

### Colab

Open `hw1.ipynb` in [Colab](https://colab.research.google.com/), select a GPU under
`Runtime > Change runtime type` (any GPU available would work as the training is not heavy), and
run the cells in order. PyTorch, torchvision and matplotlib are already installed there, and
Kaggle works in the same way. Note that the virtual machine is erased when the session ends, so
download `results/` and the executed notebook before you close it.

## Files

    hw1.ipynb      the assignment - fill in every TODO
    data/          CIFAR-10, downloaded automatically on first use (~170 MB)
    results/       one JSON per training run, written automatically

## Notes

- Runs are cached by configuration under `results/`; delete a file there to
  force a re-run.
- Do the parts in order. Parts 4-5 depend on the functions you write in Parts 1-3.

## Submit

`hw1.ipynb` with all cells executed.
