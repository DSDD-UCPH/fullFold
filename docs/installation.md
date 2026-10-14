# Installing fullFold from scratch

## TL;DR

If AlphaFold 3 (DeepMind's `alphafold3>=3.0.2`) is already working in your Python 3.12+ environment, this is all you need:

```bash
pip install fullfold
```

fullFold is a thin layer on top of an existing AlphaFold 3 install. It does **not** install AlphaFold 3 for you, and it does not download the model weights. The rest of this page covers the case where you do not have a working AlphaFold 3 yet.

fullFold is in active development, so update often:

```bash
# update to the newest release
pip install --upgrade fullfold

# check what you have
pip show fullfold
python -c "import fullFold; print(fullFold.__version__)"

# pin a specific release (e.g. to reproduce a run)
pip install fullfold==0.1.3

# bleeding edge (the main branch on GitHub)
pip install --upgrade git+https://github.com/DSDD-UCPH/fullFold.git
```

Updating fullFold never touches your AlphaFold 3 install. Inside a `uv` environment, prefix with `uv`: `uv pip install --upgrade fullfold`.

---

## Pick a route

| Route | Best for |
| --- | --- |
| [A. Conda / pip](#a-conda--pip) | Most users; you already use conda. |
| [B. uv](#b-uv) | Fast, reproducible, uses DeepMind's lockfile. |
| [C. Docker / Singularity](#c-docker--singularity) | Clusters and shared machines where the container is the unit of deployment. |

Routes A and B both install AlphaFold 3 **v3.0.4** from source. This is the release the default-mode hoists in fullFold target (see the README), and it compiles a small C++ extension, which is why you need a compiler.

### Build prerequisites (routes A and B)

On Debian/Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y build-essential git zlib1g-dev
```

Without root, you can get the same tools inside a conda environment instead:

```bash
conda install -c conda-forge cxx-compiler make git zlib -y
```

The build also downloads a few C++ dependencies from GitHub (abseil, pybind11, libcifpp, dssp), so run it on a machine with internet access. The environment you build in can then be used anywhere that shares the filesystem, including offline GPU nodes.

---

## A. Conda / pip

```bash
# 1. An empty environment. pip comes with it.
#    (No extra channels needed: JAX gets its CUDA libraries from pip wheels.)
conda create -n fullfold python=3.14 -c conda-forge -y
conda activate fullfold

# 2. AlphaFold 3 v3.0.4 from source
git clone --branch v3.0.4 --depth 1 https://github.com/google-deepmind/alphafold3.git
cd alphafold3
pip install -e .

# 3. Build the chemical components data (once)
build_data

# 4. fullFold
pip install fullfold
```

## B. uv

```bash
# 0. Install uv (once)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 1. AlphaFold 3 v3.0.4
git clone --branch v3.0.4 --depth 1 https://github.com/google-deepmind/alphafold3.git
cd alphafold3

# 2. Create .venv with the locked dependencies
uv sync --python 3.14

# 3. Build the chemical components data (once)
uv run build_data

# 4. fullFold, into the same .venv
uv pip install fullfold
```

Use it by activating the environment (or prefixing with `uv run`):

```bash
source /path/to/alphafold3/.venv/bin/activate
fullfold --help
```

Two uv-specific gotchas:

- `uv pip install fullfold` is not recorded in the lockfile. A later plain `uv sync` removes packages it does not know about, including fullFold. Use `uv sync --inexact` from then on, or just re-run `uv pip install fullfold`.
- `uv sync` installs AlphaFold 3 in editable mode, so `run_alphafold.py` stays next to the package. That is what af3-faster needs.

## C. Docker / Singularity

Follow DeepMind's [installation guide](https://github.com/google-deepmind/alphafold3/blob/main/docs/installation.md) for Docker, the NVIDIA container toolkit and (if needed) Singularity. The only change is **one extra line at the very end of `docker/Dockerfile`**, after the `build_data` step:

```dockerfile
ARG FULLFOLD_VERSION=0.1.3
RUN uv pip install --python /alphafold3_venv/bin/python "fullfold==${FULLFOLD_VERSION}"
```

`/alphafold3_venv` is the virtual environment that DeepMind's Dockerfile creates, and its `bin/` is already on `PATH`, so `fullfold` works as a command in the container. Pinning the version through a build argument means that to update you only need to rebuild with a new value, which re-runs just this layer instead of the whole image (HMMER and all):

```bash
git clone --branch v3.0.4 --depth 1 https://github.com/google-deepmind/alphafold3.git
cd alphafold3
# add the two lines above to docker/Dockerfile, then:
docker build -t alphafold3 -f docker/Dockerfile .

# later, to move to a newer fullFold:
docker build --build-arg FULLFOLD_VERSION=<new version> -t alphafold3 -f docker/Dockerfile .
```

Run it (note the extra volume, which keeps fullFold's benchmark and JAX compilation caches between runs; inside a container they would otherwise be lost on exit):

```bash
docker run -it --gpus all \
    --volume $HOME/af_input:/root/af_input \
    --volume $HOME/af_output:/root/af_output \
    --volume <MODEL_PARAMETERS_DIR>:/root/models \
    --volume $HOME/.cache/fullFold:/root/.cache/fullFold \
    alphafold3 \
    fullfold run --input-dir /root/af_input --output-dir /root/af_output --model-dir /root/models
```

For Singularity, build the `.sif` from the Docker image as described in DeepMind's guide (it will contain fullFold), then:

```bash
singularity exec --nv \
    --bind $HOME/af_input:/root/af_input \
    --bind $HOME/af_output:/root/af_output \
    --bind <MODEL_PARAMETERS_DIR>:/root/models \
    alphafold3.sif \
    fullfold run --input-dir /root/af_input --output-dir /root/af_output --model-dir /root/models
```

Singularity mounts your home directory by default, so the caches persist without extra flags.

---

## Model weights

AlphaFold 3 weights are released by Google DeepMind on request, under their [terms of use](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md). Follow the request process in the [AlphaFold 3 README](https://github.com/google-deepmind/alphafold3#obtaining-model-parameters), then put the file in a directory of your choice:

```
/path/to/models/af3.bin.zst
```

and pass that directory with `--model-dir /path/to/models`.

## Optional: af3-faster

[af3-faster](https://github.com/DSDD-UCPH/af3-faster) adds faster kernels from Anthropic (see af3-faster README) on top of an existing AlphaFold 3 install. If it is installed, fullFold switches to `--mode fast` automatically. It is not published on PyPI; follow the install section in its README after finishing the steps above. It expects the AlphaFold 3 checkout to be editable (route A and B do this by default).