from setuptools import setup, find_packages

# Packages that require a non-PyPI index (NVIDIA RAPIDS) or are GPU-only
# accelerators. They must NOT land in install_requires, or `pip install castle-ai`
# from PyPI fails outright (cuml-cu12 is not on PyPI) — CASTLE already falls back
# to sklearn/umap-learn on CPU when these are absent. Install them explicitly with
# `pip install castle-ai[gpu]` (plus the NVIDIA extra-index, see requirements.txt).
GPU_ONLY = {"cuml-cu12", "xformers"}


def readme():
    with open("README.md", encoding="UTF-8") as readme_file:
        return readme_file.read()


def _pkg_name(line: str) -> str:
    """Bare distribution name from a requirement line (strip version/markers)."""
    for sep in (">=", "<=", "==", "~=", ">", "<", "!=", ";", "["):
        idx = line.find(sep)
        if idx != -1:
            line = line[:idx]
    return line.strip().lower()


def read_requirements():
    """Parse requirements.txt into (core, gpu_only).

    Comments and pip flag lines (``--extra-index-url`` …) are ignored. Packages
    in GPU_ONLY are split out so the default install stays CPU-installable from
    PyPI; they are re-exposed via the ``gpu`` extra.
    """
    core, gpu = [], []
    try:
        with open("requirements.txt", encoding="UTF-8") as req_file:
            for line in req_file:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("-"):
                    continue
                (gpu if _pkg_name(line) in GPU_ONLY else core).append(line)
    except IOError:
        pass
    return core, gpu


_core, _gpu = read_requirements()

configuration = {
    "name": "castle-ai",
    "version": "0.0.18",
    "description": "CASTLE — a training-free foundation-model pipeline for "
                   "unsupervised, cross-species behavioral classification",
    "long_description": readme(),
    "long_description_content_type": "text/markdown",
    "url": "https://github.com/CASTLE-ai/castle-ai",
    "project_urls": {
        "Documentation": "https://castle-ai.github.io/castle-ai/",
        "Source": "https://github.com/CASTLE-ai/castle-ai",
        "Paper": "https://www.biorxiv.org/content/10.1101/2025.08.22.671685v2",
    },
    "classifiers": [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    "maintainer": "Raiso Liu, IsonaEi",
    "maintainer_email": "rainsoon717@gmail.com",
    "license": "Apache-2.0",
    "packages": find_packages(),
    "include_package_data": True,
    "package_data": {
        # Ship vendored third-party license notices in the wheel
        # (DeAOT = BSD-3-Clause, SAM = Apache-2.0).
        "castle.aot": ["LICENSE", "MODEL_ZOO.md", "README.md"],
        "castle.sam": ["LICENSE"],
    },
    "python_requires": ">=3.10",
    "install_requires": _core,
    "extras_require": {
        # GPU acceleration (RAPIDS cuML UMAP/DBSCAN + xformers attention).
        # Needs the NVIDIA extra-index — see requirements.txt.
        "gpu": _gpu,
        # `castle ethogram export-nwb`
        "nwb": ["pynwb>=2.5"],
        # HDBSCAN clustering backend
        "hdbscan": ["hdbscan>=0.8.33"],
        # Test / lint / type-check tooling used by CASTLE's QC gate
        "dev": ["pytest>=7.0", "ruff>=0.5", "mypy>=1.10"],
    },
    "entry_points": {
        "console_scripts": [
            "castle=castle.cli.main:app",
        ],
    },
    "zip_safe": False,
}

setup(**configuration)
