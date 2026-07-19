FROM continuumio/miniconda3

LABEL author="Filip Ramazan"
LABEL version="v1.0"
LABEL description="BWTandem — BWT-based tandem repeat finder"

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PYTHONPATH=/opt/bwtandem

# Install system dependencies (gcc for C extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential procps \
    && rm -rf /var/lib/apt/lists/*

# Configure conda channels
RUN conda config --add channels defaults && \
    conda config --add channels bioconda && \
    conda config --add channels conda-forge

# Install Python dependencies
RUN conda install -y \
    python=3.13 numpy cython numba setuptools pip \
    && conda clean -a -y

# Install pydivsufsort
RUN pip install pydivsufsort --no-build-isolation

# Copy BWTandem source
COPY . /opt/bwtandem/

# Pre-compile C extensions (required for read-only filesystems like Singularity)
RUN cd /opt/bwtandem && python -m src.c_extensions.build

# Compile Cython extension (if present)
RUN cd /opt/bwtandem && \
    if [ -f src/_accelerators.pyx ]; then \
        python -c "from setuptools import setup, Extension; from Cython.Build import cythonize; import numpy as np; ext_modules = [Extension('src._accelerators', ['src/_accelerators.pyx'], include_dirs=[np.get_include()], extra_compile_args=['-std=c99'])]; setup(script_args=['build_ext', '--inplace'], ext_modules=cythonize(ext_modules, compiler_directives={'language_level': '3'}))"; \
    fi

ENTRYPOINT ["python3", "-m", "src.main"]
