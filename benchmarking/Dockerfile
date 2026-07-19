FROM continuumio/miniconda3

LABEL author="Filip Ramazan"
LABEL version="v1.0"
LABEL description="Benchmarking container for tandem repeat finders"

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PYTHONPATH=/opt/bwtandem

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git procps wget time pkg-config libicu-dev \
    && rm -rf /var/lib/apt/lists/*

# Configure conda channels
RUN conda config --add channels defaults && \
    conda config --add channels bioconda && \
    conda config --add channels conda-forge

# Install dependencies
RUN conda install -y \
    trf mreps tidehunter tantan ncrf \
    r-base r-stringr r-stringi r-stringdist r-optparse r-circlize r-seqinr r-doparallel r-remotes \
    bioconductor-biostrings bioconductor-pwalign \
    mafft \
    zlib gcc libcurl \
    cmake \
    python=3.13 numpy cython pytest numba setuptools pip && \
    conda clean -a -y

# Install pydivsufsort for BWTandem
RUN pip install pydivsufsort --no-build-isolation

# Build ULTRA from source
RUN git clone https://github.com/TravisWheelerLab/ULTRA /opt/ULTRA && \
    cd /opt/ULTRA && \
    cmake . && \
    make && \
    ln -s /opt/ULTRA/ultra /usr/local/bin/ultra

# Install TRASH
RUN git clone https://github.com/vlothec/TRASH /opt/TRASH && \
    cd /opt/TRASH && \
    chmod +x TRASH_install.sh TRASH_run.sh && \
    ./TRASH_install.sh --def

ENV PATH=/opt/conda/bin:/opt/TRASH:/usr/local/bin:${PATH}

# Copy BWTandem into the image
COPY . /opt/bwtandem/

# Pre-compile BWTandem C extensions (Singularity images are read-only at runtime)
RUN cd /opt/bwtandem && python -m src.c_extensions.build

# Compile BWTandem Cython extension (if present)
RUN cd /opt/bwtandem && \
    if [ -f src/_accelerators.pyx ]; then \
    python -c "from setuptools import setup, Extension; from Cython.Build import cythonize; import numpy as np; ext_modules = [Extension('src._accelerators', ['src/_accelerators.pyx'], include_dirs=[np.get_include()], extra_compile_args=['-std=c99'])]; setup(script_args=['build_ext', '--inplace'], ext_modules=cythonize(ext_modules, compiler_directives={'language_level': '3'}))"; \
    fi

CMD ["/bin/bash"]
