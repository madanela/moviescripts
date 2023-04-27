FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Yerevan

RUN apt-get update
# Install necessary dependencies for pyenv and pyenv-virtualenv
RUN apt-get update && \
    apt-get install -y \
    git \
    curl \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    llvm \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev

# Install pyenv and pyenv-virtualenv
ENV HOME /root
RUN git clone https://github.com/pyenv/pyenv.git $HOME/.pyenv
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN git clone https://github.com/pyenv/pyenv-virtualenv.git $PYENV_ROOT/plugins/pyenv-virtualenv
RUN echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
RUN echo 'eval "$(pyenv virtualenv-init -)"' >> $HOME/.bashrc

# RUN git clone https://github.com/pyenv/pyenv-virtualenv.git ~/.pyenv/plugins/pyenv-virtualenv

CMD ["/bin/bash"]


# Set up the virtual environment
WORKDIR /app
RUN pyenv install 3.9.13
RUN pyenv virtualenv 3.9.13 moviescripts
RUN echo "moviescripts" > .python-version
ENV PATH /root/.pyenv/versions/moviescripts/bin:$PATH

# Install Poetry
RUN pip install poetry
ENV PATH $HOME/.poetry/bin:$PATH

# Copy project files
COPY pyproject.toml poetry.lock ./
RUN poetry install
COPY . .

CMD [ "poetry", "run", "train" ]
