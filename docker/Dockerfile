FROM homebrew/brew:4.0.14

WORKDIR /app

RUN brew install pyenv pyenv-virtualenv

RUN sudo apt update
RUN sudo apt install -y \
    build-essential \
    software-properties-common \
    python3-pip \
    curl \
    libbz2-dev \
    libffi-dev \
    liblzma-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    libxml2-dev \
    libxmlsec1-dev \
    llvm \
    make \
    tk-dev \
    wget \
    xz-utils \
    zlib1g-dev



RUN brew install poetry
COPY pyproject.toml poetry.lock ./
RUN poetry install

COPY . .
CMD [ "poetry", "run", "train" ]
