#!/usr/bin/env bash
set -euo pipefail

export SHELL_RC="$HOME/.bashrc"

# shellcheck source=install.sh
source "$SHELL_RC"

# -------- PYENV --------
if ! command -v pyenv >/dev/null 2>&1; then
    echo "Installing pyenv..."
    curl -fsSL https://pyenv.run | bash

    # add init to shell rc
    if [ -n "${ZSH_VERSION:-}" ]; then
        SHELL_RC="$HOME/.zshrc"
    else
        SHELL_RC="$HOME/.bashrc"
    fi

    if ! grep -q 'export PATH="$HOME/.pyenv/bin' "$SHELL_RC"; then
        {
            echo ''
            echo '# Pyenv setup'
            echo 'export PATH="$HOME/.pyenv/bin:$PATH"'
            echo 'eval "$(pyenv init -)"'
            echo 'eval "$(pyenv virtualenv-init -)"'
        } >> "$SHELL_RC"
    fi

    echo "pyenv installed."
else
    echo "pyenv already installed."
fi


# -------- PIPX --------
if ! command -v pipx >/dev/null 2>&1; then
    echo "Installing pipx..."
    sudo apt-get update -y
    sudo apt-get install -y pipx
    pipx ensurepath
else
    echo "pipx already installed."
fi


# -------- POETRY --------
if ! command -v poetry >/dev/null 2>&1; then
    echo "Installing poetry..."
    pipx install poetry
else
    echo "poetry already installed."
fi

# shellcheck source=install.sh
source "$SHELL_RC"

# -------- PROJECT INSTALL --------
if [ ! -d ".venv" ]; then
    echo "Installing project dependencies via poetry..."
    poetry install
else
    echo "Project dependencies already installed."
fi
