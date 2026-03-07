#!/bin/sh
# setup.sh - Development environment setup for rtp-engine
#
# Usage: ./setup.sh

set -e

echo "=== rtp-engine development setup ==="
echo ""

# Detect OS
OS="$(uname -s)"

# 1. Check Rust toolchain
echo "[1/5] Checking Rust toolchain..."
if ! command -v rustup >/dev/null 2>&1; then
    echo "  rustup not found. Installing..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    . "$HOME/.cargo/env"
fi

RUST_VERSION=$(rustc --version 2>/dev/null || echo "none")
echo "  Rust: $RUST_VERSION"

# Ensure minimum version (1.85 required for edition 2024)
rustup update stable --no-self-update 2>/dev/null || rustup update stable

# Install components needed for CI checks
echo "  Installing rustfmt and clippy..."
rustup component add rustfmt clippy

# 2. Install system dependencies
echo ""
echo "[2/5] Installing system dependencies..."
case "$OS" in
    Linux)
        if command -v apt-get >/dev/null 2>&1; then
            echo "  Detected Debian/Ubuntu"
            sudo apt-get update -qq
            sudo apt-get install -y libasound2-dev libopus-dev pkg-config
        elif command -v dnf >/dev/null 2>&1; then
            echo "  Detected Fedora/RHEL"
            sudo dnf install -y alsa-lib-devel opus-devel pkg-config
        elif command -v pacman >/dev/null 2>&1; then
            echo "  Detected Arch Linux"
            sudo pacman -S --noconfirm alsa-lib opus pkg-config
        else
            echo "  WARNING: Unknown Linux distro. Please install ALSA and Opus dev libraries manually."
        fi
        ;;
    Darwin)
        if command -v brew >/dev/null 2>&1; then
            echo "  Installing opus via Homebrew..."
            brew install opus pkg-config 2>/dev/null || brew upgrade opus pkg-config 2>/dev/null || true
        else
            echo "  WARNING: Homebrew not found. Please install opus manually."
        fi
        ;;
    *)
        echo "  WARNING: Unsupported OS '$OS'. Please install dependencies manually."
        ;;
esac

# 3. Configure git hooks
echo ""
echo "[3/5] Configuring git hooks..."
git config core.hooksPath .githooks
echo "  Git hooks path set to .githooks/"
echo "  Pre-commit hook will run: cargo fmt --check, cargo clippy"

# 4. Build the project
echo ""
echo "[4/5] Building project..."
cargo build --all-features

# 5. Run checks
echo ""
echo "[5/5] Running checks..."
echo "  Checking formatting..."
cargo fmt --all -- --check

echo "  Running clippy..."
cargo clippy --all-features --all-targets

echo "  Running tests..."
cargo test --all-features

echo ""
echo "=== Setup complete ==="
echo ""
echo "You're ready to develop! The pre-commit hook will automatically"
echo "run 'cargo fmt --check' and 'cargo clippy' before each commit."
