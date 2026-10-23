#!/bin/sh
# omp-hf installer: installs omp (if missing) and the omp-hf extension.
# Usage: curl -fsSL https://raw.githubusercontent.com/aiguy110/omp-hf/main/install.sh | sh
#
# Environment:
#   OMP_HF_REPO   GitHub repo to install from (default aiguy110/omp-hf)
#   OMP_HF_REF    Branch, tag, or commit (default main)
set -eu

REPO="${OMP_HF_REPO:-aiguy110/omp-hf}"
REF="${OMP_HF_REF:-main}"

say() { printf '\033[1m==>\033[0m %s\n' "$*"; }
die() { printf 'omp-hf: %s\n' "$*" >&2; exit 1; }

for cmd in curl tar; do
    command -v "$cmd" >/dev/null 2>&1 || die "$cmd is required"
done
case "$(uname -s)" in
    Linux | Darwin) ;;
    *) die "only Linux and macOS are supported" ;;
esac

# omp's installer puts the binary in ~/.local/bin (or PI_INSTALL_DIR); bun installs land in ~/.bun/bin.
ORIG_PATH="$PATH"
PATH="${PI_INSTALL_DIR:-$HOME/.local/bin}:$HOME/.bun/bin:$PATH"
export PATH

if command -v omp >/dev/null 2>&1; then
    say "omp already installed: $(omp --version 2>/dev/null || echo unknown)"
else
    say "Installing omp"
    curl -fsSL https://omp.sh/install | sh
    command -v omp >/dev/null 2>&1 || die "omp install finished but omp is not on PATH"
fi

AGENT_DIR="${PI_CODING_AGENT_DIR:-$HOME/.omp/agent}"
DEST="$AGENT_DIR/extensions/omp-hf"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

say "Installing omp-hf ($REPO@$REF) into $DEST"
curl -fsSL "https://codeload.github.com/$REPO/tar.gz/$REF" | tar -xz -C "$TMP" --strip-components=1
[ -f "$TMP/index.ts" ] || die "$REPO@$REF does not look like omp-hf"
rm -rf "$DEST"
mkdir -p "$DEST"
cp -R "$TMP/index.ts" "$TMP/package.json" "$TMP/src" "$DEST/"

say "Done. Start omp and run /hf to pick a model, or jump straight in:"
echo "    omp --model hf/bonsai-2-4b"
if ! PATH="$ORIG_PATH" command -v omp >/dev/null 2>&1; then
    echo "    (omp is in $(dirname "$(command -v omp)"); open a new shell or add it to PATH)"
fi
