#!/usr/bin/env bash
# Set up the WiLoR offline path. Reproduces the steps from docs/wilor.md so a fresh clone
# (or you on a new machine) doesn't have to re-derive the chumpy patch by hand.
#
# Tested on Python 3.13 + torch 2.x + numpy 2, where the upstream pins/deps don't install
# cleanly. Run from the repo root with your target Python/venv active.
#
# Cleaner alternative: a dedicated Python 3.10 + torch<=2.5 + numpy<2 conda/micromamba env
# installs everything natively with no patching (see docs/wilor.md).
#
# NOTE: you must also obtain MANO_RIGHT.pkl from https://mano.is.tue.mpg.de (register + accept
# the license) and place it at mano_data/MANO_RIGHT.pkl. It is license-gated and NOT
# redistributable, which is why it isn't in this repo — see docs/licensing.md.
set -euo pipefail

PY="${PYTHON:-python3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "==> 1/3  runtime deps"
"$PY" -m pip install -r "$HERE/requirements-wilor.txt"

echo "==> 2/3  wilor-mini (--no-deps: its torch<=2.5 pin conflicts with modern torch)"
"$PY" -m pip install --no-deps "git+https://github.com/warmshao/WiLoR-mini.git"

echo "==> 3/3  chumpy (patched to build on Python 3.11+ / numpy 2)"
"$PY" -c "import chumpy" 2>/dev/null && { echo "    chumpy already importable, skipping"; } || {
  TMP="$(mktemp -d)"
  URL="$(curl -s https://pypi.org/pypi/chumpy/json | "$PY" -c "import json,sys;print([u['url'] for u in json.load(sys.stdin)['urls'] if u['url'].endswith('.tar.gz')][-1])")"
  curl -sL -o "$TMP/chumpy.tar.gz" "$URL"
  tar xzf "$TMP/chumpy.tar.gz" -C "$TMP"
  SRC="$(ls -d "$TMP"/chumpy-*/ | head -1)"
  "$PY" - "$SRC" <<'PYEOF'
import pathlib, sys
src = pathlib.Path(sys.argv[1])
init = src / "chumpy" / "__init__.py"
init.write_text(init.read_text().replace(
    "from numpy import bool, int, float, complex, object, unicode, str, nan, inf",
    "from numpy import nan, inf\n"
    "bool = bool; int = int; float = float; complex = complex; object = object; str = str; unicode = str"))
ch = src / "chumpy" / "ch.py"
ch.write_text(ch.read_text().replace("inspect.getargspec", "inspect.getfullargspec"))
print("    patched", src)
PYEOF
  "$PY" -m pip install six
  "$PY" -m pip install --no-build-isolation "$SRC"
}

echo "==> verifying..."
"$PY" -c "import wilor_mini, chumpy; print('OK: wilor_mini + chumpy import cleanly')"
echo "==> done. Don't forget MANO_RIGHT.pkl in mano_data/ (see docs/licensing.md)."
