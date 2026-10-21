#!/usr/bin/env bash
# Put the swegemma harness packages (swegemma, adk_eval_core, adk_submission) into DEST, from SRC:
# a folder with the harness wheels (scripts/fetch_harness.py), the competition's harness zips and/or
# the unpacked package folders, in any layout.
# Then: export PYTHONPATH=DEST
#   bash scripts/setup_harness.sh harness_src harness
set -euo pipefail
SRC=${1:?source folder}; DEST=${2:?destination folder}
TMP=$(mktemp -d)
mkdir -p "$DEST"
# wheels are zip files with the package at their root
find "$SRC" \( -name '*.zip' -o -name '*.whl' \) -print0 | while IFS= read -r -d '' z; do
  unzip -qo "$z" -d "$TMP/$(basename "$z")"
done
cp -r "$SRC" "$TMP/src"
# each package is recognised by a file inside it, so zips whose root IS the package work whatever their name
for spec in swegemma:harness/agent_runner.py adk_eval_core:tracing/trace.py adk_submission:compiler.py; do
  pkg=${spec%%:*}; marker=${spec#*:}
  dir=$(find "$TMP" -path "*/$marker" | sed "s#/$marker\$##" | while read -r d; do [ -f "$d/__init__.py" ] && echo "$d"; done \
        | awk '{ print length, $0 }' | sort -n | head -1 | cut -d' ' -f2-)
  [ -n "$dir" ] || { echo "package $pkg not found in $SRC" >&2; exit 1; }
  rm -rf "${DEST:?}/$pkg" && cp -r "$dir" "$DEST/$pkg"
  echo "$pkg <- ${dir#"$TMP"/}"
done
rm -rf "$TMP"
PYTHONPATH="$DEST" python -c "import swegemma.harness.agent_runner, adk_eval_core, adk_submission; print('harness OK')"
