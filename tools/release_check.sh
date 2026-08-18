#!/bin/bash
# Release gate check.
#
# Verifies that a given version is ready to be tagged and released by
# checking version declarations, the test suite, working tree cleanliness,
# whether HEAD has been pushed, and whether the release tag is still free.
#
# Usage: tools/release_check.sh <version>
# Example: tools/release_check.sh 1.0.0
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT" || exit 1

if [ "$#" -ne 1 ]; then
    echo "Usage: tools/release_check.sh <version>" >&2
    echo "Example: tools/release_check.sh 1.0.0" >&2
    exit 2
fi

VERSION="$1"
TAG="v${VERSION}"

TOTAL_GATES=5
PASS_COUNT=0

pass() {
    echo "PASS: $1"
    PASS_COUNT=$((PASS_COUNT + 1))
}

fail() {
    echo "FAIL: $1"
}

# Extracts the version declared in the [project] table of pyproject.toml.
extract_pyproject_version() {
    awk '
        /^\[project\]/ { in_section = 1; next }
        /^\[/ { in_section = 0 }
        in_section && /^version[ \t]*=/ {
            line = $0
            sub(/^version[ \t]*=[ \t]*"/, "", line)
            sub(/".*/, "", line)
            print line
            exit
        }
    ' pyproject.toml
}

# Extracts the __version__ assignment from eeane/__init__.py.
extract_init_version() {
    awk '
        /^__version__[ \t]*=/ {
            line = $0
            sub(/^__version__[ \t]*=[ \t]*"/, "", line)
            sub(/".*/, "", line)
            print line
            exit
        }
    ' eeane/__init__.py
}

# Extracts the version of the "eeane" package entry from uv.lock.
extract_uv_lock_version() {
    awk '
        BEGIN { found = 0 }
        /^\[\[package\]\]/ {
            if (!found && name == "eeane" && ver != "") { print ver; found = 1; exit }
            name = ""
            ver = ""
            next
        }
        /^name[ \t]*=/ {
            if (name == "") {
                line = $0
                sub(/^name[ \t]*=[ \t]*"/, "", line)
                sub(/".*/, "", line)
                name = line
            }
            next
        }
        /^version[ \t]*=/ {
            if (ver == "") {
                line = $0
                sub(/^version[ \t]*=[ \t]*"/, "", line)
                sub(/".*/, "", line)
                ver = line
            }
            next
        }
        END {
            # exit above also triggers this block; the "found" guard
            # prevents printing the matched version a second time.
            if (!found && name == "eeane") print ver
        }
    ' uv.lock
}

echo "== Release check for version ${VERSION} (tag ${TAG}) =="
echo

# ---- G1: version declarations ----
PYPROJECT_VERSION="$(extract_pyproject_version)"
INIT_VERSION="$(extract_init_version)"
LOCK_VERSION="$(extract_uv_lock_version)"

g1_ok=true
[ "$PYPROJECT_VERSION" = "$VERSION" ] || g1_ok=false
[ "$INIT_VERSION" = "$VERSION" ] || g1_ok=false
[ "$LOCK_VERSION" = "$VERSION" ] || g1_ok=false

if [ "$g1_ok" = true ]; then
    pass "G1 version declarations match ${VERSION}"
else
    fail "G1 version declarations do not all match ${VERSION}"
    echo "  pyproject.toml [project].version = ${PYPROJECT_VERSION:-<not found>}"
    echo "  eeane/__init__.py __version__    = ${INIT_VERSION:-<not found>}"
    echo "  uv.lock eeane package version    = ${LOCK_VERSION:-<not found>}"
fi
echo

# ---- G2: test suite ----
echo "Running test suite (tools/check.sh)..."
if ./tools/check.sh; then
    pass "G2 test suite (tools/check.sh)"
else
    fail "G2 test suite (tools/check.sh)"
fi
echo

# ---- G3: clean tree ----
STATUS_OUTPUT="$(git status --porcelain)"
if [ -z "$STATUS_OUTPUT" ]; then
    pass "G3 clean working tree"
else
    fail "G3 clean working tree (uncommitted or untracked changes present)"
    echo "$STATUS_OUTPUT" | sed 's/^/  /'
fi
echo

# ---- G4: pushed ----
if UPSTREAM="$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null)"; then
    if git merge-base --is-ancestor HEAD '@{u}' 2>/dev/null; then
        pass "G4 pushed (HEAD is reachable from ${UPSTREAM})"
    else
        fail "G4 pushed (HEAD is ahead of ${UPSTREAM}; push required)"
    fi
else
    fail "G4 pushed (no upstream tracking branch configured for the current branch)"
fi
echo

# ---- G5: tag absent ----
LOCAL_TAG_EXISTS=false
if git rev-parse -q --verify "refs/tags/${TAG}" >/dev/null 2>&1; then
    LOCAL_TAG_EXISTS=true
fi

REMOTE_CHECK_OK=false
REMOTE_TAG_EXISTS=false
if REMOTE_TAGS="$(git ls-remote --tags origin 2>/dev/null)"; then
    REMOTE_CHECK_OK=true
    if printf '%s\n' "$REMOTE_TAGS" | awk '{ print $2 }' | grep -Fxq "refs/tags/${TAG}"; then
        REMOTE_TAG_EXISTS=true
    fi
fi

if [ "$REMOTE_CHECK_OK" = false ]; then
    fail "G5 tag absent (could not query tags on origin; check network/remote access)"
elif [ "$LOCAL_TAG_EXISTS" = true ] || [ "$REMOTE_TAG_EXISTS" = true ]; then
    where=""
    [ "$LOCAL_TAG_EXISTS" = true ] && where="local"
    if [ "$REMOTE_TAG_EXISTS" = true ]; then
        if [ -n "$where" ]; then
            where="${where}, origin"
        else
            where="origin"
        fi
    fi
    fail "G5 tag absent (tag ${TAG} already exists: ${where})"
else
    pass "G5 tag absent (tag ${TAG} not found locally or on origin)"
fi
echo

echo "== release check: ${PASS_COUNT}/${TOTAL_GATES} gates passed =="
if [ "$PASS_COUNT" -eq "$TOTAL_GATES" ]; then
    echo "result: READY to release ${VERSION}"
    exit 0
else
    echo "result: NOT ready to release ${VERSION}"
    exit 1
fi
