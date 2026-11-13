#!/bin/sh
# Prints a short-lived publish token for one index, minted from this job's
# GitHub OIDC token (GPU-412):
#
#     TOKEN=$(sh .github/scripts/trusted-token.sh pypi)        # or crates.io
#     echo "::add-mask::$TOKEN"
#
# Trusted publishing in place of a token stored as a secret. A stored token
# lives until someone revokes it, and anyone who can make a workflow print it
# can publish from anywhere; this one is issued only to a job that GitHub
# vouches for - this repository, a workflow and an environment the index has
# been told to trust - and expires on its own: PyPI's in fifteen minutes,
# crates.io's at the end of the job or earlier, when the caller revokes it.
#
# Minted right before the upload that uses it, never at the start of a job:
# a wheel can take longer to build than PyPI's token lives.
#
# Plain curl, not the indexes' own actions: an action's tag can be moved, and
# this step holds what publishes a version that can never be taken back. The
# job needs `permissions: id-token: write`; without it GitHub hands out no
# OIDC token and the first check below says so.
set -eu

index=${1:?usage: trusted-token.sh pypi|crates.io}
case "$index" in
  pypi) url=https://pypi.org/_/oidc/mint-token; key=token ;;
  crates.io) url=https://crates.io/api/v1/trusted_publishing/tokens; key=jwt ;;
  *) echo "unknown index: $index" >&2; exit 2 ;;
esac

if [ -z "${ACTIONS_ID_TOKEN_REQUEST_URL:-}" ]; then
  echo "No OIDC token for this job: it needs permissions: id-token: write" >&2
  echo "(and, when called from another workflow, the caller's job too)." >&2
  exit 1
fi

# The audience is the index's own name: PyPI expects "pypi", crates.io its
# host.
jwt=$(curl -sSf -H "Authorization: bearer $ACTIONS_ID_TOKEN_REQUEST_TOKEN" \
  "$ACTIONS_ID_TOKEN_REQUEST_URL&audience=$index" \
  | sed -n 's/.*"value" *: *"\([^"]*\)".*/\1/p')
if [ -z "$jwt" ]; then
  echo "GitHub returned no OIDC token" >&2
  exit 1
fi

body=$(curl -sS -X POST "$url" -H 'Content-Type: application/json' \
  -d "{\"$key\":\"$jwt\"}")
token=$(printf '%s' "$body" | sed -n 's/.*"token" *: *"\([^"]*\)".*/\1/p')
if [ -z "$token" ]; then
  # The refusal names what did not match (workflow, environment, repository);
  # it carries no secret, so it is printed whole.
  echo "$index did not issue a token. Its answer:" >&2
  printf '%s\n' "$body" >&2
  echo "Check this repository's trusted publisher on $index: workflow file" >&2
  echo "and environment must be the ones this job runs in." >&2
  exit 1
fi
printf '%s\n' "$token"
