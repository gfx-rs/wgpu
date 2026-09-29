#!/usr/bin/env bash
set -euo pipefail

# The inputs come in through environment variables. See action.yml.
REPOSITORY="${AICC_REPOSITORY:-}"
PULL_REQUEST="${AICC_PULL_REQUEST:-}"
EXPECTED_COUNT="${AICC_EXPECTED_COUNT:-}"
FAIL_ON_DETECTION="${AICC_FAIL_ON_DETECTION:-true}"
PATTERN="${AICC_PATTERN:-}"

if [ -z "$REPOSITORY" ] || [ -z "$PULL_REQUEST" ]; then
    echo "::error::This action needs a pull request. Run it on a 'pull_request' event, or set the 'repository' and 'pull-request' inputs."
    exit 1
fi

# The default detection pattern flags common AI metadata signatures.
# These are based on https://botcommits.dev, the documentation of each AI platform,
# and some other miscellaneous references. Feel free to add more or fix any false positives!
if [ -z "$PATTERN" ]; then
    PATTERN='(Co-authored-by|Signed-off-by|Authorized-by):.*(Claude|Copilot|Cursor|Codex|ChatGPT|GPT-|Gemini|Jules|Devin|Aider)'
    PATTERN="$PATTERN"'|Generated with (\[)?(Claude Code|Cursor|Copilot)|🤖 Generated'
    PATTERN="$PATTERN"'|claude\.ai/code|claude\.com/claude-code'
    PATTERN="$PATTERN"'|noreply@anthropic\.com|noreply@openai\.com|github-copilot\[bot\]'
    PATTERN="$PATTERN"'|cursoragent|@cursor\.com|devin-ai-integration|@devin\.ai|google-labs-jules|[0-9]+\+Copilot@users\.noreply\.github\.com|\[aider\]|\(aider\)'
fi

echo "Reading the commits of $REPOSITORY#$PULL_REQUEST..."

# The API supplies the commit metadata, so the workflow does not need the git
# history. A shallow checkout is sufficient.
#
# Each commit becomes one line that holds the SHA and a base64 record. The
# record holds the author name, the author email, the committer name, the
# committer email, and the full message, separated by newlines.
commits=$(gh api --paginate "repos/$REPOSITORY/pulls/$PULL_REQUEST/commits" --jq '
    .[] | "\(.sha) \(
        [
            .commit.author.name,
            .commit.author.email,
            .commit.committer.name,
            .commit.committer.email,
            .commit.message
        ] | join("\n") | @base64
    )"
')

read_count=0
if [ -n "$commits" ]; then
    read_count=$(printf '%s\n' "$commits" | wc -l)
fi

# The API lists at most 250 commits. Fail on a partial list instead of passing.
if [ -n "$EXPECTED_COUNT" ] && [ "$read_count" -lt "$EXPECTED_COUNT" ]; then
    echo "::error::Read $read_count of the $EXPECTED_COUNT commits in $REPOSITORY#$PULL_REQUEST. The pull request is too long for the API to list. Check the remaining commits by hand."
    exit 1
fi

echo "Evaluating $read_count commit(s)."

failed=0
count=0
offending=()

while IFS=' ' read -r sha record; do
    [ -z "$sha" ] && continue

    text=$(printf '%s' "$record" | base64 -d)

    if printf '%s' "$text" | grep -Eiq "$PATTERN"; then
        echo "::error::Commit $sha appears to be AI-authored or contains AI trailers."
        echo "--------------------------------------------------"
        echo "  Commit: ${sha:0:9}"
        echo "  Author: $(printf '%s\n' "$text" | sed -n '1p') <$(printf '%s\n' "$text" | sed -n '2p')>"
        echo "  Committer: $(printf '%s\n' "$text" | sed -n '3p') <$(printf '%s\n' "$text" | sed -n '4p')>"
        echo "  Message:"
        printf '%s\n' "$text" | tail -n +5 | sed 's/^/    /'
        echo "--------------------------------------------------"
        failed=1
        count=$((count + 1))
        offending+=("$sha")
    fi
done <<<"$commits"

# Emit step outputs so callers can react to the result.
if [ -n "${GITHUB_OUTPUT:-}" ]; then
    if [ "$failed" -ne 0 ]; then
        echo "ai-commits-found=true" >>"$GITHUB_OUTPUT"
    else
        echo "ai-commits-found=false" >>"$GITHUB_OUTPUT"
    fi
    echo "count=$count" >>"$GITHUB_OUTPUT"
    {
        echo "commits<<__AICC_EOF__"
        if [ "${#offending[@]}" -gt 0 ]; then
            printf '%s\n' "${offending[@]}"
        fi
        echo "__AICC_EOF__"
    } >>"$GITHUB_OUTPUT"
fi

if [ "$failed" -ne 0 ]; then
    echo "Detected $count AI-authored commit(s) in $REPOSITORY#$PULL_REQUEST."
    if [ "$FAIL_ON_DETECTION" = "true" ]; then
        echo "Error: Attribution of commits to AIs is not allowed. Please remove the attribution."
        exit 1
    fi
    echo "fail-on-detection is disabled; reporting through the 'ai-commits-found' output instead of failing."
    exit 0
fi

echo "Success: No AI-authored commits found in $REPOSITORY#$PULL_REQUEST."
