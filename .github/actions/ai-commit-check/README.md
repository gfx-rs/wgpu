# AI Commit Check

A composite action that detects AI-authored commits and AI metadata trailers in a pull request.

wgpu allows contributors to write code with an LLM, but the contributor must own the
change. See the [LLMs (AI)](../../../CONTRIBUTING.md#llms-ai) section of `CONTRIBUTING.md`.
This action rejects commits that attribute authorship to an AI instead.

This is a vendored version of [`Jondolf/ai-commit-check`](https://github.com/Jondolf/ai-commit-check)
from commit `7069d4254c6310cdb0395a6ab0ca1f29af897f83` with its data collection changed to
be done using the `gh` CLI instead of raw git history, to avoid needing a deep checkout.

## Usage

```yaml
- name: Check for AI-authored commits
  if: github.event_name == 'pull_request'
  uses: ./.github/actions/ai-commit-check
```

## Inputs

| Input               | Default              | Description                                                                     |
| ------------------- | -------------------- | ------------------------------------------------------------------------------- |
| `repository`        | current repository   | The `owner/name` of the repository that holds the pull request.                 |
| `pull-request`      | triggering PR number | The number of the pull request to check.                                        |
| `expected-count`    | PR commit count      | The number of commits the pull request contains, used to detect a short list.   |
| `fail-on-detection` | `true`               | When `true`, exit non-zero on detection. Set to `false` to report outputs only. |
| `pattern`           | built-in             | The regular expression that flags AI metadata.                                  |
| `token`             | workflow token       | The token that reads the commits of the pull request.                           |

## Outputs

| Output             | Description                                                   |
| ------------------ | ------------------------------------------------------------- |
| `ai-commits-found` | `"true"` if the action detected AI-authored commits.          |
| `count`            | The number of commits that the action flagged as AI-authored. |
| `commits`          | The flagged commit SHAs, separated by newlines.               |

## Attribution

Licensed under the MIT License. See [LICENSE.MIT](LICENSE.MIT).
