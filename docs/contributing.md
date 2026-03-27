# Contributing

## Commit messages

This repository enforces Conventional Commits locally through `pre-commit` and in automation through semantic-release.

Accepted commit types:

- `feat`
- `fix`
- `refactor`
- `perf`
- `test`
- `build`
- `ci`
- `docs`
- `style`
- `chore`
- `revert`

Examples:

- `feat: add semantic release workflow`
- `fix(cache): skip poisoned restore keys`
- `refactor!: simplify slot allocation lifecycle`

Breaking changes should use `!` in the header or a `BREAKING CHANGE:` footer.

## Local setup

```bash
pip install -e .[dev]
pre-commit install
pre-commit install --hook-type commit-msg
```

The `commit-msg` hook runs `cz check --commit-msg-file` to reject non-conventional commit messages before the commit is created.

## Pull requests

PR titles are validated in GitHub Actions and must also follow the same Conventional Commit format.

Examples:

- `feat: add ghcr container publishing`
- `fix(ci): correct release workflow token handling`

If you use squash merges, enable "Default to PR title for squash merge commits" in the repository settings so the merge commit stays compatible with semantic-release.
