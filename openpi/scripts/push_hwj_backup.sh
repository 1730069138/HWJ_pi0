#!/usr/bin/env bash

set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

expected_origin="git@github.com:1730069138/HWJ_pi0.git"
actual_origin="$(git remote get-url origin)"
if [[ "$actual_origin" != "$expected_origin" ]]; then
    echo "Refusing to push: origin is '$actual_origin', expected '$expected_origin'." >&2
    exit 1
fi

source_commit="$(git rev-parse HEAD)"
source_tree="$(git rev-parse 'HEAD^{tree}')"
backup_ref="refs/heads/hwj-main"

if previous_backup="$(git rev-parse --verify "$backup_ref" 2>/dev/null)"; then
    readme_entry="$(git ls-tree "$previous_backup" README.md)"
    wrapped_tree="$(printf '%s\n040000 tree %s\topenpi\n' "$readme_entry" "$source_tree" | sed '/^$/d' | git mktree)"
    backup_commit="$(printf 'Backup openpi at %s\n' "$source_commit" | git commit-tree "$wrapped_tree" -p "$previous_backup")"
    git update-ref "$backup_ref" "$backup_commit" "$previous_backup"
else
    wrapped_tree="$(printf '040000 tree %s\topenpi\n' "$source_tree" | git mktree)"
    backup_commit="$(printf 'Initial backup of openpi at %s\n' "$source_commit" | git commit-tree "$wrapped_tree")"
    git update-ref "$backup_ref" "$backup_commit"
fi

git push origin "$backup_ref:refs/heads/main"

echo "Pushed committed source $source_commit as openpi/ to HWJ_pi0."
if [[ -n "$(git status --porcelain)" ]]; then
    echo "Note: uncommitted changes were not included; commit them before the next git hwj-push."
fi
