#!/usr/bin/env bash
# Create a GitHub repository for this folder, commit everything, and push.
#
# Usage (from the folder you want to publish):
#   bash publish_repo.sh                      # name: workshop, public
#   bash publish_repo.sh my-repo-name         # custom name
#   VISIBILITY=private bash publish_repo.sh   # private instead of public
#
# Requires: git and the GitHub CLI (gh), already logged in (gh auth login).

set -euo pipefail

NAME="${1:-workshop}"
VISIBILITY="${VISIBILITY:-public}"          # public or private
BRANCH="main"
MESSAGE="${COMMIT_MESSAGE:-Add essays, scripts and figures}"

# --- checks -----------------------------------------------------------------
command -v git >/dev/null || { echo "git is not installed."; exit 1; }
command -v gh  >/dev/null || { echo "GitHub CLI (gh) is not installed: https://cli.github.com"; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "Not logged in. Run: gh auth login"; exit 1; }

if [ ! -f .gitignore ]; then
  echo "Warning: no .gitignore here. LaTeX build files and outputs/ would be committed."
  read -r -p "Continue anyway? [y/N] " ans
  [[ "$ans" =~ ^[Yy]$ ]] || exit 1
fi

# --- local repository -------------------------------------------------------
if [ ! -d .git ]; then
  git init -b "$BRANCH"
fi

git add -A
echo
echo "Files to be committed:"
git status --short | head -60
COUNT=$(git status --short | wc -l)
[ "$COUNT" -gt 60 ] && echo "... ($COUNT changes in total)"
echo

read -r -p "Create '$NAME' ($VISIBILITY) on GitHub and push these? [y/N] " ans
[[ "$ans" =~ ^[Yy]$ ]] || { echo "Cancelled. Nothing was pushed."; exit 0; }

if git diff --cached --quiet; then
  echo "Nothing new to commit."
else
  git commit -m "$MESSAGE"
fi

# --- GitHub -----------------------------------------------------------------
if git remote get-url origin >/dev/null 2>&1; then
  echo "Remote 'origin' already exists: $(git remote get-url origin)"
  git push -u origin "$BRANCH"
else
  gh repo create "$NAME" "--$VISIBILITY" --source=. --remote=origin --push
fi

echo
echo "Done: $(gh repo view --json url -q .url)"
