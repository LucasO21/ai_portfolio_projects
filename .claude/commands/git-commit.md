# Git Commit Skill

Commit changes to the repository with optional push to GitHub.

## Instructions

Execute the following steps with appropriate safety checks:

### Step 1: Verify Git Repository
- Confirm we're inside a valid git repository by running `git rev-parse --is-inside-work-tree`
- If not a git repo, stop and inform the user

### Step 2: Check Current Branch
- Run `git branch --show-current` to display the current branch
- Warn if on `main` or `master` branch and confirm the user wants to proceed

### Step 3: Show Current Status
- Run `git status` to show the user what files have changed
- If there are no changes (working tree clean), inform the user and stop

### Step 4: Review Changes
- Run `git diff --stat` to show a summary of changes
- Ask the user to confirm they want to proceed with these changes

### Step 5: Stage Changes
- Run `git add -A` to stage all changes
- Confirm staging was successful

### Step 6: Validate Commit Message
- Generate a commit message summarizing changes in no more than 25 words.
- Ensure the commit message is not empty
- Ensure the commit message is descriptive (more than 3 characters)

### Step 7: Commit Changes
- Run `git commit -m "$COMMIT_MESSAGE"`
- Confirm the commit was successful

### Step 8: Push Decision
- Ask the user: "Do you want to push to GitHub? (yes/no)"
- If yes, proceed to Step 9
- If no, inform the user the commit is complete locally and stop

### Step 9: Verify Remote
- Run `git remote -v` to verify the remote is configured
- Expected remote: https://github.com/LucasO21/ai_portfolio_projects.git
- If remote is not configured, offer to add it with:
  `git remote add origin https://github.com/LucasO21/ai_portfolio_projects.git`

### Step 10: Push to GitHub
- Run `git push origin <current-branch>`
- If push fails due to upstream not set, run:
  `git push --set-upstream origin <current-branch>`
- Confirm push was successful and provide the GitHub link

### Safety Reminders
- Never force push without explicit user confirmation
- Always show what will be committed before committing
- Check for sensitive files (.env, secrets, API keys) in staged changes and warn the user
```

