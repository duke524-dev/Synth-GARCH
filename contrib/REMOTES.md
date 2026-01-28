# Git remotes and syncing with source

This project tracks two remotes:

| Remote    | URL                                             | Use |
|-----------|--------------------------------------------------|-----|
| **origin**   | https://github.com/duke524-dev/Synth-GARCH.git   | Your repo — push your work here. |
| **upstream** | https://github.com/mode-network/synth-subnet.git  | Source project — fetch updates from here. |

## Push your work to Synth-GARCH

```bash
git add ...
git commit -m "Your message"
git push -u origin main
```

Subsequent pushes:

```bash
git push origin main
```

## When the source project is updated — fetch and merge

To bring in changes from the upstream (source) repo:

```bash
# Fetch latest from source
git fetch upstream

# Merge upstream/main into your current branch (e.g. main)
git merge upstream/main

# Resolve any conflicts, then push to your repo
git push origin main
```

Or rebase on top of upstream (keeps a linear history):

```bash
git fetch upstream
git rebase upstream/main
git push origin main
```

## Check remotes

```bash
git remote -v
```

## One-time setup (already done)

- `origin` = https://github.com/duke524-dev/Synth-GARCH.git  
- `upstream` = https://github.com/mode-network/synth-subnet.git  

If you ever need to re-add them:

```bash
git remote add origin https://github.com/duke524-dev/Synth-GARCH.git
git remote add upstream https://github.com/mode-network/synth-subnet.git
```
