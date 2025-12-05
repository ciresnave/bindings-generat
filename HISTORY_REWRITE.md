HISTORY REWRITE NOTICE
======================

What happened
-------------
On 2025-12-05 we ran a history rewrite on the `main` branch to remove inadvertently committed build artifacts (notably all `**/target/**` directories and `*.profraw` profiling outputs). This reduced repository size by removing those large files from the repository history.

Why
---
Build outputs and profiling data should not be part of the repository history. They make the repository large and slow to clone. We removed them to keep the repository lean.

What was removed
----------------
- All historical blobs under any `target/` directory
- `*.profraw` files

Important consequences
----------------------
- The repository history was rewritten and `origin/main` was force-updated. This is a destructive operation for branch history: any local clones that still reference the old history will diverge.

Recommended action for collaborators
-----------------------------------
The safest approach is to re-clone the repository:

```powershell
git clone https://github.com/ciresnave/bindings-generat.git
```

If you cannot re-clone and understand the risks, you can update an existing local clone by running:

```powershell
# Discard local changes — this will overwrite local state
git fetch origin
git reset --hard origin/main
git clean -fdx
```

If you have local commits you need to keep, create a backup branch first:

```powershell
git branch backup-before-rewrite
git fetch origin
git rebase --onto origin/main <commit-old-base> backup-before-rewrite
```

If you need help recovering local work, contact the repository maintainer before running the reset.

Notes
-----
- This rewrite removed the `origin` remote during the operation as a safety precaution; it was re-added and the rewritten refs were force-pushed to `origin`.
- Removing files from history via `git filter-repo` does not scrub them from every mirror or cached location; if you host Git on services with LFS or cache, additional cleanup may be necessary.

Contact
-------
If anything looks wrong, or you want assistance migrating local branches, ping the repository owner.
