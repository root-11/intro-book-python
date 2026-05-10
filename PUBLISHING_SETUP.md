# Publishing setup — Forgejo → Codeberg + GitHub

This document is a one-time setup checklist. After it is done, every push to `main` on Forgejo automatically:

1. Builds the book on the `nuc` runner.
2. Pushes the rendered HTML to Codeberg's `pages` branch (serves at `https://root-11.codeberg.page/intro-book-python/`).
3. Pushes the same rendered HTML to GitHub's `gh-pages` branch (serves at `https://root-11.github.io/intro-book-python/`).
4. Mirrors the source `main` branch to both Codeberg and GitHub via Forgejo's Push Mirror feature.

**Two failure-independent hosts** for both the live book and the source clone. Codeberg outages no longer take the book offline; readers fall through to GitHub Pages and `git clone` from GitHub continues to work.

---

## Assumptions

If any of these are wrong, fix the names everywhere they appear and re-run the affected step.

| name | value | where it appears |
|---|---|---|
| Codeberg username | `root-11` | publish.yml `CODEBERG_USER` secret, front_matter.md |
| GitHub username | `root-11` (assumed parallel to Codeberg) | publish.yml `GITHUB_USER` secret, front_matter.md |
| Codeberg public repo name | `intro-book-python` | publish.yml, front_matter.md |
| GitHub public repo name | `intro-book-python` (matches Codeberg) | publish.yml, front_matter.md |
| Forgejo source repo | `intro-to-programming-python-edition` | unchanged; this is the upstream you push to |
| Forgejo runner label | `linux-amd64-nuc` | publish.yml `runs-on:` |

The Rust edition uses the same template with `intro-book` everywhere instead of `intro-book-python`; the same setup steps apply.

---

## Phase A — Create the public repos (one-time)

### A1. Codeberg public repo

1. Sign in to https://codeberg.org as `root-11`.
2. Create a new repo named **`intro-book-python`** (matching the front_matter URLs).
3. Mark it public.
4. Do NOT initialise with README/LICENSE — Forgejo will push to it.

### A2. GitHub public repo

1. Sign in to https://github.com as `root-11` (or whatever your handle is — update `PUBLISHING_SETUP.md`, `front_matter.md`, and `publish.yml` if different).
2. Create a new repo named **`intro-book-python`**.
3. Mark it public.
4. Do NOT initialise with README/LICENSE.

### A3. Repeat for the Rust edition

If you haven't already mirrored the Rust edition to GitHub: create `root-11/intro-book` on GitHub. (Codeberg already has it.)

---

## Phase B — Generate SSH keys for the CI runner (one-time)

The runner pushes via SSH (not HTTPS+PAT) because it's simpler and doesn't expire. Two keys, one per host. Generate them on the pi (or wherever Forgejo's runner config lives):

```bash
ssh-keygen -t ed25519 -f ~/.ssh/forgejo-ci-codeberg -C "forgejo-ci → codeberg" -N ""
ssh-keygen -t ed25519 -f ~/.ssh/forgejo-ci-github   -C "forgejo-ci → github"   -N ""
```

This produces four files. The `.pub` files go to the host; the private keys go into Forgejo secrets.

### B1. Add the public keys to Codeberg and GitHub

**Important: deploy keys are per-repo on both platforms.** A deploy key authorised for `intro-book` cannot push to `intro-book-python` — same scoping rule on Codeberg as on GitHub. You have two options per platform; pick the same model on both for consistency.

#### Option 1 — User-level SSH keys (simpler, recommended)

One keypair per platform, attached to your *user account*; covers all current and future repos.

- **Codeberg:** https://codeberg.org/user/settings/keys → "Add Key". Paste `~/.ssh/forgejo-ci-codeberg.pub`. Title: "forgejo-ci".
- **GitHub:** https://github.com/settings/keys → "New SSH key". Paste `~/.ssh/forgejo-ci-github.pub`. Title: "forgejo-ci".

This is the simplest path for a single-author book. Two keys total (one per platform), one private key per Forgejo repo's secret.

#### Option 2 — Per-repo deploy keys (tighter, more setup)

If you want each key scoped to one repo, generate **one keypair per repo per platform** — so four keypairs total for two repos × two platforms. For each:

- **Codeberg:** Repo → Settings → Deploy Keys → "Add Deploy Key". Paste the `.pub`. Tick "Enable write access".
- **GitHub:** Repo → Settings → Deploy keys → "Add deploy key". Paste the `.pub`. Tick "Allow write access".

Each Forgejo repo's `CODEBERG_SSH_KEY` / `GIT_HUB_SSH_KEY` secret then holds the *private* key matching that repo's deploy key.

> **If the runner reports** `(Deploy) Key ... is not authorized to write to .../<other-repo>`: you used Option 2 (deploy key) but the key is attached to a different repo than the one being pushed to. Either add the same `.pub` as a deploy key on the failing repo, or move to Option 1 by removing it from the deploy-keys list and adding it as a user-level key.

---

## Phase C — Configure Forgejo secrets (one-time per repo)

For each Forgejo repo (Rust edition + Python edition), add these secrets via the repo's Settings → Actions → Secrets UI:

| secret | value |
|---|---|
| `CODEBERG_SSH_KEY` | contents of `~/.ssh/forgejo-ci-codeberg` (the **private** key, full file including `-----BEGIN OPENSSH PRIVATE KEY-----` headers) |
| `CODEBERG_USER` | `root-11` |
| `GIT_HUB_SSH_KEY` | contents of `~/.ssh/forgejo-ci-github` (private key) |
| `GIT_HUB_USER` | your GitHub username (assumed `root-11`) |

> **Why `GIT_HUB_*` and not `GITHUB_*`:** Forgejo reserves the `GITHUB_*` namespace for actions-protocol-compatibility secrets and refuses to create secrets starting with that prefix. The workflow yaml uses `GIT_HUB_SSH_KEY` and `GIT_HUB_USER` (with underscore) to dodge the reservation; the secrets you store must match that name exactly.

If the Forgejo Actions UI doesn't have a secrets manager, the equivalent is environment variables on the runner (less secure but workable for a homelab).

---

## Phase D — Configure Forgejo Push Mirror for source `main` (one-time per repo)

This is what gets the *source code* onto Codeberg and GitHub `main` branches, so `git clone` works for readers.

For each Forgejo repo:

1. Repo → Settings → Repository → **Mirror** tab.
2. Click "Add Push Mirror".
3. **Mirror to Codeberg:**
   - Git Remote URL: `git@codeberg.org:root-11/intro-book-python.git` (use the SSH form so the key from B1 authenticates).
   - Authorization: select the SSH key you added.
   - Sync interval: "On every push" (or every 5 minutes if Forgejo doesn't expose on-push).
4. Click "Add Push Mirror" again.
5. **Mirror to GitHub:**
   - Git Remote URL: `git@github.com:root-11/intro-book-python.git`.
   - Authorization: select the SSH key from B2.
   - Sync interval: same as above.

Forgejo will now push the `main` branch to both remotes after every commit you make on Forgejo. This handles the *source* mirror; the publish.yml workflow handles the *rendered output* mirror.

---

## Phase E — First-time push to populate everything

```bash
cd ~/code/intro-py
git add -A
git commit -m "Initial publish"
git push origin main
```

This triggers:
- Forgejo Push Mirror → pushes `main` to Codeberg and GitHub
- `.forgejo/workflows/publish.yml` → builds, pushes rendered output to Codeberg `pages` and GitHub `gh-pages`

Watch the workflow log on Forgejo to confirm both push steps succeed. If GitHub Pages doesn't appear immediately, check Settings → Pages on the GitHub repo and select "Deploy from a branch → gh-pages" if not already.

---

## Phase F — Set the default branch on each platform

Each platform serves `https://<host>/<user>/<repo>` based on its default branch.

- **Codeberg:** Set default branch to `pages` (so Codeberg's repo viewer renders the README inline as the SEO surface). Reader who wants source: `git clone -b main`.
- **GitHub:** Keep default branch as `main` (GitHub's social conventions; people expect to land on the source). Reader who wants the rendered book: visit the GitHub Pages URL or click "GitHub Pages" in the repo header.
- **Forgejo:** Default branch stays `main` (Forgejo is your private upstream).

This default-branch asymmetry is the same trick the Rust edition uses; readers get the right experience on each platform.

---

## Verification

After Phase E completes, confirm:

```bash
# Source clones from each public host
cd /tmp
git clone https://codeberg.org/root-11/intro-book-python.git -b main intro-py-codeberg
git clone https://github.com/root-11/intro-book-python.git intro-py-github
diff -r intro-py-codeberg intro-py-github  # should be silent

# Rendered output URLs respond
curl -sI https://root-11.codeberg.page/intro-book-python/ | head -1   # 200 OK
curl -sI https://root-11.github.io/intro-book-python/ | head -1        # 200 OK

# The four URLs in front_matter.md all resolve
```

Both `git clone` commands now produce a working source tree (with `code/measurement/`, `book/trunk/`, `build.py` etc.), so a reader can run the exhibits.

---

## Maintenance notes

- **Adding a new platform** (GitLab, Sourcehut, ...): add another Push Mirror in Forgejo (Phase D) and another set of dist-push steps in `publish.yml`.
- **Rotating SSH keys**: regenerate the keypair, replace the public key on the host, update the Forgejo secret with the new private key. The runner picks up the change on next workflow run.
- **A platform goes down**: nothing to do. Forgejo retries the mirror push on its schedule; the surviving platforms keep serving.
- **Renaming the public repo**: update the URL in three places: Forgejo's mirror config, the `git remote add` line in publish.yml, and the front_matter URLs.
