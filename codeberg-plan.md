# Codeberg Actions Improvement Plan

This plan covers fixes and improvements for the Forgejo Actions workflows in
`.forgejo/workflows/`, plus related repo metadata that should move with the
Codeberg migration. The intent is to keep GitHub workflows working during the
transition while making the Codeberg workflows reliable, idiomatic, and useful
as release infrastructure.

## Goals

- Make Forgejo Actions pass reliably on Codeberg/self-hosted Forgejo runners.
- Preserve release behavior currently provided by GitHub Actions unless a change
  is intentional and documented.
- Use Codeberg/Forgejo-native features where they add value, especially Pages
  and Packages.
- Reduce GitHub-specific naming in Forgejo workflows so future maintenance is
  less confusing.
- Keep the migration reversible until Codeberg is the primary forge.

## Current Workflow Inventory

- `.forgejo/workflows/ci.yml`
  - Rust checks.
  - Python checks.
  - Linux, Windows, and macOS Python wheel builds.
  - Free-threaded Python wheel builds.
  - Source distributions.
  - PyPI release.
  - crates.io release.
  - Forgejo release creation.
- `.forgejo/workflows/docs.yml`
  - Builds the docs site.
  - Pushes generated output to a `pages` branch.
- `.github/workflows/ci.yml` and `.github/workflows/docs.yml`
  - Existing GitHub equivalents retained for comparison and fallback.

## Phase 1: Correctness Fixes

These should be done before adding new capabilities.

### 1. Use Explicit Action URLs

Problem: Forgejo resolves bare `uses:` entries through the instance
`DEFAULT_ACTIONS_URL`, not necessarily GitHub. The Forgejo workflows currently
still use bare third-party actions such as `astral-sh/setup-uv@v7`.

Files:

- `.forgejo/workflows/ci.yml`
- `.forgejo/workflows/docs.yml`

Changes:

- Replace bare third-party actions with absolute URLs, for example:
  - `astral-sh/setup-uv@v7` -> `https://github.com/astral-sh/setup-uv@v7`
- Keep Forgejo-hosted actions as Forgejo URLs:
  - `https://code.forgejo.org/forgejo/upload-artifact@v4`
  - `https://code.forgejo.org/forgejo/download-artifact@v4`
  - `https://code.forgejo.org/actions/forgejo-release@v2`
- Consider pinning critical third-party actions to full commit SHAs after the
  first successful Codeberg run.

Validation:

- Run the workflows on a non-tag branch.
- Confirm action checkout/setup steps resolve without relying on a local mirror.

### 2. Gate Wheel Builds to Tags or Manual Dispatch

Problem: The Forgejo workflow currently builds wheels on every push and pull
request. We do not need wheel builds on every commit, on any platform. Normal CI
should run checks; wheel builds should run for releases and when explicitly
requested for a branch or pull request.

Files:

- `.forgejo/workflows/ci.yml`

Changes:

- Add a manual workflow input that can request wheel builds:

```yaml
on:
  workflow_dispatch:
    inputs:
      build_wheels:
        description: "Build Python wheels"
        required: false
        default: "false"
        type: choice
        options:
          - "false"
          - "true"
```

- Gate all wheel and sdist jobs to release tags or that manual input:
  - `linux`
  - `linux-freethreaded`
  - `windows`
  - `windows-freethreaded`
  - `macos`
  - `macos-freethreaded`
  - `sdist`
- Use a shared condition on those jobs:

```yaml
if: startsWith(forge.ref, 'refs/tags/') || inputs.build_wheels == 'true'
```

- Keep publish jobs tag-only:

```yaml
if: startsWith(github.ref, 'refs/tags/')
```

- Use native Forgejo context where supported:
  - `forge.ref`
  - `forge.ref_name`
  - `inputs.build_wheels`
- Fallback to `github.ref` only if the deployed runner is too old for `forge.*`.

Validation:

- Push a branch: only Rust and Python checks should run.
- Open/update a pull request: only Rust and Python checks should run.
- Use `workflow_dispatch` on a branch or pull request ref with
  `build_wheels=true`: wheel and sdist jobs should run, but publish/release jobs
  should not.
- Push a tag: all release wheel jobs should run.

### 3. Keep Manual Wheel Builds Ergonomic

Manual wheel builds should be easy to run before merging a packaging-sensitive
pull request or branch.

Files:

- `.forgejo/workflows/ci.yml`

Changes:

- Keep `workflow_dispatch` available on the CI workflow.
- Add `build_wheels` as described above.
- Consider adding narrower manual inputs later if runner capacity requires it:
  - `wheel_platform: all/linux/windows/macos`
  - `free_threaded: true/false`
  - `publish: false`
- Do not add a manual `publish` input initially. Publication should remain
  tag-driven until the release process is stable.

Recommendation:

- Start with a single `build_wheels` boolean. It is simple and avoids divergent
  behavior between ad-hoc builds and release builds.
- If wheel builds are too expensive, split the workflow later into:
  - `ci.yml` for checks.
  - `wheels.yml` for tag/manual wheel builds.

Validation:

- Manually trigger the workflow on a PR branch with `build_wheels=false`.
- Manually trigger the workflow on the same ref with `build_wheels=true`.
- Confirm artifacts are produced only for the second run.

### 4. Document macOS x86_64 Deprecation

Policy: We have no `x86_64` macOS runners, and Intel macOS wheels are being
deprecated. Forgejo should build only Apple Silicon macOS wheels.

Files:

- `.forgejo/workflows/ci.yml`
- Release notes or changelog, when the deprecation is announced.

Changes:

- Keep macOS wheel matrices at `target: [aarch64]`.
- Update comments in the Forgejo workflow to say this is intentional, not a
  temporary parity gap.
- Remove migration-plan language that suggests restoring `x86_64` macOS wheels.
- When Codeberg becomes the primary release path, announce that macOS `x86_64`
  users should install from source or use the last release that provided Intel
  wheels.

Validation:

- Tag-run artifacts should include macOS `aarch64` wheels only.
- Artifact comparison against GitHub should treat missing macOS `x86_64` wheels
  as an intentional release policy change, not a failure.

Validation:

- Compare artifacts from a tag run against the latest GitHub release artifact
  set.
- Confirm both `rgrow` and `rgrow-cli` wheels are present for each target.

### 5. Use Forgejo-Native Release Context

The current Forgejo release step uses `release-notes-file`, which is supported
by current `actions/forgejo-release`. Prefer Forgejo-native context names over
GitHub compatibility aliases.

Files:

- `.forgejo/workflows/ci.yml`

Implemented changes:

- Use Forgejo-native names:
  - `${{ forge.token }}`
  - `${{ forge.server_url }}`
  - `${{ forge.repository }}`
  - `${{ forge.ref_name }}`
- Keep compatibility aliases only where a third-party action specifically
  expects GitHub naming.

Validation:

- Create a dry-run tag in a test repository or temporary branch/repo.
- Confirm release title, tag, notes, and assets are correct.

## Phase 2: Codeberg Pages Modernization

### 1. Switch Docs Deployment to git-pages Action

Problem: The current docs workflow force-pushes generated docs to a `pages`
branch. Codeberg now recommends the `git-pages/action` deployment path for
sites hosted under `codeberg.page`.

Files:

- `.forgejo/workflows/docs.yml`

Changes:

- Replace the manual `git init`, `git commit`, and `git push --force` step with:

```yaml
- name: Deploy to Codeberg Pages
  uses: https://codeberg.org/git-pages/action@v2
  with:
    site: "https://${{ forge.repository_owner }}.codeberg.page/rgrow/"
    token: ${{ forge.token }}
    source: site/
```

Notes:

- If the final URL is not `https://cgevans.codeberg.page/rgrow/`, adjust the
  `site` value.
- If the project uses a custom domain, check Codeberg's current custom-domain
  Pages limitations before switching.

Validation:

- Run `workflow_dispatch` for docs.
- Confirm the deployed site serves from the expected Codeberg Pages URL.
- Confirm the response is served by the new git-pages service if relevant.

### 2. Update Documentation Metadata

Problem: The generated site and README still advertise GitHub Pages and GitHub
as canonical locations.

Files:

- `mkdocs.yml`
- `README.md`

Changes:

- Update `site_url` to the Codeberg Pages URL.
- Update `repo_url` and `repo_name` to Codeberg once the Codeberg repository is
  canonical.
- Update README docs badge and docs link.
- Consider keeping a note that GitHub remains a mirror while migration is in
  progress.

Validation:

- Build docs locally or in CI.
- Inspect generated canonical links and repository links.

## Phase 3: Codeberg Packages

Codeberg/Forgejo Packages are owner-level, not repository-level, but packages
can be linked back to a repository in the UI. Use Packages as a release mirror
and artifact store, not as a replacement for PyPI or crates.io.

### 1. Add Generic Package Upload for Release Artifacts

Use the Generic Package registry to store every file in `dist/` under a version
matching the release tag.

Files:

- `.forgejo/workflows/ci.yml`

New secret:

- `PACKAGE_TOKEN`
  - Token with `write:packages`.
  - Prefer a dedicated bot/user token if Codeberg's automatic workflow token is
    insufficient or does not have package write scope.

Proposed job:

```yaml
  package_release:
    name: Codeberg Packages release mirror
    runs-on: docker
    container:
      image: docker.io/catthehacker/ubuntu:act-22.04
    if: startsWith(forge.ref, 'refs/tags/')
    needs: [linux, linux-freethreaded, windows, windows-freethreaded, macos, macos-freethreaded, sdist]
    steps:
      - uses: https://code.forgejo.org/forgejo/download-artifact@v4
        with:
          pattern: wheels-*
          merge-multiple: true
          path: dist
      - name: Upload release files to Generic Packages
        env:
          TOKEN: ${{ secrets.PACKAGE_TOKEN }}
          OWNER: ${{ forge.repository_owner }}
          VERSION: ${{ forge.ref_name }}
          SERVER_URL: ${{ forge.server_url }}
        run: |
          set -euo pipefail
          for file in dist/*; do
            curl --fail --header "Authorization: token ${TOKEN}" \
              --upload-file "$file" \
              "${SERVER_URL}/api/packages/${OWNER}/generic/rgrow-dist/${VERSION}/$(basename "$file")"
          done
```

Notes:

- Generic package uploads are immutable per file name. Re-running a tag release
  will return conflict unless the package version/file is deleted first.
- Do not add automatic deletion until the release process is mature.

Validation:

- Trigger on a test tag.
- Confirm package appears under the Codeberg owner Packages UI.
- Link the package to the `rgrow` repository in the package settings.

### 2. Consider PyPI Registry Mirroring

Forgejo also supports PyPI packages. This can mirror `rgrow` and `rgrow-cli`
inside Codeberg Packages.

Pros:

- Users can install from Codeberg if PyPI is unavailable or if they prefer a
  forge-hosted package source.
- The package appears as a first-class Codeberg package.

Cons:

- It is another package index to document and support.
- It can create dependency confusion risk if users use `--extra-index-url`
  carelessly.
- Upload tooling may require `twine`; `uv publish` support for arbitrary
  Forgejo PyPI endpoints should be confirmed before implementation.

Recommendation:

- Start with Generic Packages.
- Add PyPI registry mirroring only if there is a concrete user or archival need.
- If added, document installation with `--index-url` rather than
  `--extra-index-url`.

### 3. Consider Cargo Registry Mirroring Later

Forgejo supports a Cargo registry, but using it requires owner-level Cargo index
setup and user configuration.

Recommendation:

- Keep crates.io as the canonical Rust distribution.
- Do not mirror Cargo packages in the first Codeberg workflow update.
- Revisit if there is a reason to distribute Rust crates through Codeberg.

### 4. Add Cleanup Rules

Once Packages are used:

- Keep release-tagged Generic package versions indefinitely.
- If nightly/dev packages are added later, keep only the most recent N builds.
- Avoid publishing branch artifacts to Packages until cleanup policy is in
  place.

## Phase 4: Security and Runner Hardening

### 1. Reduce Privileges Per Job

Forgejo may ignore some GitHub-style `permissions` keys depending on version,
but keeping intent in the workflow is still useful.

Changes:

- Keep read-only defaults where possible.
- Isolate write operations to:
  - docs deploy
  - Forgejo release
  - package upload
  - PyPI/crates.io publication
- Use separate secrets for:
  - `PYPI_TOKEN`
  - `CRATES_IO_TOKEN`
  - `PACKAGE_TOKEN`
  - `CODECOV_TOKEN`

### 2. Limit Secret Exposure on Pull Requests

Check Codeberg/Forgejo pull request secret behavior for the deployed runner
version. Ensure package/release/publish jobs only run on tags from trusted refs.

Actions:

- Keep all publish jobs guarded by `startsWith(forge.ref, 'refs/tags/')`.
- Avoid `pull_request_target` unless there is a specific need.
- Do not expose package tokens to PR-triggered jobs.

### 3. Pin Tooling Where It Matters

Current workflows use moving tags such as `@v7`, `@v2`, and `@master`.

Recommendation:

- During initial migration, keep version tags for easier updates.
- After workflows are stable, pin high-impact third-party actions:
  - `dtolnay/rust-toolchain`
  - `astral-sh/setup-uv`
  - `taiki-e/install-action`
  - `PyO3/maturin-action`
- Keep Forgejo-provided artifact actions on supported patched versions.

## Phase 5: Maintainability Improvements

### 1. Remove Unused Top-Level Env Constants

`.forgejo/workflows/ci.yml` defines `LINUX_CONTAINER` and
`MANYLINUX_CONTAINER`, but the workflow currently repeats image strings instead
of using those env vars. Workflow `container.image` cannot always interpolate
env values consistently across implementations.

Recommendation:

- Either remove the unused env constants or convert repeated image references
  only if Forgejo runner supports the interpolation.
- Prefer clarity over clever indirection.

### 2. Add a Short Migration Comment Block

The current comments are helpful but long. After the workflows stabilize,
replace the large migration notes with:

- runner labels required
- secrets required
- package publishing behavior
- link to this plan or a shorter maintainer doc

### 3. Add Workflow Validation

Options:

- Add Forgejo runner workflow validation as a local pre-commit hook.
- Add a small CI job that validates workflow syntax if the runner supports it.
- Use `actionlint` locally for GitHub-compatible checks, but do not rely on it
  for Forgejo-specific semantics.

## Suggested Implementation Order

1. Replace bare third-party `uses:` entries in Forgejo workflows.
2. Gate all wheel and sdist jobs to release tags or manual `workflow_dispatch`
   with `build_wheels=true`.
3. Switch docs deployment to `git-pages/action@v2`.
4. Update `mkdocs.yml` and README Codeberg links once the Codeberg URL is final.
5. Run branch workflows and fix runner/tool assumptions.
6. Run a test tag in a non-production repository or with a disposable version.
7. Add Generic Package publishing for release artifacts.
8. Compare release artifacts against GitHub output.
9. Document macOS `x86_64` wheel deprecation in release-facing notes.
10. Pin critical actions after the workflow is stable.

## Acceptance Criteria

- A normal branch push runs checks only and does not build wheels.
- A pull request update runs checks only and does not build wheels.
- A manual workflow run with `build_wheels=true` builds wheels and sdists for
  the selected branch or pull request ref, without publishing.
- A tag push produces the expected Python wheels and sdists.
- macOS release artifacts intentionally include `aarch64` wheels only.
- PyPI and crates.io publication still work from Codeberg tags.
- A Forgejo release is created with the expected notes and assets.
- Docs deploy to the expected Codeberg Pages URL.
- Release artifacts are mirrored to Codeberg Generic Packages, if Phase 3 is
  implemented.
- README and generated docs do not point users to GitHub as the canonical home
  once Codeberg becomes primary.

## Open Decisions

- What is the final Codeberg Pages URL?
- Will GitHub remain the primary release publisher during a transition period?
- Should Codeberg Packages mirror only release bundles, or also PyPI packages?
