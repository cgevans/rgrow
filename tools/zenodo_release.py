#!/usr/bin/env python3
"""Publish a new version of the rgrow record on Zenodo.

Triggered from CI on a tag push (see .forgejo/workflows/zenodo.yml). Reads:

- .zenodo.json   — static metadata (creators, title, license)
- CHANGELOG.md   — release notes for this version
- CLI args       — tag, source archive URL, Zenodo concept record ID
- env vars:
    ZENODO_TOKEN — personal access token (scopes: deposit:write,deposit:actions)
    ZENODO_HOST  — defaults to zenodo.org; set sandbox.zenodo.org for testing

Run with --dry-run to create the draft (with the file uploaded and metadata
set) but not publish — useful when iterating against sandbox.zenodo.org.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import requests


def extract_changelog(changelog: Path, version: str, tag: str) -> str:
    """Return the body of the CHANGELOG section whose H1 header matches.

    Mirrors the awk extractor used by the Forgejo release job.
    """
    pattern = re.compile(r"^# +(.+?)\s*$")
    section: list[str] = []
    in_section = False
    for line in changelog.read_text().splitlines():
        m = pattern.match(line)
        if m:
            header = m.group(1).strip()
            if in_section:
                break
            if header == version or header == tag:
                in_section = True
            continue
        if in_section:
            section.append(line)
    return "\n".join(section).strip()


def md_to_html(md: str) -> str:
    """Minimal Markdown → HTML good enough for Zenodo's description field."""
    out: list[str] = []
    in_list = False

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    for raw in md.splitlines():
        line = raw.rstrip()
        if line.startswith("### "):
            close_list()
            out.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("## "):
            close_list()
            out.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith(("- ", "* ")):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{line[2:]}</li>")
        elif not line:
            close_list()
        else:
            close_list()
            out.append(f"<p>{line}</p>")
    close_list()
    return "\n".join(out)


def session(token: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {token}"})
    return s


def latest_version_id(host: str, concept_id: str) -> str:
    """Resolve the latest published version record id from a concept id."""
    r = requests.get(
        f"https://{host}/api/records/{concept_id}/versions",
        params={"sort": "mostrecent", "size": 1},
        timeout=30,
    )
    r.raise_for_status()
    hits = r.json()["hits"]["hits"]
    if not hits:
        sys.exit(f"No versions found for concept record {concept_id} on {host}")
    return str(hits[0]["id"])


def new_version(s: requests.Session, host: str, parent_id: str) -> dict[str, Any]:
    base = f"https://{host}/api/deposit/depositions"
    r = s.post(f"{base}/{parent_id}/actions/newversion", timeout=60)
    r.raise_for_status()
    draft_url = r.json()["links"]["latest_draft"]
    r = s.get(draft_url, timeout=30)
    r.raise_for_status()
    return r.json()


def create_deposit(s: requests.Session, host: str) -> dict[str, Any]:
    """Create a brand-new deposit (no concept chain). Used on sandbox."""
    r = s.post(f"https://{host}/api/deposit/depositions", json={}, timeout=60)
    r.raise_for_status()
    return r.json()


def replace_files(s: requests.Session, draft: dict[str, Any], archive: Path) -> None:
    """Drop files copied over from the prior version, then upload the new one."""
    for f in draft.get("files", []):
        s.delete(f["links"]["self"], timeout=30).raise_for_status()
    bucket = draft["links"]["bucket"]
    with archive.open("rb") as fh:
        s.put(f"{bucket}/{archive.name}", data=fh, timeout=600).raise_for_status()


def update_metadata(
    s: requests.Session,
    draft: dict[str, Any],
    *,
    static: dict[str, Any],
    version: str,
    description_html: str,
    repo_url: str,
    tag: str,
) -> None:
    license_id = (static.get("license") or "MIT").lower()
    metadata = {
        "upload_type": "software",
        "title": static["title"],
        "creators": static["creators"],
        "license": license_id,
        "access_right": "open",
        "version": version,
        "description": description_html,
        "related_identifiers": [
            {
                "identifier": f"{repo_url}/src/tag/{tag}",
                "relation": "isSupplementTo",
            }
        ],
    }
    s.put(draft["links"]["self"], json={"metadata": metadata}, timeout=60).raise_for_status()


def publish(s: requests.Session, draft: dict[str, Any]) -> dict[str, Any]:
    r = s.post(draft["links"]["publish"], timeout=120)
    r.raise_for_status()
    return r.json()


def discard_draft(s: requests.Session, draft: dict[str, Any]) -> None:
    self_url = draft.get("links", {}).get("self")
    if self_url:
        try:
            s.delete(self_url, timeout=30)
        except requests.RequestException:
            pass


def download(url: str, target: Path) -> None:
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with target.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 16):
                fh.write(chunk)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", required=True, help="Git tag, e.g. v0.23.0")
    p.add_argument("--archive-url", required=True, help="URL to source tarball")
    p.add_argument(
        "--concept-id",
        default="",
        help="Zenodo concept record ID; if omitted, creates a brand-new deposit "
        "(useful on sandbox.zenodo.org which has its own database).",
    )
    p.add_argument("--repo-url", required=True, help="Repository web URL")
    p.add_argument("--zenodo-json", default=".zenodo.json")
    p.add_argument("--changelog", default="CHANGELOG.md")
    p.add_argument("--host", default=os.environ.get("ZENODO_HOST", "zenodo.org"))
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Create the draft and upload, but don't publish",
    )
    args = p.parse_args()

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("ZENODO_TOKEN env var is required")

    version = args.tag.removeprefix("v")
    static = json.loads(Path(args.zenodo_json).read_text())
    changelog_md = extract_changelog(Path(args.changelog), version, args.tag)
    if not changelog_md:
        print(f"warning: no CHANGELOG section for {version!r}", file=sys.stderr)
        changelog_md = f"Release {args.tag}."
    description_html = (
        md_to_html(changelog_md)
        + f'\n<p>Source: <a href="{args.repo_url}/src/tag/{args.tag}">'
        + f"{args.repo_url}/src/tag/{args.tag}</a></p>"
    )

    s = session(token)

    if args.concept_id:
        print(f"Resolving latest version of concept {args.concept_id} on {args.host}...")
        parent_id = latest_version_id(args.host, args.concept_id)
        print(f"  parent record: {parent_id}")
        print("Creating new version draft...")
        draft = new_version(s, args.host, parent_id)
    else:
        print(f"No concept id supplied; creating fresh deposit on {args.host}...")
        draft = create_deposit(s, args.host)
    new_id = draft["id"]
    print(f"  draft id: {new_id}")

    try:
        with tempfile.TemporaryDirectory() as td:
            archive = Path(td) / f"rgrow-{version}.tar.gz"
            print(f"Downloading {args.archive_url} -> {archive.name}...")
            download(args.archive_url, archive)

            print("Replacing files on draft...")
            replace_files(s, draft, archive)

        print("Updating metadata...")
        update_metadata(
            s,
            draft,
            static=static,
            version=version,
            description_html=description_html,
            repo_url=args.repo_url,
            tag=args.tag,
        )

        if args.dry_run:
            print(
                f"Dry run: draft created, not publishing. Edit at "
                f"https://{args.host}/uploads/{new_id}"
            )
            return 0

        print("Publishing...")
        record = publish(s, draft)
    except Exception:
        print(f"Failed; discarding draft {new_id}", file=sys.stderr)
        discard_draft(s, draft)
        raise

    rec_id = record["id"]
    doi = record.get("doi") or record.get("metadata", {}).get("doi")
    print(f"Published: https://{args.host}/records/{rec_id}  DOI: {doi}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
