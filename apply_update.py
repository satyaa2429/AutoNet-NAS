"""Apply this package to an existing AutoNet-NAS project, with a backup."""

import argparse
import hashlib
import json
import re
import shutil
import tomllib
from datetime import datetime
from pathlib import Path, PurePosixPath


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def theme_matches(source, target):
    expected = tomllib.loads(source.read_text(encoding="utf-8-sig")).get("theme", {})
    current = tomllib.loads(target.read_text(encoding="utf-8-sig")).get("theme", {})
    return current == expected


def merge_theme(source, target):
    """Change the theme while preserving server and other configuration tables."""
    replacement = source.read_text(encoding="utf-8-sig").strip() + "\n"
    text = target.read_text(encoding="utf-8-sig")
    expected = tomllib.loads(text)
    expected["theme"] = tomllib.loads(replacement)["theme"]
    remaining = []
    start = 0
    in_theme = False
    for match in re.finditer(r"(?m)^[ \t]*\[[^\n]*", text):
        try:
            table = tomllib.loads(match[0])
            # A header-like line inside a multiline value is not a new table.
            tomllib.loads(text[:match.start()])
        except tomllib.TOMLDecodeError:
            continue
        if not in_theme:
            remaining.append(text[start:match.start()])
        start = match.start()
        in_theme = "theme" in table
    if not in_theme:
        remaining.append(text[start:])
    other_settings = "".join(remaining).rstrip()
    merged = other_settings + "\n\n" + replacement if other_settings else replacement
    if tomllib.loads(merged) != expected:
        raise ValueError("The theme update would change another Streamlit setting.")
    return merged


def apply(package, project):
    package, project = package.resolve(), project.resolve()
    if not (project / "src" / "architectures.py").is_file():
        raise ValueError("Choose the existing AutoNet-NAS project folder with --project.")
    if package == project:
        raise ValueError("Extract the update to a separate folder before applying it.")
    manifest = json.loads((package / "update_manifest.json").read_text(encoding="utf-8"))
    planned = []
    for item in manifest["files"]:
        relative = PurePosixPath(item["path"])
        if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
            raise ValueError("Invalid path in update manifest.")
        source, target = package / str(relative), project / str(relative)
        if not target.resolve().is_relative_to(project):
            raise ValueError("An update target points outside the project.")
        if digest(source) != item["sha256"]:
            raise ValueError(f"The update is incomplete or damaged: {relative}")
        if item.get("preserve_existing") and target.is_file():
            continue
        if target.is_file() and digest(target) == item["sha256"]:
            continue
        if str(relative) == ".streamlit/config.toml" and target.is_file() and theme_matches(source, target):
            continue
        planned.append((source, target, str(relative), target.exists()))
    if not planned:
        print("This update is already installed.")
        return
    backup = project / "backups" / ("update_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    for _, target, relative, existed in planned:
        if existed:
            saved = backup / relative
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, saved)
    completed = []
    try:
        for source, target, relative, existed in planned:
            completed.append((target, relative, existed))
            target.parent.mkdir(parents=True, exist_ok=True)
            if relative == ".streamlit/config.toml" and existed:
                target.write_text(merge_theme(source, target), encoding="utf-8")
            else:
                shutil.copy2(source, target)
    except BaseException:
        for target, relative, existed in reversed(completed):
            if existed:
                shutil.copy2(backup / relative, target)
            elif target.is_file():
                target.unlink()
        raise
    print(f"Updated {len(completed)} files in: {project}")
    print(f"Previous versions backed up in: {backup}")
    print("Start the app: python -m streamlit run app/streamlit_app.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    apply(Path(__file__).resolve().parent, args.project)


if __name__ == "__main__":
    main()
