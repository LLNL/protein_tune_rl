#!/usr/bin/env python3
"""
Patch *_template.json configs by substituting a placeholder (default: "${IGLM_DIR}")
and writing the result to the same filename without "template".

Example:
  python scripts/patch_iglm_dir.py --value "/abs/path/to/iglm/trained_models/IgLM-S"

Also supports writing another token (e.g., "${IGLM_DIR}") as the value:
  python scripts/patch_iglm_dir.py --value "${IGLM_DIR}"
"""
from pathlib import Path
import sys
import click

DEFAULT_DIR = "./configs/examples"
DEFAULT_PLACEHOLDER = "${IGLM_DIR}"


def derive_output_path(src: Path) -> Path:
    name = src.name
    if "_template" in name:
        new_name = name.replace("_template", "", 1)
    elif "template" in name:
        # Fallback: remove first occurrence of "template" if underscore variant isn't present
        new_name = name.replace("template", "", 1)
    else:
        raise ValueError(f"File does not look like a template: {src.name}")
    return src.with_name(new_name)


@click.command()
@click.option(
    "--config-dir",
    default=DEFAULT_DIR,
    show_default=True,
    help="Directory containing *_template.json files.",
)
@click.option(
    "--placeholder",
    default=DEFAULT_PLACEHOLDER,
    show_default=True,
    help='Exact placeholder text to replace (e.g., "${IGLM_DIR}").',
)
@click.option(
    "--value",
    "replacement",
    required=True,
    help='Replacement text (e.g., an absolute path or another token like "${IGLM_DIR}").',
)
@click.option(
    "--recursive/--no-recursive",
    default=False,
    show_default=True,
    help="Recurse into subdirectories.",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    show_default=True,
    help="Allow overwriting existing non-template files.",
)
def main(config_dir, placeholder, replacement, recursive, overwrite):
    base = Path(config_dir)
    if not base.exists() or not base.is_dir():
        click.echo(f"ERROR: directory not found: {base}", err=True)
        sys.exit(1)

    pattern = "**/*_template.json" if recursive else "*_template.json"
    files = sorted(base.glob(pattern))

    if not files:
        click.echo("No *_template.json files found.")
        return

    total_files = 0
    total_repls = 0

    for src in files:
        try:
            text = src.read_text(encoding="utf-8")
        except Exception as e:
            click.echo(f"Skipping unreadable file {src}: {e}", err=True)
            continue

        count = text.count(placeholder)
        new_text = text.replace(placeholder, replacement)

        dst = derive_output_path(src)

        if dst.exists() and not overwrite:
            click.echo(f"SKIP (exists, use --overwrite): {dst}")
            continue

        try:
            dst.write_text(new_text, encoding="utf-8")
        except Exception as e:
            click.echo(f"ERROR writing {dst}: {e}", err=True)
            continue

        total_files += 1
        total_repls += count
        click.echo(f"Wrote {dst} (from {src.name}; {count} replacements)")

    click.echo(f"Done. Files written: {total_files}, total replacements: {total_repls}")


if __name__ == "__main__":
    main()
