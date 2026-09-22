import os
import shutil
import uuid
from pathlib import Path
from typing import Dict, Any

from validate_build import validate_build


def publish_build(
    candidate_dir: Path,
    out_dir: Path,
) -> Dict[str, Any]:
    """
    Validate and atomically publish a PRRISM build.

    The currently published build is left untouched unless
    the candidate passes validation and is fully staged.
    """

    candidate_dir = Path(candidate_dir)
    out_dir = Path(out_dir)

    # Never publish an invalid candidate.
    validation = validate_build(candidate_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    current_dir = out_dir / "current"

    # Stage the candidate beside current/.
    #
    # os.replace() is atomic when source and destination are
    # on the same filesystem, so staging inside out/ matters.
    staging_dir = out_dir / f".staging-{uuid.uuid4().hex}"

    backup_dir = out_dir / f".previous-{uuid.uuid4().hex}"

    try:
        shutil.copytree(candidate_dir, staging_dir)

        # Validate the actual staged copy too.
        validate_build(staging_dir)

        had_current = current_dir.exists()

        if had_current:
            os.replace(current_dir, backup_dir)

        try:
            os.replace(staging_dir, current_dir)

        except Exception:
            # Restore the previous working build if publishing
            # the new one fails after current was moved.
            if had_current and backup_dir.exists():
                os.replace(backup_dir, current_dir)

            raise

        # New build is safely active.
        if backup_dir.exists():
            shutil.rmtree(backup_dir)

    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)

        # Only remove an orphaned backup if current exists.
        if backup_dir.exists() and current_dir.exists():
            shutil.rmtree(backup_dir)

    return validation