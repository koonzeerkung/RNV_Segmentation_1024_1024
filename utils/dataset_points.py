"""Single source of truth for dataset-specific nearby-filter defaults.

Both `visualize_nearby.py` (auto-resolves when points_csv/points_image_name_template
are left blank) and `run_experiment_queue.py` (fills in per-experiment overrides) read
from these mappings so the two never drift apart.

Two point sets exist per dataset, selected by `points_mode`:

* ``"multiple"`` -- every point the localizer emits, several per lesion. TU has 591 points
  over 27 images (median 10 per image), DRAC 280 over 39 (median 4).
* ``"single"``   -- one representative point per lesion, obtained by clustering the above.
  TU has 55 points, DRAC 89 (median 2 per image, i.e. two lesions per image).

The image-name format is the same in both sets, so only the CSV path depends on the mode.
"""

from __future__ import annotations

POINTS_MODES = ("multiple", "single")
DEFAULT_POINTS_MODE = "multiple"

# Kept under its original name and shape: run_experiment_queue.py imports it directly.
DATASET_POINTS_CSV = {
    "tu": "TU_RNV_Points_Unclustered.csv",
    "drac": "DRAC_RNV_Points_Unclustered.csv",
}

DATASET_POINTS_CSV_SINGLE = {
    "tu": "combined_representative_points.csv",
    "drac": "combined_representative_points_drac22.csv",
}

DATASET_POINTS_IMAGE_NAME_TEMPLATE = {
    "tu": "nv ({sample_id}).jpg",
    "drac": "{sample_id}.png",
}


def dataset_short_name(dataset: str) -> str:
    """Normalize "dataset_tu"/"dataset_drac"/"tu"/"drac" to "tu"/"drac"."""
    return str(dataset).replace("dataset_", "")


def normalize_points_mode(mode: str | None) -> str:
    """Validate and normalize a points mode, defaulting when blank."""
    if not mode:
        return DEFAULT_POINTS_MODE
    normalized = str(mode).strip().lower()
    if normalized not in POINTS_MODES:
        raise ValueError(
            f"Unknown points_mode {mode!r}; expected one of {', '.join(POINTS_MODES)}"
        )
    return normalized


def resolve_points_csv(dataset: str, mode: str | None = DEFAULT_POINTS_MODE) -> str | None:
    """Return the point CSV for a dataset under the given mode, or None if unmapped."""
    table = DATASET_POINTS_CSV_SINGLE if normalize_points_mode(mode) == "single" else DATASET_POINTS_CSV
    return table.get(dataset_short_name(dataset))
