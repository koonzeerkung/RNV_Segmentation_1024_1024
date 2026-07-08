"""Single source of truth for dataset-specific nearby-filter defaults.

Both `visualize_nearby.py` (auto-resolves when points_csv/points_image_name_template
are left blank) and `run_experiment_queue.py` (fills in per-experiment overrides) read
from these mappings so the two never drift apart.
"""

from __future__ import annotations

DATASET_POINTS_CSV = {
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
