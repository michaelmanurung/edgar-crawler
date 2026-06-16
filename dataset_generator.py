"""
Dataset Generation Pipeline for EDGAR-Crawler.

Converts extracted SEC filing JSONs (10-K, 10-Q, 8-K) into sharded,
HuggingFace-compatible JSONL datasets with train/validation/test splits.

Usage:
    python dataset_generator.py                        # use defaults from config.json
    python dataset_generator.py --config custom.json   # custom config file
    python dataset_generator.py -v                     # info-level logging
    python dataset_generator.py -vv                    # debug-level logging
"""

import csv
import hashlib
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logger = logging.getLogger("dataset_generator")


def configure_logging(verbosity: int) -> None:
    """Configure logging level based on verbosity count.

    Args:
        verbosity: 0 = WARNING (minimal), 1 = INFO, 2+ = DEBUG.
    """
    levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    level = levels.get(verbosity, logging.DEBUG)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(handler)
    logger.setLevel(level)


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict = {
    "dataset_generation": {
        "extracted_filings_folder": "datasets/EXTRACTED_FILINGS",
        "filings_metadata_file": "datasets/FILINGS_METADATA.csv",
        "output_dir": "shards",
        "filing_types": ["10-K", "10-Q", "8-K"],
        "shard_max_size_mb": 100,
        "split_ratios": {"train": 0.90, "validation": 0.05, "test": 0.05},
        "random_seed": 42,
        "progress_file": "dataset_generation_progress.json",
        "verbosity": 0,
    }
}


def load_config(config_path: str = "config.json") -> dict:
    """Load configuration from JSON file, merging with defaults.

    Args:
        config_path: Path to the config JSON file.

    Returns:
        Merged configuration dictionary (dataset_generation section).
    """
    config = DEFAULT_CONFIG["dataset_generation"].copy()
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            file_config = json.load(f)
        user_section = file_config.get("dataset_generation", {})
        config.update(user_section)
    else:
        logger.warning("config.json not found at %s, using all defaults", config_path)
    return config


# ---------------------------------------------------------------------------
# Field schema definitions
# ---------------------------------------------------------------------------

# Standard 10-K section fields (20 sections, in order)
_BASE_SECTIONS: List[str] = [
    "section_1", "section_1A", "section_1B",
    "section_2", "section_3", "section_4", "section_5",
    "section_6", "section_7", "section_7A", "section_8",
    "section_9", "section_9A", "section_9B",
    "section_10", "section_11", "section_12",
    "section_13", "section_14", "section_15",
]

# 10-Q additional descriptive fields (unique content not in base schema)
_10Q_EXTRA_SECTIONS: List[str] = [
    "section_unregistered_sales_equity",
    "section_defaults_senior_securities",
]

# 8-K descriptive fields — one per SEC item, all unique (non-duplicate) content
_8K_DESCRIPTIVE_SECTIONS: List[str] = [
    "section_entry_material_definitive_agreement",
    "section_termination_material_definitive_agreement",
    "section_bankruptcy_receivership",
    "section_mine_safety_reporting",
    "section_material_cybersecurity_incidents",
    "section_completion_acquisition_disposition",
    "section_results_operations_financial",
    "section_direct_financial_obligation",
    "section_triggering_events_accelerate_obligation",
    "section_costs_exit_disposal",
    "section_material_impairments",
    "section_notice_delisting_transfer_listing",
    "section_unregistered_sales_equity_securities",
    "section_material_modification_rights",
    "section_changes_certifying_accountant",
    "section_non_reliance_financial_statements",
    "section_changes_control_registrant",
    "section_departure_directors_officers",
    "section_amendments_articles_bylaws",
    "section_suspension_employee_benefit_plans",
    "section_amendments_code_ethics",
    "section_change_shell_company_status",
    "section_submission_matters_vote",
    "section_shareholder_director_nominations",
    "section_abs_informational_material",
    "section_change_servicer_trustee",
    "section_change_credit_enhancement",
    "section_failure_required_distribution",
    "section_securities_act_updating_disclosure",
    "section_regulation_fd_disclosure",
    "section_other_events",
    "section_financial_statements_exhibits",
]

# The full output field list for each filing type
_METADATA_FIELDS: List[str] = ["filename", "cik", "year"]

_SCHEMA_BY_TYPE: Dict[str, List[str]] = {
    "10-K": _METADATA_FIELDS + _BASE_SECTIONS,
    "10-Q": _METADATA_FIELDS + _BASE_SECTIONS + _10Q_EXTRA_SECTIONS,
    "8-K":  _METADATA_FIELDS + _BASE_SECTIONS + _8K_DESCRIPTIVE_SECTIONS,
}

# ---------------------------------------------------------------------------
# Section mapping tables
# ---------------------------------------------------------------------------

# 10-K: direct item_N -> section_N. Items not in output schema are excluded.
# Keys present in 10-K JSON that we DROP: item_1C, item_9C, item_16, SIGNATURE
_10K_MAPPING: Dict[str, str] = {
    "item_1": "section_1",
    "item_1A": "section_1A",
    "item_1B": "section_1B",
    "item_2": "section_2",
    "item_3": "section_3",
    "item_4": "section_4",
    "item_5": "section_5",
    "item_6": "section_6",
    "item_7": "section_7",
    "item_7A": "section_7A",
    "item_8": "section_8",
    "item_9": "section_9",
    "item_9A": "section_9A",
    "item_9B": "section_9B",
    "item_10": "section_10",
    "item_11": "section_11",
    "item_12": "section_12",
    "item_13": "section_13",
    "item_14": "section_14",
    "item_15": "section_15",
}

# 10-Q: semantic mapping. Aggregate keys (part_1, part_2) are excluded
# because they duplicate individual item content.
_10Q_MAPPING: Dict[str, str] = {
    "part_1_item_1": "section_8",
    "part_1_item_2": "section_7",
    "part_1_item_3": "section_7A",
    "part_1_item_4": "section_9A",
    "part_2_item_1": "section_3",
    "part_2_item_1A": "section_1A",
    "part_2_item_2": "section_unregistered_sales_equity",
    "part_2_item_3": "section_defaults_senior_securities",
    "part_2_item_4": "section_4",
    "part_2_item_5": "section_9B",
    "part_2_item_6": "section_15",
}

# 8-K: all item_X.XX -> descriptive field names.
# Every 8-K item is unique content; nothing is dropped.
_8K_MAPPING: Dict[str, str] = {
    "item_1.01": "section_entry_material_definitive_agreement",
    "item_1.02": "section_termination_material_definitive_agreement",
    "item_1.03": "section_bankruptcy_receivership",
    "item_1.04": "section_mine_safety_reporting",
    "item_1.05": "section_material_cybersecurity_incidents",
    "item_2.01": "section_completion_acquisition_disposition",
    "item_2.02": "section_results_operations_financial",
    "item_2.03": "section_direct_financial_obligation",
    "item_2.04": "section_triggering_events_accelerate_obligation",
    "item_2.05": "section_costs_exit_disposal",
    "item_2.06": "section_material_impairments",
    "item_3.01": "section_notice_delisting_transfer_listing",
    "item_3.02": "section_unregistered_sales_equity_securities",
    "item_3.03": "section_material_modification_rights",
    "item_4.01": "section_changes_certifying_accountant",
    "item_4.02": "section_non_reliance_financial_statements",
    "item_5.01": "section_changes_control_registrant",
    "item_5.02": "section_departure_directors_officers",
    "item_5.03": "section_amendments_articles_bylaws",
    "item_5.04": "section_suspension_employee_benefit_plans",
    "item_5.05": "section_amendments_code_ethics",
    "item_5.06": "section_change_shell_company_status",
    "item_5.07": "section_submission_matters_vote",
    "item_5.08": "section_shareholder_director_nominations",
    "item_6.01": "section_abs_informational_material",
    "item_6.02": "section_change_servicer_trustee",
    "item_6.03": "section_change_credit_enhancement",
    "item_6.04": "section_failure_required_distribution",
    "item_6.05": "section_securities_act_updating_disclosure",
    "item_7.01": "section_regulation_fd_disclosure",
    "item_8.01": "section_other_events",
    "item_9.01": "section_financial_statements_exhibits",
}

_MAPPING_BY_TYPE: Dict[str, Dict[str, str]] = {
    "10-K": _10K_MAPPING,
    "10-Q": _10Q_MAPPING,
    "8-K": _8K_MAPPING,
}


def map_sections(filing_json: dict, filing_type: str) -> dict:
    """Map extracted filing item keys to the unified output section schema.

    Args:
        filing_json: The parsed extracted filing JSON.
        filing_type: One of '10-K', '10-Q', '8-K'.

    Returns:
        Dict with section field names as keys and text content as values.
        Missing sections are filled with empty strings.
    """
    schema = _SCHEMA_BY_TYPE[filing_type]
    mapping = _MAPPING_BY_TYPE.get(filing_type, {})

    # Initialize all section fields as empty strings
    record: dict = {field: "" for field in schema}

    # Populate metadata
    record["filename"] = filing_json.get("filename", "")
    record["cik"] = str(filing_json.get("cik", ""))
    # year extracted from period_of_report (e.g., "2020-12-31" -> "2020")
    period = filing_json.get("period_of_report", "")
    record["year"] = str(period)[:4] if period else ""

    # Map items to sections
    for item_key, section_key in mapping.items():
        content = filing_json.get(item_key, "")
        if content and isinstance(content, str) and content.strip():
            record[section_key] = content

    return record


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------

def assign_split(
    cik: str,
    filing_type: str,
    period_of_report: str,
    seed: int,
    split_ratios: Dict[str, float],
) -> str:
    """Deterministically assign a filing to train/validation/test.

    Uses MD5 hashing of (cik, filing_type, period_of_report, seed) to produce
    a stable float in [0, 1). The same tuple always maps to the same bucket.

    Args:
        cik: Company CIK identifier.
        filing_type: '10-K', '10-Q', or '8-K'.
        period_of_report: Period of report date string (e.g., '2020-12-31').
        seed: Random seed for the hash.
        split_ratios: Dict with 'train', 'validation', 'test' float ratios
                      that should sum to 1.0.

    Returns:
        One of 'train', 'validation', or 'test'.
    """
    hash_input = f"{cik}|{filing_type}|{period_of_report}|{seed}"
    digest = hashlib.md5(hash_input.encode("utf-8")).hexdigest()
    # Normalize to [0, 1)
    value = int(digest, 16) / (16 ** len(digest))
    if value < split_ratios["train"]:
        return "train"
    elif value < split_ratios["train"] + split_ratios["validation"]:
        return "validation"
    else:
        return "test"


# ---------------------------------------------------------------------------
# Filing discovery from CSV
# ---------------------------------------------------------------------------

def discover_filings(
    metadata_csv_path: Path,
    filing_types: List[str],
) -> List[dict]:
    """Read the filings metadata CSV and return a list of filing descriptors.

    Only returns filings whose type is in the configured filing_types list.

    Args:
        metadata_csv_path: Path to FILINGS_METADATA.csv.
        filing_types: List of filing types to include (e.g., ['10-K', '10-Q']).

    Returns:
        List of dicts with keys: cik, filing_type, period_of_report, filename.
    """
    if not metadata_csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv_path}")

    filings: List[dict] = []
    type_set = set(filing_types)

    with open(metadata_csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ftype = row.get("Type", "").strip()
            if ftype not in type_set:
                continue
            cik = row.get("CIK", "").strip()
            period = row.get("Period of Report", "").strip()
            filename = row.get("filename", "").strip()
            if not cik or not period or not filename:
                logger.debug("Skipping CSV row with missing fields: %s", row)
                continue
            filings.append({
                "cik": cik,
                "filing_type": ftype,
                "period_of_report": period,
                "filename": filename,
            })

    logger.info(
        "Discovered %d filings across types %s from %s",
        len(filings), filing_types, metadata_csv_path,
    )
    return filings


# ---------------------------------------------------------------------------
# Progress tracking (resumability)
# ---------------------------------------------------------------------------

class ProgressTracker:
    """Tracks completed filings for resumability.

    Stores a set of unique keys (one per filing) in a JSON file.
    On restart, loads the set and skips already-processed filings.
    """

    def __init__(self, progress_path: Path, flush_interval: int = 100):
        """Initialize the progress tracker.

        Args:
            progress_path: Path to the progress JSON file.
            flush_interval: How many filings to process between writes.
        """
        self.progress_path = progress_path
        self.flush_interval = flush_interval
        self._completed: set = set()
        self._counter: int = 0

        if progress_path.exists():
            try:
                with open(progress_path, "r") as f:
                    data = json.load(f)
                self._completed = set(data.get("completed_filings", []))
                logger.info(
                    "Loaded %d completed filings from progress file",
                    len(self._completed),
                )
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning("Could not parse progress file, starting fresh: %s", exc)
                self._completed = set()

    def is_completed(self, key: str) -> bool:
        """Check if a filing has already been processed."""
        return key in self._completed

    def mark_completed(self, key: str) -> None:
        """Record a filing as processed. Flushes periodically."""
        self._completed.add(key)
        self._counter += 1
        if self._counter % self.flush_interval == 0:
            self._flush()

    def _flush(self) -> None:
        """Write the completed set to disk."""
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.progress_path, "w") as f:
            json.dump({"completed_filings": sorted(self._completed)}, f)

    def finalize(self) -> None:
        """Flush final state to disk."""
        self._flush()

    def __len__(self) -> int:
        return len(self._completed)


# ---------------------------------------------------------------------------
# Shard writer
# ---------------------------------------------------------------------------

class ShardWriter:
    """Manages writing JSONL records to sharded output files.

    Automatically rotates to a new shard when the current file exceeds
    the configured size limit. Tracks one file per (year, filing_type, split).
    """

    def __init__(self, output_dir: Path, shard_max_size_bytes: int):
        """Initialize the shard writer.

        Args:
            output_dir: Root output directory (e.g., Path('shards')).
            shard_max_size_bytes: Maximum size of a single shard file in bytes.
        """
        self.output_dir = output_dir
        self.max_size = shard_max_size_bytes
        # open files keyed by (year, filing_type, split)
        self._handles: Dict[Tuple[str, str, str], Tuple[int, Path, object]] = {}
        self._sizes: Dict[Tuple[str, str, str], int] = defaultdict(int)
        self._shard_indices: Dict[Tuple[str, str, str], int] = defaultdict(int)
        # Track stats
        self.shard_count: int = 0
        self.total_records: int = 0

    def _get_shard_path(
        self, year: str, filing_type: str, split: str, shard_index: int
    ) -> Path:
        """Build the output path for a shard file."""
        return (
            self.output_dir
            / year
            / filing_type
            / split
            / f"shard_{shard_index:03d}.jsonl"
        )

    def _open_shard(
        self, year: str, filing_type: str, split: str, shard_index: int
    ) -> object:
        """Open (or reopen) a shard file for writing."""
        path = self._get_shard_path(year, filing_type, split, shard_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(path, "w", encoding="utf-8")
        logger.debug("Opened shard: %s", path)
        self.shard_count = max(self.shard_count, shard_index + 1)
        return fh

    def write_record(
        self, year: str, filing_type: str, split: str, record: dict
    ) -> None:
        """Write a single JSONL record to the appropriate shard.

        Rotates to a new shard if the current one would exceed max_size.

        Args:
            year: The period_of_report year (e.g., '2021').
            filing_type: '10-K', '10-Q', or '8-K'.
            split: 'train', 'validation', or 'test'.
            record: The mapped output record dict.
        """
        key = (year, filing_type, split)
        idx = self._shard_indices[key]

        # Serialize the line to measure its size
        line = json.dumps(record, ensure_ascii=False) + "\n"
        line_bytes = line.encode("utf-8")

        # Get or create the current file handle
        if key not in self._handles:
            self._handles[key] = self._open_shard(year, filing_type, split, idx)
            self._sizes[key] = 0

        # Rotate if needed
        current_size = self._sizes[key]
        if current_size > 0 and current_size + len(line_bytes) > self.max_size:
            self._handles[key].close()
            idx += 1
            self._shard_indices[key] = idx
            self._handles[key] = self._open_shard(year, filing_type, split, idx)
            self._sizes[key] = 0
            logger.info(
                "Rotating shard for %s/%s/%s -> shard_%03d.jsonl",
                year, filing_type, split, idx,
            )

        self._handles[key].write(line)
        self._sizes[key] += len(line_bytes)
        self.total_records += 1

    def close_all(self) -> None:
        """Close all open shard file handles."""
        for (year, ftype, split), fh in self._handles.items():
            fh.close()
            logger.debug(
                "Closed shard %s/%s/%s/shard_%03d.jsonl (%d bytes)",
                year, ftype, split,
                self._shard_indices.get((year, ftype, split), 0),
                self._sizes.get((year, ftype, split), 0),
            )
        self._handles.clear()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate_dataset(config: dict) -> dict:
    """Execute the full dataset generation pipeline.

    Args:
        config: Dataset generation configuration dictionary.

    Returns:
        Statistics dict with keys: total_processed, total_written, total_skipped,
        skipped_corrupted, skipped_completed, shards_created, splits_by_type.
    """
    extracted_dir = Path(config["extracted_filings_folder"])
    metadata_csv = Path(config["filings_metadata_file"])
    output_dir = Path(config["output_dir"])
    filing_types = config["filing_types"]
    shard_max_bytes = int(config["shard_max_size_mb"] * 1024 * 1024)
    split_ratios = config["split_ratios"]
    seed = config["random_seed"]
    progress_path = Path(config["progress_file"])

    # Validate split ratios
    ratio_sum = split_ratios["train"] + split_ratios["validation"] + split_ratios["test"]
    if abs(ratio_sum - 1.0) > 0.001:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    # Discover filings from CSV
    filings = discover_filings(metadata_csv, filing_types)
    if not filings:
        logger.warning("No filings discovered. Check metadata CSV and filing_types config.")
        return {}

    # Initialize components
    progress = ProgressTracker(progress_path)
    writer = ShardWriter(output_dir, shard_max_bytes)

    # Statistics
    stats = {
        "total_discovered": len(filings),
        "total_processed": 0,
        "total_written": 0,
        "skipped_completed": 0,
        "skipped_corrupted": 0,
        "skipped_missing_file": 0,
        "by_type": defaultdict(lambda: {"train": 0, "validation": 0, "test": 0}),
    }

    # Pre-compute splits for all filings (deterministic; no I/O needed)
    filing_assignments: List[Tuple[str, dict]] = []
    for filing in filings:
        year = filing["period_of_report"][:4]
        key = f"{year}|{filing['filing_type']}|{filing['filename']}"

        if progress.is_completed(key):
            stats["skipped_completed"] += 1
            continue

        split = assign_split(
            filing["cik"],
            filing["filing_type"],
            filing["period_of_report"],
            seed,
            split_ratios,
        )
        filing_assignments.append((key, {**filing, "year": year, "split": split}))

    logger.info(
        "Assignments complete: %d to process, %d already completed",
        len(filing_assignments), stats["skipped_completed"],
    )

    # Stream process each filing
    for progress_key, filing in tqdm(
        filing_assignments,
        desc="Processing filings",
        unit="filing",
        disable=logger.level > logging.WARNING,
    ):
        ftype = filing["filing_type"]
        filename_htm = filing["filename"]
        # The extracted JSON filename replaces .htm with .json
        filename_json = filename_htm.rsplit(".", 1)[0] + ".json"
        json_path = extracted_dir / ftype / filename_json

        try:
            # Skip if JSON file doesn't exist on disk
            if not json_path.exists():
                logger.debug("JSON not found, skipping: %s", json_path)
                stats["skipped_missing_file"] += 1
                continue

            # Read one filing at a time (streaming)
            with open(json_path, "r", encoding="utf-8") as jf:
                filing_json = json.load(jf)

            # Map to output schema
            record = map_sections(filing_json, ftype)

            # Write to appropriate shard
            writer.write_record(
                filing["year"], ftype, filing["split"], record
            )

            # Mark as completed for resumability
            progress.mark_completed(progress_key)

            stats["total_processed"] += 1
            stats["total_written"] += 1
            stats["by_type"][ftype][filing["split"]] += 1

        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.warning("Corrupted JSON, skipping %s: %s", json_path, exc)
            stats["skipped_corrupted"] += 1
        except Exception as exc:
            logger.error("Unexpected error processing %s: %s", json_path, exc)
            stats["skipped_corrupted"] += 1

    # Cleanup
    writer.close_all()
    progress.finalize()

    stats["shards_created"] = writer.shard_count
    stats["total_records_written"] = writer.total_records

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def print_summary(stats: dict) -> None:
    """Print a human-readable summary of the generation run."""
    if not stats:
        click.echo("No filings were processed.")
        return

    click.echo("\n" + "=" * 60)
    click.echo("  Dataset Generation Summary")
    click.echo("=" * 60)
    click.echo(f"  Filings discovered:    {stats['total_discovered']:>8d}")
    click.echo(f"  Already completed:     {stats['skipped_completed']:>8d}")
    click.echo(f"  Processed this run:    {stats['total_processed']:>8d}")
    click.echo(f"  Records written:       {stats['total_records_written']:>8d}")
    click.echo(f"  Missing JSON files:    {stats['skipped_missing_file']:>8d}")
    click.echo(f"  Corrupted/skipped:     {stats['skipped_corrupted']:>8d}")
    click.echo(f"  Total shards created:  {stats['shards_created']:>8d}")
    click.echo()

    by_type = stats.get("by_type", {})
    if by_type:
        click.echo("  Records per type/split:")
        for ftype in sorted(by_type.keys()):
            counts = by_type[ftype]
            click.echo(
                f"    {ftype:6s}  "
                f"train: {counts['train']:>6d}  "
                f"validation: {counts['validation']:>5d}  "
                f"test: {counts['test']:>5d}  "
                f"total: {sum(counts.values()):>6d}"
            )
    click.echo("=" * 60 + "\n")


@click.command()
@click.option(
    "--config", "-c",
    default="config.json",
    help="Path to config JSON file.",
    type=click.Path(exists=False),
)
@click.option(
    "--verbose", "-v",
    count=True,
    help="Increase verbosity. -v for INFO, -vv for DEBUG.",
)
def main(config: str, verbose: int) -> None:
    """Generate sharded JSONL datasets from extracted SEC filings.

    Reads filing metadata from a CSV, assigns train/validation/test splits
    deterministically, and writes sharded JSONL output compatible with
    HuggingFace datasets.
    """
    # Load config first to check for stored verbosity
    cfg = load_config(config)

    # CLI verbosity flag overrides config verbosity
    verbosity = verbose if verbose > 0 else cfg.get("verbosity", 0)
    configure_logging(verbosity)

    logger.warning("Dataset generation started (verbosity=%d)", verbosity)
    logger.info("Using config from: %s", config)
    logger.debug("Full configuration: %s", json.dumps(cfg, indent=2))

    try:
        stats = generate_dataset(cfg)
        print_summary(stats)
    except Exception as exc:
        logger.error("Dataset generation failed: %s", exc)
        raise

    logger.warning("Dataset generation complete.")


if __name__ == "__main__":
    main()
