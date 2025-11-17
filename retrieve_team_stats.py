import argparse
from collections import Counter
import re
from pathlib import Path
from io import StringIO
from typing import Iterable, List, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup


TEAM_ID = "746"
YEARS = list(range(2025, 2015, -1))
URL_TEMPLATE = "https://cfbstats.com/{year}/team/{team_id}/index.html"

# Default output directory is now 'data/scraped'
DEFAULT_OUTPUT_DIR = Path("data/scraped")


def _slugify(value: str) -> str:
    """
    Convert a string into a filesystem-friendly slug.
    """
    value = value.strip().lower()
    value = re.sub(r"[^\w\s-]", "", value)
    value = re.sub(r"[\s-]+", "_", value)
    return value or "table"


def _normalize_header(value: str, index: int) -> str:
    value = re.sub(r"\s+", " ", str(value)).strip()
    if not value or re.match(r"^Unnamed", value, re.IGNORECASE):
        value = f"column_{index}"
    return value


def _normalize_cell(value):
    if isinstance(value, str):
        value = re.sub(r"\s+", " ", value).strip()
    return value


def _row_is_noise(row: pd.Series) -> bool:
    values = [
        str(val).strip().lower()
        for val in row
        if not (pd.isna(val) or str(val).strip() == "")
    ]
    if not values:
        return True

    unique_values = set(values)
    if len(unique_values) == 1:
        value = next(iter(unique_values))
        if any(token in value for token in ("note", "legend", "neutral site", "see ", "forfeit", "@ :")):
            return True
    return False


def _convert_series(series: pd.Series) -> pd.Series:
    if series.dtype != object:
        return series

    try:
        cleaned = series.str.replace(",", "", regex=False)
        cleaned = cleaned.str.replace("%", "", regex=False)

        converted = pd.to_numeric(cleaned)
        # Leave as numeric only if it introduces some numeric data.
        if converted.notna().sum() > 0:
            return converted
    except (ValueError, AttributeError):
        pass

    return series


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_normalize_header(col, idx) for idx, col in enumerate(df.columns)]
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.apply(lambda col: col.map(_normalize_cell) if col.dtype == object else col)
    df.replace({"": pd.NA, "—": pd.NA, "–": pd.NA}, inplace=True)
    df = df.dropna(how="all")
    df = df.loc[~df.apply(_row_is_noise, axis=1)]
    df = df.apply(_convert_series, axis=0)
    df = df.reset_index(drop=True)
    return df


def _extract_tables(html: str) -> List[Tuple[str, pd.DataFrame]]:
    """
    Parse tables from the HTML document, pairing each table with the closest
    preceding heading (if available) to use as the table name.
    """
    soup = BeautifulSoup(html, "html.parser")
    headings_counter: Counter[str] = Counter()
    tables: List[Tuple[str, pd.DataFrame]] = []

    for table in soup.find_all("table"):
        df_list = pd.read_html(StringIO(str(table)))
        if not df_list:
            continue

        df = _clean_dataframe(df_list[0])
        if df.empty:
            continue

        heading_name = None
        for sibling in table.find_all_previous():
            if sibling.name and sibling.name.lower() in {"h1", "h2", "h3", "h4", "h5", "h6", "strong"}:
                heading_name = sibling.get_text(strip=True)
                break

        slug = _slugify(heading_name or "table")
        count = headings_counter[slug]
        headings_counter[slug] += 1

        if count:
            slug = f"{slug}_{count + 1}"

        tables.append((slug, df))

    return tables


def save_tables(tables: Iterable[Tuple[str, pd.DataFrame]], output_dir: Path, basename: str) -> List[Path]:
    """
    Write the tables to CSV files and return the file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    for idx, (slug, df) in enumerate(tables, start=1):
        filename = f"{basename}_{slug}.csv"
        path = output_dir / filename
        df.to_csv(path, index=False)
        saved_paths.append(path)

    return saved_paths


def fetch_year(year: int, team_id: str, output_dir: Path) -> List[Path]:
    url = URL_TEMPLATE.format(year=year, team_id=team_id)

    response = requests.get(url, timeout=20)
    response.raise_for_status()

    tables = _extract_tables(response.text)

    if not tables:
        raise ValueError(f"No tables found for {url}")

    basename = f"{year}_{_slugify(Path(url).stem or 'cfbstats')}"
    
    # Delete old files that match the pattern up to _through
    # Files are named like: {year}_index_{year}_{team}_through_{date}.csv
    # We want to delete all variants (including _2, _3 suffixes) for files matching up to _through
    # Try to be team-specific by checking existing files for this year
    existing_files = list(output_dir.glob(f"{year}_index_{year}_*_through_*.csv"))
    deleted_count = 0
    
    if existing_files:
        # Extract team pattern from first existing file to make deletion team-specific
        # Example: "2025_index_2025_temple_owls_through_11152025.csv"
        # Pattern: 2025_index_2025_(temple_owls)_through_
        match = re.search(rf"^{year}_index_{year}_([^_]+(?:_[^_]+)*?)_through_", existing_files[0].name)
        if match:
            team_pattern = match.group(1)
            # Delete files matching this specific team pattern (all date variants)
            pattern = f"{year}_index_{year}_{team_pattern}_through_*.csv"
            for old_file in output_dir.glob(pattern):
                old_file.unlink()
                deleted_count += 1
        else:
            # Fallback: if pattern doesn't match, delete all files for this year
            for old_file in existing_files:
                old_file.unlink()
                deleted_count += 1
    
    if deleted_count > 0:
        print(f"  Deleted {deleted_count} old file(s) matching pattern up to '_through'")
    
    return save_tables(tables, output_dir, basename)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download tables from a cfbstats.com page and save them as CSV files."
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where CSV files will be saved (default: data/scraped).",
    )
    parser.add_argument(
        "--team-id",
        default=TEAM_ID,
        help=f"cfbstats team identifier (default: {TEAM_ID}).",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=max(YEARS),
        help=f"First season to fetch (default: {max(YEARS)}).",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=min(YEARS),
        help=f"Last season to fetch (inclusive, default: {min(YEARS)}).",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    team_id = args.team_id
    start_year = args.start_year
    end_year = args.end_year

    if start_year >= end_year:
        years = range(start_year, end_year - 1, -1)
    else:
        years = range(start_year, end_year + 1)

    all_saved: List[Path] = []

    for year in years:
        url = URL_TEMPLATE.format(year=year, team_id=team_id)
        print(f"Fetching {url} ...")

        try:
            saved_paths = fetch_year(year, team_id, output_dir)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"  Failed: {exc}")
            continue

        for path in saved_paths:
            all_saved.append(path)

        print(f"  Saved {len(saved_paths)} tables.")

    if not all_saved:
        raise SystemExit("No tables were saved.")

    print("All saved tables:")
    for path in all_saved:
        print(f"- {path}")


if __name__ == "__main__":
    main()
