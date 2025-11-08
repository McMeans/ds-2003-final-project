import argparse
from collections import Counter
import re
from pathlib import Path
from io import StringIO
from typing import Iterable, List, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup


CFBSTATS_URL = "https://cfbstats.com/2025/team/746/index.html"


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download tables from a cfbstats.com page and save them as CSV files."
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory where CSV files will be saved (default: data).",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()

    response = requests.get(CFBSTATS_URL, timeout=20)
    response.raise_for_status()

    tables = _extract_tables(response.text)

    if not tables:
        raise SystemExit("No tables found at the provided URL.")

    basename = _slugify(Path(CFBSTATS_URL).stem or "cfbstats")
    saved_paths = save_tables(tables, output_dir, basename)

    print("Saved tables:")
    for path in saved_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()

