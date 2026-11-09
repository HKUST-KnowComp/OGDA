"""A Dataset's rows as a typed DataFrame: File row numbers for the index, the File's headers for the columns."""

import re
import warnings
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pandas._libs.parsers import STR_NA_VALUES
from pandas.tseries.api import guess_datetime_format

from .binding import field_source, header_text
from .lake import TABULAR, read_rows
from .ontology import Harness
from .ossie import ogda_extension

BOOLEANS = {"true": True, "yes": True, "1": True, "false": False, "no": False, "0": False}
CONVERTED = {"Integer", "Decimal", "Float", "Date", "DateTime", "Boolean"}  # Datatypes read as other than text.
MISSING = frozenset(STR_NA_VALUES)  # pandas' read_csv missing-value tokens: "NA", "NaN", "N/A", "null", ...
LEADING_ZERO = re.compile(r"[+-]?0\d")  # A code such as "01001", which a number would turn into 1001.


def readable(harness: Harness, name: str) -> bool:
    """Whether code reads the Dataset: its File is a text table or a workbook."""
    return Path(ogda_extension(harness.dataset(name)).path).suffix.lower() in TABULAR


def convert(values: pd.Series, datatype: str) -> pd.Series:
    """A column of text as its datatype when every nonempty value converts, read_csv's missing-value tokens counting
    as empty; otherwise the text, empty cells missing."""
    values = values.replace("", pd.NA)
    if datatype in CONVERTED:
        try:
            converted = typed(values.mask(values.isin(MISSING)), datatype)
        except (ValueError, TypeError, OverflowError):  # A value that does not convert, or an integer past int64.
            converted = None
        if converted is not None:
            return converted
    return values.astype("str")


def typed(values: pd.Series, datatype: str) -> pd.Series | None:
    """The column as its datatype, or None when a value keeps it text; conversion errors propagate."""
    nonempty = values.dropna()
    if datatype == "Integer":
        if nonempty.str.match(LEADING_ZERO).any():
            return None
        numbers = pd.to_numeric(values, errors="raise")
        return numbers.astype("Int64") if (numbers.dropna() % 1 == 0).all() else None
    if datatype in ("Decimal", "Float"):
        return pd.to_numeric(values, errors="raise").astype("float64")
    if datatype in ("Date", "DateTime"):
        # One format for the whole column, guessed from its first value as pandas does, else ISO 8601; a time with
        # no date has no format and stays text rather than taking today's date.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # Its note that it guessed day first.
            guessed = guess_datetime_format(nonempty.iloc[0]) if len(nonempty) else None
        for form in ([guessed] if guessed else []) + ["ISO8601"]:
            try:
                return pd.to_datetime(values, format=form)
            except ValueError:
                pass
        return None
    if datatype == "Boolean" and nonempty.str.lower().isin(BOOLEANS).all():
        flags = values.str.lower().map(BOOLEANS)
        return flags.astype("boolean") if values.isna().any() else flags.astype(bool)
    return None


def file_column(harness: Harness, name: str) -> str | None:
    """The column naming each row's File in a layout group's table: `file`, or with underscores added until no Field
    has the name; None for a Dataset with no layout group."""
    if not harness.members(name):
        return None
    taken, column = {field.name for field in harness.dataset(name).fields}, "file"
    while column in taken:
        column += "_"
    return column


def column_names(harness: Harness, name: str) -> dict[str, str]:
    """Each Field's column name in the Dataset's DataFrame, in Field order: the header text of the column it reads,
    or its Field name when it reads none (a literal, a value from the File path, a region with no header row) or when
    another column has that name already (the File column, then earlier Fields); `_` is added to a taken Field name."""
    header_row = ogda_extension(harness.dataset(name)).header_row
    taken = {column} if (column := file_column(harness, name)) else set()
    names = {}
    for field in harness.dataset(name).fields:
        kind, text = field_source(field)
        candidate = text.strip() if kind == "column" and header_row is not None else field.name
        if candidate in taken:
            candidate = field.name
        while candidate in taken:
            candidate += "_"
        taken.add(candidate)
        names[field.name] = candidate
    return names


def read_as_text(harness: Harness, name: str, table: Path) -> set[str]:
    """The Fields whose datatype converts but whose column in the cached table stayed text."""
    schema, names = pq.read_schema(table), column_names(harness, name)
    return {field.name for field in harness.dataset(name).fields if field.datatype in CONVERTED
            and names[field.name] in schema.names
            and (pa.types.is_string(schema.field(names[field.name]).type)
                 or pa.types.is_large_string(schema.field(names[field.name]).type))}


def read_region(harness: Harness, name: str, lake_root: Path) -> pd.DataFrame:
    """One Dataset's rows as text, columns named by Field, indexed by File row number."""
    dataset = harness.dataset(name)
    region = ogda_extension(dataset)
    rows = read_rows(Path(lake_root) / region.path, region.sheet)
    header = header_text(rows, region.header_row, region.header_rows)
    end = len(rows) if region.end_row is None else region.end_row
    data = [(number, [cell.strip() for cell in rows[number]][:len(header)] + [""] * (len(header) - len(rows[number])))
            for number in range(region.start_row, min(end, len(rows)))]
    data = [(number, values) for number, values in data if any(values)]  # Blank rows hold no data.
    frame = pd.DataFrame([values for _, values in data], index=[number for number, _ in data],
                         columns=header, dtype=object)
    columns = {}
    for field in dataset.fields:
        kind, text = field_source(field)
        columns[field.name] = (frame[text.strip()] if kind == "column"
                               else pd.Series(text, index=frame.index, dtype=object))
    return pd.DataFrame(columns, index=frame.index)


def read_dataset(harness: Harness, name: str, lake_root: Path) -> pd.DataFrame | None:
    """A Dataset's rows, stacked over its layout group with a column naming each row's File (see `file_column`), its
    columns named by `column_names`; None for Files code does not read."""
    if not readable(harness, name):
        return None
    group, column = harness.members(name), file_column(harness, name)
    parts = [name] + [member for member in group if ogda_extension(harness.dataset(member)).relation == "part"]
    frames = []
    for part in parts:
        frame = read_region(harness, part, lake_root)
        if column:
            frame.insert(0, column, ogda_extension(harness.dataset(part)).path)
        frames.append(frame)
    stacked = pd.concat(frames)
    for field in harness.dataset(name).fields:
        stacked[field.name] = convert(stacked[field.name], field.datatype)
    return stacked.rename(columns=column_names(harness, name))


def cache_dataset(harness: Harness, name: str, lake_root: Path, cache: Path) -> tuple[Path, int] | None:
    """Write the Dataset's DataFrame to `cache/<name>.parquet` once; its path and row count, None when not readable."""
    if not readable(harness, name):
        return None
    path = Path(cache) / f"{name}.parquet"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.tmp")  # Renamed once whole, so a failed write leaves no table.
        try:
            read_dataset(harness, name, lake_root).to_parquet(temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return path, pq.read_metadata(path).num_rows
