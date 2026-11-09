"""A Dataset's region as a DataFrame: File row numbers, Field names, datatypes, layout groups and the cache."""

import datetime
from pathlib import Path

import openpyxl
import pandas as pd
import pytest
from county_lake import column, county_harness, path_field, table

from ogda.dataset_reader import cache_dataset, column_names, file_column, read_as_text, read_dataset
from ogda.ontology import Harness


def one(tmp_path, text, fields, **region):
    """A Harness with one Dataset `t` on `t.csv`."""
    lake = tmp_path / "lake"
    lake.mkdir(exist_ok=True)
    (lake / "t.csv").write_text(text)
    harness = Harness.empty("toy")
    harness.semantic_model.datasets = [table("t", "t.csv", fields, "A test table", **region)]
    return harness, lake


def test_a_dataset_reads_its_region_with_file_row_numbers_and_file_headers(tmp_path):
    df = read_dataset(county_harness(tmp_path / "lake"), "burned", tmp_path / "lake")
    assert list(df.columns) == ["County", "2020"] and list(df.index) == [1, 2]
    assert df.County.tolist() == ["Coos", "Washoe"] and df["2020"].tolist() == [10.0, 20.0]


def test_blank_rows_are_dropped_and_end_row_ends_the_region(tmp_path):
    harness, lake = one(tmp_path, "v\n1\n,\n2\n3\n", [column("v", "v", "Integer")], end_row=4)
    df = read_dataset(harness, "t", lake)
    assert list(df.index) == [1, 3] and df.v.tolist() == [1, 2]


def test_cells_beyond_the_header_are_dropped(tmp_path):
    harness, lake = one(tmp_path, "a,b\n1,2,\n3,4,\n", [column("a", "a", "Integer"), column("b", "b", "Integer")])
    df = read_dataset(harness, "t", lake)
    assert df.a.tolist() == [1, 3] and df.b.tolist() == [2, 4]


def test_a_column_takes_its_datatype_only_when_every_value_converts(tmp_path):
    harness, lake = one(tmp_path, "a,b,c,d\n1,1,2024-01-02,yes\n,<10,2024-02-03,No\n",
                        [column("a", "a", "Integer"), column("b", "b", "Integer"),
                         column("c", "c", "Date"), column("d", "d", "Boolean")])
    df = read_dataset(harness, "t", lake)
    assert str(df.a.dtype) == "Int64" and df.a.isna().tolist() == [False, True]
    assert df.b.tolist() == ["1", "<10"]
    assert df.c.tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-02-03")]
    assert df.d.tolist() == [True, False]


def test_na_stays_text(tmp_path):
    harness, lake = one(tmp_path, "iso2\nNA\nJP\n", [column("iso2", "iso2")])
    assert read_dataset(harness, "t", lake).iso2.tolist() == ["NA", "JP"]


def test_dates_take_one_format_for_the_whole_column(tmp_path):
    harness, lake = one(tmp_path, "a,b,c\n13/01/2020,2020-01-02,01/13/2020\n01/02/2020,2020-01-03 10:00,2020-01-01\n",
                        [column("a", "a", "Date"), column("b", "b", "DateTime"), column("c", "c", "Date")])
    df = read_dataset(harness, "t", lake)
    assert df.a.tolist() == [pd.Timestamp("2020-01-13"), pd.Timestamp("2020-02-01")]
    assert df.b.tolist() == [pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03 10:00")]
    assert df.c.tolist() == ["01/13/2020", "2020-01-01"]


def test_times_without_a_date_stay_text(tmp_path):
    harness, lake = one(tmp_path, "t\n10:30\n11:00\n", [column("t", "t", "DateTime")])
    assert read_dataset(harness, "t", lake).t.tolist() == ["10:30", "11:00"]


def test_missing_value_tokens_are_empty_before_conversion(tmp_path):
    harness, lake = one(tmp_path, "x,n\n1.5,3\nNaN,NA\nN/A,\nnull,4\n", [column("x", "x", "Decimal"),
                                                                       column("n", "n", "Integer")])
    df = read_dataset(harness, "t", lake)
    assert df.x.dtype == "float64" and df.x.isna().tolist() == [False, True, True, True]
    assert str(df.n.dtype) == "Int64" and df.n.tolist()[0] == 3 and df.n.isna().tolist() == [False, True, True, False]


def test_a_column_that_does_not_convert_keeps_its_text(tmp_path):
    harness, lake = one(tmp_path, "x\n1.5\n-\n", [column("x", "x", "Decimal")])
    assert read_dataset(harness, "t", lake).x.tolist() == ["1.5", "-"]


def test_integers_past_int64_and_codes_with_leading_zeros_stay_text(tmp_path):
    harness, lake = one(tmp_path, "big,code\n99999999999999999999,01001\n1,2\n",
                        [column("big", "big", "Integer"), column("code", "code", "Integer")])
    df = read_dataset(harness, "t", lake)
    assert df.big.tolist() == ["99999999999999999999", "1"] and df.code.tolist() == ["01001", "2"]


def test_repeated_headers_map_to_their_suffixed_names(tmp_path):
    harness, lake = one(tmp_path, "Value,Value\n1,2\n",
                        [column("first", "Value", "Integer"), column("second", "Value.1", "Integer")])
    df = read_dataset(harness, "t", lake)
    assert df["Value"].tolist() == [1] and df["Value.1"].tolist() == [2]


def test_a_headerless_region_names_columns_by_position(tmp_path):
    harness, lake = one(tmp_path, "BRCA1\nTP53\n", [column("gene", "0")], header_row=None, start_row=0)
    df = read_dataset(harness, "t", lake)
    assert df.gene.tolist() == ["BRCA1", "TP53"] and list(df.index) == [0, 1]


def test_a_layout_group_stacks_its_files_with_file_and_path_fields(tmp_path):
    df = read_dataset(county_harness(tmp_path / "lake"), "fires", tmp_path / "lake")
    assert df.file.tolist() == ["fires/NewHampshire.csv", "fires/Nevada.csv"]
    assert df.state.tolist() == ["NewHampshire", "Nevada"] and df.County.tolist() == ["Coos", "Washoe"]


def test_a_layout_group_with_a_file_field_names_its_file_column_freely(tmp_path):
    harness = county_harness(tmp_path / "lake")
    for name in ("fires", "fires__nevada"):
        harness.dataset(name).fields.append(path_field("file", "x", "A Field named file"))
    assert file_column(harness, "fires") == "file_" and file_column(harness, "counties") is None
    df = read_dataset(harness, "fires", tmp_path / "lake")
    assert df.file_.tolist() == ["fires/NewHampshire.csv", "fires/Nevada.csv"] and df.file.tolist() == ["x", "x"]
    harness.dataset("fires").fields.append(path_field("file_", "y", "A Field named file_"))
    assert file_column(harness, "fires") == "file__"


def test_a_copy_member_is_left_out(tmp_path):
    harness = county_harness(tmp_path / "lake")
    (tmp_path / "lake/fires/NevadaCopy.csv").write_text("County\nWashoe\n")
    harness.semantic_model.datasets.append(table(
        "fires__copy", "fires/NevadaCopy.csv", [column("county", "County"), path_field("state", "Nevada", "")],
        "A copy", layout_of="fires", relation="copy", relation_to="fires__nevada"))
    assert "fires/NevadaCopy.csv" not in read_dataset(harness, "fires", tmp_path / "lake").file.tolist()


def test_excel_whole_floats_read_as_integers_and_datetimes_as_dates(tmp_path):
    lake = tmp_path / "lake"
    lake.mkdir()
    workbook = openpyxl.Workbook()
    workbook.active.title = "S"
    workbook.active.append(["n", "day"])
    workbook.active.append([3.0, datetime.datetime(2024, 1, 2)])
    workbook.save(lake / "w.xlsx")
    harness = Harness.empty("toy")
    harness.semantic_model.datasets = [table("w", "w.xlsx", [column("n", "n", "Integer"), column("day", "day", "Date")],
                                             "A sheet", sheet="S")]
    df = read_dataset(harness, "w", lake)
    assert df.n.tolist() == [3] and df.day.tolist() == [pd.Timestamp("2024-01-02")]


def test_a_dataset_in_another_format_is_not_read(tmp_path):
    harness, lake = one(tmp_path, "x\n1\n", [column("x", "x")])
    harness.semantic_model.datasets.append(table("g", "grid.npz", [column("x", "x")], "A grid"))
    assert read_dataset(harness, "g", lake) is None and cache_dataset(harness, "g", lake, tmp_path / "cache") is None


def test_the_cache_reads_a_dataset_once(tmp_path):
    harness = county_harness(tmp_path / "lake")
    path, rows = cache_dataset(harness, "burned", tmp_path / "lake", tmp_path / "cache")
    expected = read_dataset(harness, "burned", tmp_path / "lake")
    (tmp_path / "lake/burned.csv").write_text("County,2020\nCoos,10\n")
    assert rows == 2 and cache_dataset(harness, "burned", tmp_path / "lake", tmp_path / "cache") == (path, 2)
    pd.testing.assert_frame_equal(pd.read_parquet(path), expected)


def test_the_cache_leaves_no_table_when_writing_fails(tmp_path, monkeypatch):
    harness = county_harness(tmp_path / "lake")

    def broken(frame, path, **options):
        Path(path).write_bytes(b"PAR1 partial")
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", broken)
    with pytest.raises(OSError):
        cache_dataset(harness, "burned", tmp_path / "lake", tmp_path / "cache")
    assert not (tmp_path / "cache/burned.parquet").exists()


def test_the_cache_names_the_columns_read_as_text(tmp_path):
    harness, lake = one(tmp_path, "x,y,z\n1.5,2,a\n-,3,b\n",
                        [column("x", "x", "Decimal"), column("y", "y", "Integer"), column("z", "z")])
    path, _ = cache_dataset(harness, "t", lake, tmp_path / "cache")
    assert read_as_text(harness, "t", path) == {"x"}


def test_columns_take_their_file_header(tmp_path):
    df = read_dataset(county_harness(tmp_path / "lake"), "burned", tmp_path / "lake")
    assert list(df.columns) == ["County", "2020"] and df["2020"].tolist() == [10.0, 20.0]


def test_a_header_is_trimmed_and_header_rows_join(tmp_path):
    harness, lake = one(tmp_path, "Fraud ,Year\n Count,\n5,2001\n",
                        [column("fraud", "Fraud / Count", "Integer"), column("year", "Year", "Integer")],
                        header_rows=2, start_row=2)
    assert list(read_dataset(harness, "t", lake).columns) == ["Fraud / Count", "Year"]


def test_literal_path_and_headerless_fields_keep_their_field_names(tmp_path):
    df = read_dataset(county_harness(tmp_path / "lake"), "fires", tmp_path / "lake")
    assert list(df.columns) == ["file", "County", "state"]
    harness, lake = one(tmp_path, "BRCA1\nTP53\n", [column("gene", "0")], header_row=None, start_row=0)
    assert list(read_dataset(harness, "t", lake).columns) == ["gene"]


def test_a_header_another_column_has_falls_back_to_the_field_name(tmp_path):
    harness, lake = one(tmp_path, "a,b\n1,2\n", [column("x", "a"), column("y", "a"), path_field("a", "k", ""),
                                                  column("b", "b")])
    assert column_names(harness, "t") == {"x": "a", "y": "y", "a": "a_", "b": "b"}
    harness = county_harness(tmp_path / "lake2")
    harness.dataset("fires").fields[0] = column("county", "file")
    harness.dataset("fires__nevada").fields[0] = column("county", "file")
    assert column_names(harness, "fires") == {"county": "county", "state": "state"}


def test_read_as_text_names_fields_under_header_columns(tmp_path):
    harness, lake = one(tmp_path, "X,Y\n1.5,2\n-,3\n", [column("x", "X", "Decimal"), column("y", "Y", "Integer")])
    path, _ = cache_dataset(harness, "t", lake, tmp_path / "cache")
    assert read_as_text(harness, "t", path) == {"x"}
