import json

from ogda.lake import copy_lake


def test_copy_lake_strips_keys_from_cp1252_json(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "metadata_0.json").write_bytes('{"name": "café ²", "queries": [1]}'.encode("cp1252"))
    copy_lake(source, tmp_path / "copy", stripped_keys=["queries"])
    assert json.loads((tmp_path / "copy" / "metadata_0.json").read_text(encoding="utf-8")) == {"name": "café ²"}


def test_a_row_reader_reads_a_table_again_only_for_more_rows(tmp_path, monkeypatch):
    import ogda.lake as lake

    (tmp_path / "t.csv").write_text("a\n" + "".join(f"{i}\n" for i in range(20)))
    reads, real = [], lake.read_rows
    monkeypatch.setattr(lake, "read_rows", lambda path, sheet, count=None: reads.append(count) or real(path, sheet, count))
    reader = lake.RowReader()
    assert len(reader(tmp_path / "t.csv", None, 5)) == 5 and len(reader(tmp_path / "t.csv", None, 3)) == 3
    assert len(reader(tmp_path / "t.csv", None, 50)) == 21 and len(reader(tmp_path / "t.csv", None, 40)) == 21
    assert len(reader(tmp_path / "t.csv", None)) == 21
    assert reads == [5, 50]


def test_reading_a_worksheet_stops_at_the_rows_asked_for(tmp_path):
    import openpyxl

    from ogda.lake import read_rows

    workbook = openpyxl.Workbook()
    for value in range(10):
        workbook.active.append([value, f"v{value}"])
    workbook.save(tmp_path / "t.xlsx")
    assert read_rows(tmp_path / "t.xlsx", None, 3) == [["0", "v0"], ["1", "v1"], ["2", "v2"]]
    assert len(read_rows(tmp_path / "t.xlsx", None)) == 10
