"""The worker boundary: agent code reads and writes the Lake copy only. Needs Linux Landlock and seccomp, or srt on macOS."""

import pytest

from ogda.sandbox import IsolatedExecutor


@pytest.fixture
def executor(tmp_path):
    (tmp_path / "lake").mkdir()
    (tmp_path / "lake" / "values.csv").write_text("value\n42\n")
    (tmp_path / "secret.txt").write_text("SECRET")
    executor = IsolatedExecutor(tmp_path / "lake", tmp_path / "scratch", tmp_path / "logs", timeout=5)
    yield executor
    executor.cleanup()


def test_worker_reads_and_writes_the_lake_and_keeps_state(executor, tmp_path):
    executor("rows = (LAKE_ROOT / 'values.csv').read_text().splitlines()")
    output = executor("(LAKE_ROOT / 'derived.txt').write_text(rows[1])\nprint(len(rows))")
    assert output.logs.strip() == "2"
    assert (tmp_path / "lake" / "derived.txt").read_text() == "42"


def test_worker_cannot_read_outside_the_lake_or_use_the_network(executor, tmp_path):
    with pytest.raises(RuntimeError) as error:
        executor(f"print('before')\nopen({str(tmp_path / 'secret.txt')!r}).read()")
    assert "before" in str(error.value) and "SECRET" not in str(error.value)
    with pytest.raises(RuntimeError):
        executor("import socket\nsocket.socket()")


def test_stuck_cell_times_out_and_the_next_cell_gets_a_new_worker(executor):
    executor("kept = 1")
    with pytest.raises(TimeoutError, match="restarted"):
        executor("import time\ntime.sleep(30)")
    assert executor("print('alive')").logs.strip() == "alive"
    with pytest.raises(RuntimeError, match="kept"):
        executor("print(kept)")


def test_writes_to_the_stdout_descriptor_keep_the_worker_and_its_state(executor):
    executor("kept = 7")
    output = executor("import os\nos.write(1, b'raw\\n')\nprint(kept)")
    assert output.logs.strip() == "7"


def test_return_break_and_continue_inside_try_are_not_caught_by_except(executor):
    output = executor(
        "def number(text):\n"
        "    try:\n"
        "        return float(text)\n"
        "    except:\n"
        "        return -1.0\n"
        "def first_number(texts):\n"
        "    for text in texts:\n"
        "        try:\n"
        "            return float(text)\n"
        "        except Exception:\n"
        "            continue\n"
        "seen = []\n"
        "for i in range(5):\n"
        "    try:\n"
        "        if i == 1:\n"
        "            continue\n"
        "        if i == 3:\n"
        "            break\n"
        "        seen.append(i)\n"
        "    except:\n"
        "        seen.append('caught')\n"
        "print(number('3.5'), number('x'), first_number(['x', '2']), seen)")
    assert output.logs.strip() == "3.5 -1.0 2.0 [0, 2]"


def test_nested_generator_expressions_see_outer_loop_variables(executor):
    output = executor("shapes = {'a': [[1, 2], [3]], 'b': [[4, 5, 6]]}\n"
                      "print(sum(1 for parts in shapes.values() for part in parts if len(part) < 3),\n"
                      "      list(n for parts in shapes.values() for part in parts for n in part))")
    assert output.logs.strip() == "2 [1, 2, 3, 4, 5, 6]"


def test_worker_provides_object_and_bytes_builtins(executor):
    output = executor("print(pd.Series([1]).astype(object).dtype, bytes([104, 105]))")
    assert output.logs.strip() == "object b'hi'"


def test_lake_and_output_directories_must_be_separate(tmp_path):
    with pytest.raises(ValueError):
        IsolatedExecutor(tmp_path / "lake", tmp_path / "lake" / "scratch", tmp_path / "logs")


def test_worker_runs_ordinary_python_inside_the_lake(executor):
    output = executor("import os\nwith open(LAKE_ROOT / 'values.csv') as f:\n    print(repr(f.readline()), os.path.basename(f.name))")
    assert output.logs.strip() == "'value\\n' values.csv"


def test_worker_starts_with_common_modules_imported(executor):
    output = executor("print(json.dumps({'n': math.floor(2.5)}), pd.DataFrame({'a': [1]}).shape, np.int64(3))")
    assert output.logs.strip() == '{"n": 2} (1, 1) 3'


def test_worker_reads_html_tables(executor):
    output = executor("import pandas as pd\n"
                      "(LAKE_ROOT / 't.html').write_text('<table><tr><th>A</th></tr><tr><td>1</td></tr></table>')\n"
                      "print(pd.read_html(LAKE_ROOT / 't.html')[0]['A'].tolist())")
    assert output.logs.strip() == "[1]"


def test_worker_decodes_binary_and_compressed_files(executor):
    output = executor("import struct, zlib, gzip, bz2, lzma, zipfile, tarfile, tempfile\n"
                      "data = gzip.compress(zlib.compress(struct.pack('<i', 7)))\n"
                      "with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as handle:\n"
                      "    path = handle.name\n"
                      "with zipfile.ZipFile(path, 'w') as archive:\n"
                      "    archive.writestr('x.bin', data)\n"
                      "with zipfile.ZipFile(path) as archive:\n"
                      "    print(struct.unpack('<i', zlib.decompress(gzip.decompress(archive.read('x.bin'))))[0])")
    assert output.logs.strip() == "7"


def test_read_only_lake_with_a_workspace_that_outlives_the_worker(tmp_path):
    (tmp_path / "lake").mkdir()
    (tmp_path / "lake" / "values.csv").write_text("value\n42\n")
    workspace = tmp_path / "task" / "workspace"
    executor = IsolatedExecutor(tmp_path / "lake", tmp_path / "task" / ".workers", tmp_path / "logs", timeout=5,
                                lake="read", workspace=workspace)
    try:
        assert executor("print((LAKE_ROOT / 'values.csv').read_text().splitlines()[1])").logs.strip() == "42"
        with pytest.raises(RuntimeError, match="PermissionError"):
            executor("(LAKE_ROOT / 'derived.txt').write_text('x')")
        executor("import shutil\nshutil.copyfile(LAKE_ROOT / 'values.csv', WORKSPACE / 'values.csv')\n"
                 "(WORKSPACE / 'values.csv').write_text((WORKSPACE / 'values.csv').read_text() + '43\\n')")
    finally:
        executor.cleanup()
    assert not (tmp_path / "lake" / "derived.txt").exists()
    assert (workspace / "values.csv").read_text() == "value\n42\n43\n"
    assert (tmp_path / "lake" / "values.csv").read_text() == "value\n42\n"


def test_hidden_lake_cannot_be_read_and_has_no_lake_root(tmp_path):
    (tmp_path / "lake").mkdir()
    (tmp_path / "lake" / "values.csv").write_text("value\n42\n")
    executor = IsolatedExecutor(tmp_path / "lake", tmp_path / "scratch", tmp_path / "logs", timeout=5, lake="none")
    try:
        with pytest.raises(RuntimeError, match="PermissionError"):
            executor(f"open({str(tmp_path / 'lake' / 'values.csv')!r}).read()")
        with pytest.raises(RuntimeError, match="LAKE_ROOT"):
            executor("print(LAKE_ROOT)")
    finally:
        executor.cleanup()




def test_a_tool_correction_reaches_the_agent_with_its_message_and_other_errors_only_by_type(executor):
    from smolagents import tool

    from ogda.sandbox.executor import ToolCorrection

    @tool
    def check(value: int) -> str:
        """Check a value in the host process.

        Args:
            value: The value.
        """
        if value < 0:
            raise ToolCorrection("value must be positive")
        raise OSError("/private/host/path is unavailable")

    executor.send_tools({"check": check})
    output = executor("try:\n    check(-1)\nexcept ValueError as error:\n    print('correction:', error)\n"
                      "try:\n    check(1)\nexcept Exception as error:\n    print('other:', error)")
    assert output.logs.splitlines() == ["correction: value must be positive", "other: OSError"]


def test_workspace_must_be_outside_the_lake(tmp_path):
    with pytest.raises(ValueError):
        IsolatedExecutor(tmp_path / "lake", tmp_path / "scratch", tmp_path / "logs", workspace=tmp_path / "lake" / "work")


def test_worker_reads_excel_and_runs_statistics(executor, tmp_path):
    import pandas as pd

    pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [2.1, 3.9, 6.2, 7.8]}).to_excel(tmp_path / "lake" / "values.xlsx", index=False)
    output = executor(
        "import pandas as pd\nimport numpy as np\nimport numpy.linalg\nfrom scipy import stats\nimport statsmodels.api as sm\n"
        "frame = pd.read_excel(LAKE_ROOT / 'values.xlsx')\n"
        "fit = sm.OLS(frame['y'], sm.add_constant(frame['x'])).fit()\n"
        "print(round(stats.pearsonr(frame['x'], frame['y'])[0], 2), round(fit.params['x'], 2),"
        " round(float(np.linalg.norm(np.random.default_rng(0).normal(size=0))), 1))")
    assert output.logs.strip() == "1.0 1.94 0.0"




def test_library_network_access_is_blocked_by_the_os_sandbox(executor):
    # pandas opens URLs through urllib itself, past the interpreter's import list.
    with pytest.raises(RuntimeError, match="URLError"):
        executor("import pandas as pd\npd.read_csv('https://example.com/data.csv')")


def test_tool_time_does_not_count_toward_the_cell_timeout(tmp_path):
    import time

    from smolagents import tool

    @tool
    def wait(seconds: float) -> str:
        """Wait in the host process.

        Args:
            seconds: How long to wait.
        """
        time.sleep(seconds)
        return "waited"

    (tmp_path / "lake").mkdir()
    executor = IsolatedExecutor(tmp_path / "lake", tmp_path / "scratch", tmp_path / "logs", timeout=2)
    executor.send_tools({"wait": wait})
    try:
        assert executor("print(wait(3))").logs.strip() == "waited"
        with pytest.raises(TimeoutError):
            executor("import time\nwhile True:\n    time.sleep(0.01)")
    finally:
        executor.cleanup()


def ground_executor(tmp_path, **settings):
    (tmp_path / "lake").mkdir()
    (tmp_path / "lake" / "values.csv").write_text("value\n42\n")
    (tmp_path / "cache").mkdir()
    return IsolatedExecutor(tmp_path / "lake", tmp_path / "scratch", tmp_path / "logs",
                            **{"timeout": 5, "lake": "ground", "readable": [tmp_path / "cache"], **settings})


def test_ground_mode_has_no_lake_root_but_reads_the_lake_and_readable_roots(tmp_path):
    executor = ground_executor(tmp_path)
    (tmp_path / "cache" / "t.txt").write_text("cached")
    try:
        with pytest.raises(RuntimeError, match="LAKE_ROOT"):
            executor("print(LAKE_ROOT)")
        assert executor(f"print(open({str(tmp_path / 'lake/values.csv')!r}).read().split()[1])").logs.strip() == "42"
        assert executor(f"print(open({str(tmp_path / 'cache/t.txt')!r}).read())").logs.strip() == "cached"
        with pytest.raises(RuntimeError, match="PermissionError"):
            executor(f"open({str(tmp_path / 'lake/new.txt')!r}, 'w').write('x')")
    finally:
        executor.cleanup()


def test_readable_paths_must_be_outside_the_lake_and_exist(tmp_path):
    with pytest.raises(ValueError):
        IsolatedExecutor(tmp_path / "lake", tmp_path / "scratch", tmp_path / "logs", readable=[tmp_path / "lake" / "c"])
    executor = IsolatedExecutor(tmp_path / "lake", tmp_path / "scratch", tmp_path / "logs", lake="ground",
                                readable=[tmp_path / "cache"])
    try:
        with pytest.raises(FileNotFoundError):
            executor("print(1)")
    finally:
        executor.cleanup()


def test_a_tool_result_prints_its_text_and_returns_tables_series_and_paths(tmp_path):
    import pandas as pd
    from smolagents import tool

    executor = ground_executor(tmp_path)
    pd.DataFrame({"Proton_flux_>30_Mev": [1, 2]}, index=[3, 5]).to_parquet(tmp_path / "cache/t.parquet")
    table, path = str(tmp_path / "cache/t.parquet"), str(tmp_path / "lake/values.csv")

    @tool
    def show(names: list) -> dict:
        """Show data.

        Args:
            names: The names to show.
        """
        return {"__print__": "meaning of " + ", ".join(names), "__single__": False,
                "__values__": [{"__table__": table}, {"__table__": table, "__field__": "Proton_flux_>30_Mev"}, {"__path__": path},
                               {"__paths__": [path]}]}

    executor.send_tools({"show": show})
    try:
        # Agent code cannot read dunder attributes such as __name__, so the cell checks types with isinstance.
        output = executor("from pathlib import Path\nprint('before')\na, b, c, d = show('x', 'y')\n"
                          "print(isinstance(a, pd.DataFrame), list(a.index), isinstance(b, pd.Series), b.tolist(),"
                          " isinstance(c, Path), len(d))")
        assert output.logs.splitlines() == ["before", "meaning of x, y", "True [3, 5] True [1, 2] True 1"]
    finally:
        executor.cleanup()


def test_a_tool_result_for_one_file_returns_that_files_rows(tmp_path):
    import pandas as pd
    from smolagents import tool

    executor = ground_executor(tmp_path)
    pd.DataFrame({"file": ["a.csv", "b.csv", "b.csv"], "v": [1, 2, 3]}, index=[1, 1, 2]).to_parquet(
        tmp_path / "cache/t.parquet")
    table = str(tmp_path / "cache/t.parquet")

    @tool
    def show(names: list) -> dict:
        """Show data.

        Args:
            names: The names to show.
        """
        return {"__print__": "one File", "__single__": True,
                "__values__": [{"__table__": table, "__file__": "b.csv", "__file_column__": "file"}]}

    executor.send_tools({"show": show})
    try:
        output = executor("a = show('x')\nprint(a.v.tolist(), list(a.index))")
        assert output.logs.splitlines() == ["one File", "[2, 3] [1, 2]"]
    finally:
        executor.cleanup()


def test_a_tool_whose_only_parameter_is_names_takes_them_separately_or_as_a_list(executor):
    from smolagents import tool

    @tool
    def echo(names: list) -> list:
        """Echo names.

        Args:
            names: The names.
        """
        return names

    executor.send_tools({"echo": echo})
    assert executor("print(echo('a', 'b'), echo(['c']), echo('d'))").logs.strip() == "['a', 'b'] ['c'] ['d']"


class Recorder:
    def __init__(self, names):
        self.names, self.calls = names, []

    def before(self, code):
        return self.names

    def after(self, code, error, summaries, restarted):
        self.calls.append({"error": error, "summaries": summaries, "restarted": restarted})


def test_the_worker_summarizes_the_variables_the_recorder_names(tmp_path):
    recorder = Recorder(["x", "n", "df", "s", "nan", "missing"])
    executor = ground_executor(tmp_path, recorder=recorder)
    try:
        executor("x = np.float64(0.5)\nn = [1, 2]\ndf = pd.DataFrame({'a': [1, 2]})\ns = 'hello'\nnan = float('nan')")
        with pytest.raises(RuntimeError):
            executor("y = 1 / 0")
    finally:
        executor.cleanup()
    first, second = recorder.calls
    assert first["error"] is None and first["summaries"] == {
        "x": {"type": "float64", "value": 0.5}, "n": {"type": "list", "length": 2, "value": [1, 2]},
        "df": {"type": "DataFrame", "shape": [2, 1]}, "s": {"type": "str", "value": "hello"},
        "nan": {"type": "float"}}
    assert "Code execution failed at line 'y = 1 / 0'" in second["error"] and second["restarted"] is False


def test_a_restarted_worker_tells_the_recorder(tmp_path):
    recorder = Recorder([])
    executor = ground_executor(tmp_path, recorder=recorder, timeout=1)
    try:
        with pytest.raises(TimeoutError):
            executor("import time\nwhile True:\n    time.sleep(0.01)")
    finally:
        executor.cleanup()
    assert recorder.calls[-1]["restarted"] is True


def test_the_recorder_gets_the_whole_error_of_a_failing_statement(tmp_path):
    recorder = Recorder([])
    executor = ground_executor(tmp_path, recorder=recorder)
    statement = 'y = ["' + "a" * 4500 + '", 1 / 0]'
    try:
        with pytest.raises(RuntimeError):
            executor("x = 1\n" + statement)
    finally:
        executor.cleanup()
    assert f"Code execution failed at line '{statement}' due to: ZeroDivisionError" in recorder.calls[0]["error"]


def test_worker_reads_a_geopackage_and_reprojects_it(tmp_path):
    executor = IsolatedExecutor(tmp_path / "lake", tmp_path / "scratch", tmp_path / "logs", timeout=60)
    (tmp_path / "lake").mkdir()
    try:
        output = executor("import geopandas as gpd\nfrom shapely.geometry import Point, box\n"
                          "gpd.GeoDataFrame({'name': ['a']}, geometry=[box(0, 0, 2, 2)], crs='EPSG:4326')"
                          ".to_file(LAKE_ROOT / 'areas.gpkg', driver='GPKG')\n"
                          "areas = gpd.read_file(LAKE_ROOT / 'areas.gpkg')\n"
                          "print(areas.contains(Point(1, 1)).tolist(), round(areas.to_crs('EPSG:3857').total_bounds[2]))")
    finally:
        executor.cleanup()
    assert output.logs.strip() == "[True] 222639"


def test_worker_reads_a_cdf_file(tmp_path):
    executor = IsolatedExecutor(tmp_path / "lake", tmp_path / "scratch", tmp_path / "logs", timeout=60)
    (tmp_path / "lake").mkdir()
    try:
        output = executor("import numpy as np, cdflib\nfrom cdflib.cdfwrite import CDF\n"
                          "cdf = CDF(str(LAKE_ROOT / 'acc.cdf'))\n"
                          "cdf.write_var({'Variable': 'a', 'Data_Type': CDF.CDF_DOUBLE, 'Num_Elements': 1, 'Rec_Vary': True,"
                          " 'Dim_Sizes': []}, var_data=np.array([1.5, 2.5]))\n"
                          "cdf.close()\n"
                          "print(cdflib.CDF(str(LAKE_ROOT / 'acc.cdf')).varget('a').tolist())")
    finally:
        executor.cleanup()
    assert output.logs.strip() == "[1.5, 2.5]"


def test_worker_propagates_a_tle_with_sgp4_and_skyfield(tmp_path):
    executor = IsolatedExecutor(tmp_path / "lake", tmp_path / "scratch", tmp_path / "logs", timeout=60)
    (tmp_path / "lake").mkdir()
    line1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9005"
    line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    try:
        output = executor("from sgp4.api import Satrec\nfrom skyfield.api import EarthSatellite, load, wgs84\n"
                          f"line1, line2 = {line1!r}, {line2!r}\n"
                          "error, position, velocity = Satrec.twoline2rv(line1, line2).sgp4(2460311, 0.0)\n"
                          "ts = load.timescale(builtin=True)\n"
                          "height = wgs84.height_of(EarthSatellite(line1, line2, 'ISS', ts).at(ts.utc(2024, 1, 1, 12)))\n"
                          "print(error, 300 < height.km < 500)")
    finally:
        executor.cleanup()
    assert output.logs.strip() == "0 True"


def test_worker_runs_the_nrlmsise_00_atmosphere_model(tmp_path):
    executor = IsolatedExecutor(tmp_path / "lake", tmp_path / "scratch", tmp_path / "logs", timeout=60)
    (tmp_path / "lake").mkdir()
    try:
        output = executor("import numpy as np, pymsis\n"
                          "out = pymsis.calculate(np.datetime64('2024-05-11T00:00'), 0, 0, 450, f107s=150, f107as=150,"
                          " aps=[[4] * 7], version=0)\n"
                          "print(1e-14 < float(np.squeeze(out)[pymsis.Variable.MASS_DENSITY]) < 1e-11)")
    finally:
        executor.cleanup()
    assert output.logs.strip() == "True"
