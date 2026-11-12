import pytest

from engine_health_flow import data
from engine_health_flow.cli import main


def test_md5sum_of_known_bytes(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"abc")
    assert data.md5sum(f) == "900150983cd24fb0d6963f7d28e17f72"


def test_download_writes_the_file_when_md5_matches(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"abc")
    dest = tmp_path / "out" / "copy.bin"
    got = data.download(dest, url=src.as_uri(), md5=data.md5sum(src))
    assert got == dest
    assert dest.read_bytes() == b"abc"
    assert not dest.with_name("copy.bin.part").exists()


def test_download_refuses_a_wrong_checksum(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"abc")
    dest = tmp_path / "copy.bin"
    with pytest.raises(ValueError, match="checksum"):
        data.download(dest, url=src.as_uri(), md5="0" * 32)
    assert not dest.exists()
    assert not dest.with_name("copy.bin.part").exists()


def test_download_skips_a_file_that_is_already_good(tmp_path):
    dest = tmp_path / "have.bin"
    dest.write_bytes(b"abc")
    # the url is not reachable, so any attempt to fetch would raise
    got = data.download(dest, url="file:///does/not/exist", md5=data.md5sum(dest))
    assert got == dest


def test_info_prints_counts(toy_csv, capsys):
    main(["info", "--csv", str(toy_csv)])
    out = capsys.readouterr().out
    assert "3 sequences, 16 rows" in out
    assert "length 4 to 7" in out
    assert "3 maintenance events" in out
