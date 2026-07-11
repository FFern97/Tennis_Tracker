"""Housekeeping post-feature-001: smoke test en tests/smoke/, no en raíz."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_smoke_test_supabase_not_in_repo_root():
    assert not (REPO_ROOT / "smoke_test_supabase.py").exists()


def test_smoke_test_supabase_in_tests_smoke():
    path = REPO_ROOT / "tests" / "smoke" / "smoke_test_supabase.py"
    assert path.is_file()


def test_data_images_not_in_data_root():
    assert not (REPO_ROOT / "data" / "images").exists()


def test_court_fixtures_in_tests_fixtures():
    fixtures = REPO_ROOT / "tests" / "fixtures" / "court_images"
    assert fixtures.is_dir()
    png_count = len(list(fixtures.glob("*.png")))
    assert png_count >= 15
