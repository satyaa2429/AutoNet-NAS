"""Updating the UI theme must preserve unrelated app configuration."""

import tomllib

from apply_update import merge_theme, theme_matches


def test_existing_server_settings_survive_theme_update(tmp_path):
    source = tmp_path / "new.toml"
    target = tmp_path / "existing.toml"
    source.write_text(
        '[theme]\nbase="light"\ntextColor="#171717"\n[theme.sidebar]\nbackgroundColor="#141414"\n',
        encoding="utf-8",
    )
    target.write_text(
        '[server]\nport=8501\n\n[theme]\nbase="dark"\n[theme.sidebar]\nbackgroundColor="#000022"\n'
        '\n[browser]\ngatherUsageStats=false\n\n[theme.dark]\nprimaryColor="#008844"\n',
        encoding="utf-8",
    )
    assert not theme_matches(source, target)
    target.write_text(merge_theme(source, target), encoding="utf-8")
    parsed = tomllib.loads(target.read_text())
    assert parsed["server"]["port"] == 8501
    assert parsed["browser"]["gatherUsageStats"] is False
    assert parsed["theme"] == tomllib.loads(source.read_text())["theme"]
    assert theme_matches(source, target)
