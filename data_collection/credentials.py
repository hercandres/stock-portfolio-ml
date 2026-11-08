"""Read Alpaca key names from the environment or a local .env file.

Values are never printed by this module. The example env file lists names
only and is safe to commit. A real ``.env`` is gitignored.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

# Names Alpaca documents for the Trading API key pair. The market-data API
# expects these values in the APCA-API-KEY-ID and APCA-API-SECRET-KEY headers.
APCA_API_KEY_ID = "APCA_API_KEY_ID"
APCA_API_SECRET_KEY = "APCA_API_SECRET_KEY"
REQUIRED_ALPACA_KEYS = (APCA_API_KEY_ID, APCA_API_SECRET_KEY)


def load_env_file(path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE file. Missing files return an empty dict."""
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        values[name.strip()] = value.strip().strip('"').strip("'")
    return values


def resolve_alpaca_keys(
    env_path: Path,
    environ: Mapping[str, str],
) -> tuple[dict[str, str] | None, list[str]]:
    """Return key values, or the names that are still empty.

    A non-empty process environment value wins over the file. No request
    should be sent when either required value is missing.
    """
    file_values = load_env_file(env_path)
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for name in REQUIRED_ALPACA_KEYS:
        process_value = str(environ.get(name, "")).strip()
        file_value = file_values.get(name, "").strip()
        value = process_value or file_value
        if not value:
            missing.append(name)
        else:
            resolved[name] = value
    if missing:
        return None, missing
    return resolved, []
