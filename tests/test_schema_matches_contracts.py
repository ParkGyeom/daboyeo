"""DB enum 과 contracts enum 이 어긋나지 않는지 확인한다."""

import re
from pathlib import Path

import pytest

from daboyeo_contracts import DefectType, RiskLevel

SQL = (Path(__file__).parents[1] / "server/sql/001_init.sql").read_text(encoding="utf-8")


def sql_enum(name: str) -> set[str]:
    m = re.search(rf"CREATE TYPE {name} AS ENUM \(([^)]*)\)", SQL)
    assert m, f"enum {name} not found in 001_init.sql"
    return set(re.findall(r"'([^']+)'", m.group(1)))


@pytest.mark.parametrize("sql_name,enum", [("defect_type", DefectType), ("risk_level", RiskLevel)])
def test_enum_matches(sql_name, enum):
    assert sql_enum(sql_name) == {e.value for e in enum}
