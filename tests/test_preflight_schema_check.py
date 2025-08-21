from __future__ import annotations

from typing import Dict, Iterable, List

import patient_count_ingestor as pci


class FakeClient(pci.SchemaClientProtocol):
    """In-memory fake database client for testing."""

    def __init__(self, schema: Dict[str, List[str]] | None = None) -> None:
        self.schema: Dict[str, List[str]] = schema or {}
        self.created: Dict[str, List[str]] = {}

    def get_columns(self, table: str) -> Iterable[str]:
        return list(self.schema.get(table, []))

    def create_column(self, table: str, column: str) -> None:
        self.schema.setdefault(table, []).append(column)
        self.created.setdefault(table, []).append(column)


def test_alias_satisfies_requirement() -> None:
    client = FakeClient({"medicare_clinics": ["number_of_chairs"]})
    required = {"medicare_clinics": ["clinic_chair_count"]}
    aliases = {"medicare_clinics": {"clinic_chair_count": ["number_of_chairs"]}}

    report = pci.preflight_schema_check(client, required, aliases)

    assert not report.created_columns
    assert client.schema["medicare_clinics"] == ["number_of_chairs"]


def test_missing_column_is_created() -> None:
    client = FakeClient({"facility_patient_counts": []})
    required = {"facility_patient_counts": ["total_patients"]}

    report = pci.preflight_schema_check(client, required)

    assert report.created_columns == {"facility_patient_counts": ["total_patients"]}
    assert client.schema["facility_patient_counts"] == ["total_patients"]
