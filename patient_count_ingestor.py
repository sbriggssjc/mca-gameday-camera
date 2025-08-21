from __future__ import annotations

"""Patient count ingestion utilities.

This module provides a ``preflight_schema_check`` function that ensures required
columns exist before ingestion. It supports field aliases so that existing
columns with equivalent meaning can satisfy a requirement. If neither the
expected column nor any of its aliases are present, the function will create
that column via the provided database client.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, MutableMapping


class SchemaClientProtocol:
    """Protocol for minimal database client used in preflight checks."""

    def get_columns(self, table: str) -> Iterable[str]:
        """Return an iterable of column names for ``table``."""

    def create_column(self, table: str, column: str) -> None:
        """Create ``column`` in ``table`` if it does not exist."""


@dataclass
class PreflightReport:
    """Report produced by :func:`preflight_schema_check`."""

    created_columns: Dict[str, List[str]] = field(default_factory=dict)

    def record(self, table: str, column: str) -> None:
        self.created_columns.setdefault(table, []).append(column)

    def __bool__(self) -> bool:  # pragma: no cover - convenience method
        return bool(self.created_columns)


FieldAliases = Mapping[str, Mapping[str, Iterable[str]]]
"""Type alias for field alias mapping.

The mapping is ``{table: {required_column: [alias1, alias2, ...]}}``.
"""


def preflight_schema_check(
    client: SchemaClientProtocol,
    required_schema: Mapping[str, Iterable[str]],
    field_aliases: FieldAliases | None = None,
) -> PreflightReport:
    """Ensure required columns exist, creating them when necessary.

    Parameters
    ----------
    client:
        Database client implementing :class:`SchemaClientProtocol`.
    required_schema:
        Mapping of table names to the columns that must be present.
    field_aliases:
        Optional mapping of aliases for required columns. If any alias exists
        in the table, the column is considered satisfied.

    Returns
    -------
    PreflightReport
        Report describing any columns that were created.
    """

    aliases: FieldAliases = field_aliases or {}
    report = PreflightReport()

    for table, columns in required_schema.items():
        existing = set(client.get_columns(table))
        table_aliases = aliases.get(table, {})

        for column in columns:
            possible_names = {column}
            possible_names.update(table_aliases.get(column, []))

            if any(name in existing for name in possible_names):
                continue

            client.create_column(table, column)
            report.record(table, column)
            existing.add(column)

    return report
