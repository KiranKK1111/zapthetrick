"""Declarative driver catalog.

Each driver describes:
  - the kind (relational vs vector vs cache vs blob)
  - the connection fields the UI should render
  - the test command (used by the 10-step probe)
  - a default port
  - any optional Python package gate

The UI reads this to render the multi-driver form. The backend reads
it to validate a workspace's connection shape before saving.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DriverKind(str, Enum):
    RELATIONAL = "relational"
    VECTOR = "vector"
    CACHE = "cache"
    BLOB = "blob"


@dataclass
class FormField:
    name: str
    type: str = "string"   # string | int | bool | secret | enum
    label: str = ""
    default: object = None
    choices: list[str] = field(default_factory=list)
    secret: bool = False
    required: bool = True
    notes: str = ""


@dataclass
class Driver:
    id: str
    kind: DriverKind
    label: str
    requires_pkg: str | None = None
    default_port: int | None = None
    fields: list[FormField] = field(default_factory=list)


_PG_FIELDS = [
    FormField("host", "string", "Host", default="localhost"),
    FormField("port", "int", "Port", default=5432),
    FormField("db", "string", "Database", required=True),
    FormField("schema_name", "string", "Schema", default="zapthetrick"),
    FormField("user", "string", "User", required=True),
    FormField(
        "password",
        "secret",
        "Password",
        secret=True,
        required=False,
        notes="Leave blank if your Postgres user has no password (rare on localhost).",
    ),
]

_MYSQL_FIELDS = [
    FormField("host", "string", "Host", default="localhost"),
    FormField("port", "int", "Port", default=3306),
    FormField("db", "string", "Database", required=True),
    FormField("user", "string", "User", required=True),
    FormField("password", "secret", "Password", secret=True, required=False),
]

_SQLITE_FIELDS = [
    FormField("path", "string", "File path", default="./data/zapthetrick.db", required=True,
              notes="Path to the .db file. Created on first connect."),
]

_LANCEDB_FIELDS = [
    FormField("uri", "string", "URI / path", default="./data/lancedb",
              notes="Local path or `s3://bucket/lancedb` URL.", required=True),
    FormField("api_key", "secret", "API key", secret=True, required=False),
]

_REDIS_FIELDS = [
    FormField("url", "string", "URL", default="redis://localhost:6379"),
    FormField("default_ttl_seconds", "int", "Default TTL (s)", default=3600, required=False),
]

_MINIO_FIELDS = [
    FormField("endpoint", "string", "Endpoint", default="http://localhost:9000"),
    FormField("access", "string", "Access key", required=True),
    FormField("secret", "secret", "Secret key", secret=True, required=True),
    FormField("bucket", "string", "Bucket", default="zapthetrick"),
]

_FS_BLOB_FIELDS = [
    FormField("path", "string", "Path", default="./data/blobs",
              notes="Local directory for binary blobs."),
]


DRIVERS: dict[str, Driver] = {
    "postgres": Driver(
        id="postgres", kind=DriverKind.RELATIONAL, label="PostgreSQL",
        requires_pkg="asyncpg", default_port=5432, fields=_PG_FIELDS,
    ),
    "mysql": Driver(
        id="mysql", kind=DriverKind.RELATIONAL, label="MySQL",
        requires_pkg="aiomysql", default_port=3306, fields=_MYSQL_FIELDS,
    ),
    "sqlite": Driver(
        id="sqlite", kind=DriverKind.RELATIONAL, label="SQLite (local)",
        requires_pkg=None, default_port=None, fields=_SQLITE_FIELDS,
    ),
    "lancedb": Driver(
        id="lancedb", kind=DriverKind.VECTOR, label="LanceDB",
        requires_pkg="lancedb", default_port=None, fields=_LANCEDB_FIELDS,
    ),
    "redis": Driver(
        id="redis", kind=DriverKind.CACHE, label="Redis / Dragonfly",
        requires_pkg="redis", default_port=6379, fields=_REDIS_FIELDS,
    ),
    "minio": Driver(
        id="minio", kind=DriverKind.BLOB, label="MinIO / S3",
        requires_pkg="aiobotocore", default_port=9000, fields=_MINIO_FIELDS,
    ),
    "filesystem": Driver(
        id="filesystem", kind=DriverKind.BLOB, label="Local filesystem",
        requires_pkg=None, fields=_FS_BLOB_FIELDS,
    ),
}


__all__ = ["DRIVERS", "Driver", "DriverKind", "FormField"]
