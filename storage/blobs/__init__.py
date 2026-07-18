"""Raw-file storage abstraction — filesystem (default) or MinIO/S3."""
from .base import BlobStore
from .factory import close_blobs, get_blobs

__all__ = ["BlobStore", "get_blobs", "close_blobs"]
