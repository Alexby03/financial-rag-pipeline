import mimetypes
from datetime import timedelta
from io import BytesIO
from urllib.parse import quote

from minio import Minio
from minio.error import S3Error

from app.config import Settings

INLINE_MIME_TYPES = {"application/pdf", "text/plain", "text/markdown", "text/html"}


def is_viewable_in_browser(filename: str) -> bool:
    """Whether a browser can render the file itself instead of downloading it."""
    mime_type, _ = mimetypes.guess_type(filename)
    if mime_type is None:
        return False
    return mime_type.startswith("image/") or mime_type in INLINE_MIME_TYPES


class MinioStorage:
    def __init__(self, settings: Settings) -> None:
        self._bucket = settings.minio_bucket
        self._ttl = timedelta(seconds=settings.presigned_url_ttl_seconds)
        self._client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        # Presigned URLs are signed against the host, so they must carry the browser-facing one.
        if settings.minio_public_endpoint == settings.minio_endpoint:
            self._presign_client = self._client
        else:
            self._presign_client = Minio(
                settings.minio_public_endpoint,
                access_key=settings.minio_access_key,
                secret_key=settings.minio_secret_key,
                secure=settings.minio_secure,
            )

    @property
    def bucket(self) -> str:
        return self._bucket

    def ensure_bucket(self) -> None:
        if not self._client.bucket_exists(self._bucket):
            self._client.make_bucket(self._bucket)

    def upload(self, object_key: str, data: bytes, content_type: str) -> str:
        self.ensure_bucket()
        self._client.put_object(
            self._bucket,
            object_key,
            BytesIO(data),
            length=len(data),
            content_type=content_type,
        )
        return object_key

    def delete(self, object_key: str) -> None:
        """Remove an object. Missing keys are ignored so rollback can be best-effort."""
        try:
            self._client.remove_object(self._bucket, object_key)
        except S3Error as error:
            if error.code in {"NoSuchKey", "NoSuchObject"}:
                return
            raise

    @property
    def url_ttl_seconds(self) -> int:
        return int(self._ttl.total_seconds())

    def presigned_url(self, object_key: str, filename: str, inline: bool) -> str:
        disposition = "inline" if inline else "attachment"
        return self._presign_client.presigned_get_object(
            self._bucket,
            object_key,
            expires=self._ttl,
            response_headers={
                "response-content-disposition": (
                    f"{disposition}; filename*=UTF-8''{quote(filename)}"
                )
            },
        )

    def object_exists(self, object_key: str) -> bool:
        try:
            self._client.stat_object(self._bucket, object_key)
            return True
        except S3Error:
            return False

    def download(self, object_key: str) -> bytes:
        response = self._client.get_object(self._bucket, object_key)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def ping(self) -> bool:
        try:
            self._client.bucket_exists(self._bucket)
            return True
        except S3Error:
            return False
