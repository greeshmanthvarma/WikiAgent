from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.db_models import SourceKind, SourceStatus


class SourceCreate(BaseModel):

    model_config = ConfigDict(extra="forbid")  # unknown JSON/body keys → validation error

    agent_id: int = Field(..., ge=1)
    kind: SourceKind
    status: SourceStatus = SourceStatus.PENDING

    storage_key: str | None = None
    source_url: str | None = None
    original_filename: str | None = None
    content_type: str | None = None
    byte_size: int | None = Field(None, ge=0)

    extracted_text: str | None = None
    error_message: str | None = None

    raw_sha256: str | None = None
    sha256: str | None = None
    extra: dict[str, Any] | None = None


class SourceRead(BaseModel):
    # Omits extracted_text/extra in responses (size / leakage)
    model_config = ConfigDict(
        from_attributes=True,  # model_validate(sqlalchemy_row): map column attrs to fields
    )

    id: int
    agent_id: int
    kind: SourceKind
    status: SourceStatus
    storage_key: str | None = None
    source_url: str | None = None
    original_filename: str | None = None
    content_type: str | None = None
    byte_size: int | None = None
    error_message: str | None = None
    raw_sha256: str | None = None
    sha256: str | None = None
    created_at: datetime
    updated_at: datetime
