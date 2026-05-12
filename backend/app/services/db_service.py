from sqlalchemy.ext.asyncio import AsyncSession

from app.db_models import SourceModel
from app.schemas.source import SourceCreate


async def db_create_source(db: AsyncSession, data: SourceCreate) -> SourceModel:
    row = SourceModel(
        agent_id=data.agent_id,
        kind=data.kind,
        status=data.status,
        storage_key=data.storage_key,
        source_url=data.source_url,
        original_filename=data.original_filename,
        content_type=data.content_type,
        byte_size=data.byte_size,
        extracted_text=data.extracted_text,
        error_message=data.error_message,
        raw_sha256=data.raw_sha256,
        sha256=data.sha256,
        extra=data.extra,
    )
    db.add(row)
    try:
        await db.commit()
    except Exception:
        await db.rollback()
        raise
    await db.refresh(row)
    return row
