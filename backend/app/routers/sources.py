import hashlib
from io import BytesIO
from zipfile import BadZipFile

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from docx import Document
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.db_models import SourceKind, SourceStatus
from app.schemas.source import SourceCreate
from app.services.db_service import db_create_source

router = APIRouter()

_ALLOWED = frozenset(
    {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/markdown",
        "text/plain",
    }
)


def _effective_content_type(upload: UploadFile) -> str | None:
    ct = upload.content_type
    if ct:
        return ct
    name = (upload.filename or "").lower()
    if name.endswith(".pdf"):
        return "application/pdf"
    if name.endswith(".docx"):
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if name.endswith(".md"):
        return "text/markdown"
    if name.endswith(".txt"):
        return "text/plain"
    return None


def _build_source_create(
    *,
    agent_id: int,
    kind: SourceKind,
    content_type: str,
    content_bytes: bytes,
    extracted_text: str,
    original_filename: str | None,
) -> SourceCreate:
    return SourceCreate(
        agent_id=agent_id,
        kind=kind,
        status=SourceStatus.READY,
        extracted_text=extracted_text,
        raw_sha256=hashlib.sha256(content_bytes).hexdigest(),
        sha256=hashlib.sha256(extracted_text.encode("utf-8")).hexdigest(),
        byte_size=len(content_bytes),
        original_filename=original_filename,
        content_type=content_type,
    )


def _decode_utf8_text(content_bytes: bytes) -> str:
    try:
        return content_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail="File is not valid UTF-8; save as UTF-8 or use a binary format.",
        ) from e


@router.post("/sources/upload")
async def create_source(
    file: UploadFile = File(...),
    agent_id: int = Form(..., ge=1),
    db: AsyncSession = Depends(get_db),
):
    content_type = _effective_content_type(file)
    if content_type not in _ALLOWED:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported or missing content type; set Content-Type or use "
                "a recognized extension (.pdf, .docx, .md, .txt)."
            ),
        )

    content_bytes = await file.read()

    match content_type:
        case "application/pdf":
            try:
                reader = PdfReader(BytesIO(content_bytes))
            except PdfReadError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid or unreadable PDF: {e}",
                ) from e
            extracted_text = "\n".join(
                (page.extract_text() or "") for page in reader.pages
            )
            source_create = _build_source_create(
                agent_id=agent_id,
                kind=SourceKind.UPLOAD_PDF,
                content_type=content_type,
                content_bytes=content_bytes,
                extracted_text=extracted_text,
                original_filename=file.filename,
            )
            return await db_create_source(db, source_create)
        case "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            try:
                document = Document(BytesIO(content_bytes))
            except BadZipFile as e:
                raise HTTPException(
                    status_code=400,
                    detail="Not a valid DOCX file (Office Open XML / ZIP).",
                ) from e
            except ValueError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid DOCX: {e}",
                ) from e
            extracted_text = "\n".join(p.text for p in document.paragraphs)
            source_create = _build_source_create(
                agent_id=agent_id,
                kind=SourceKind.UPLOAD_DOCX,
                content_type=content_type,
                content_bytes=content_bytes,
                extracted_text=extracted_text,
                original_filename=file.filename,
            )
            return await db_create_source(db, source_create)
        case "text/markdown":
            extracted_text = _decode_utf8_text(content_bytes)
            source_create = _build_source_create(
                agent_id=agent_id,
                kind=SourceKind.UPLOAD_MARKDOWN,
                content_type=content_type,
                content_bytes=content_bytes,
                extracted_text=extracted_text,
                original_filename=file.filename,
            )
            return await db_create_source(db, source_create)
        case "text/plain":
            extracted_text = _decode_utf8_text(content_bytes)
            source_create = _build_source_create(
                agent_id=agent_id,
                kind=SourceKind.UPLOAD_PLAIN_TEXT,
                content_type=content_type,
                content_bytes=content_bytes,
                extracted_text=extracted_text,
                original_filename=file.filename,
            )
            return await db_create_source(db, source_create)
        case _:
            raise HTTPException(status_code=400, detail="Unsupported file type")
