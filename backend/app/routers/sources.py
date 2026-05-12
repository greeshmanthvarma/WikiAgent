import hashlib
from io import BytesIO
from urllib.parse import urlparse
from zipfile import BadZipFile

import httpx
import trafilatura
from docx import Document
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.db_models import SourceKind, SourceStatus
from app.schemas.source import SourceCreate, SourceRead
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

_ALLOWED_HTML_MIME = frozenset({"text/html", "application/xhtml+xml"})
# Only scan the start of the body when Content-Type is wrong/missing; real HTML declares itself near the top.
_HTML_SNIFF_BYTES = 8192
_FETCH_TIMEOUT = httpx.Timeout(30.0)
# Outgoing User-Agent for URL fetches; many sites block empty defaults. Point the URL at your real contact or policy page.
_DEFAULT_UA = "wikiagent/0.1 (+https://example.local)"


# Strip parameters (charset, boundary) so ``text/html; charset=utf-8`` matches ``text/html``.
def _mime_base(content_type: str | None) -> str:
    if not content_type:
        return ""
    return content_type.split(";")[0].strip().lower()


def _looks_like_html(text: str) -> bool:
    head = text[:_HTML_SNIFF_BYTES].lower().lstrip()
    return (
        head.startswith("<!doctype html")
        or head.startswith("<html")
        or "<html" in head[:2000]
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
    content_type: str | None,
    content_bytes: bytes,
    extracted_text: str,
    original_filename: str | None = None,
    source_url: str | None = None,
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
        source_url=source_url,
    )


def _decode_utf8_text(content_bytes: bytes) -> str:
    try:
        return content_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        raise HTTPException(
            status_code=400,
            detail="File is not valid UTF-8; save as UTF-8 or use a binary format.",
        ) from e


@router.post("/sources/upload", response_model=SourceRead)
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
            row = await db_create_source(db, source_create)
            return SourceRead.model_validate(row)
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
            row = await db_create_source(db, source_create)
            return SourceRead.model_validate(row)
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
            row = await db_create_source(db, source_create)
            return SourceRead.model_validate(row)
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
            row = await db_create_source(db, source_create)
            return SourceRead.model_validate(row)
        case _:
            raise HTTPException(status_code=400, detail="Unsupported file type")


@router.post("/sources/url", response_model=SourceRead)
async def create_source_from_url(
    url: str = Form(...),
    agent_id: int = Form(..., ge=1),
    db: AsyncSession = Depends(get_db),
):
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(
            status_code=400,
            detail="Only http and https URLs are supported.",
        )

    try:
        async with httpx.AsyncClient(
            timeout=_FETCH_TIMEOUT,
            headers={"User-Agent": _DEFAULT_UA},
            follow_redirects=True,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.RequestError as e:
        req_url = e.request.url if e.request is not None else url
        raise HTTPException(
            status_code=400,
            detail=f"Request failed for {req_url!r}.",
        ) from e
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=400,
            detail=(
                f"HTTP {e.response.status_code} while requesting "
                f"{e.request.url!r}."
            ),
        ) from e

    content_bytes = response.content
    html = response.text
    ct_header = response.headers.get("content-type")
    mime = _mime_base(ct_header)
    if mime not in _ALLOWED_HTML_MIME and not _looks_like_html(html):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected an HTML page (text/html or application/xhtml+xml); "
                f"Content-Type was {ct_header!r}."
            ),
        )

    extracted = trafilatura.extract(html, url=str(response.url))
    extracted_text = extracted or ""

    source_create = _build_source_create(
        agent_id=agent_id,
        kind=SourceKind.URL_HTML,
        content_type=ct_header or "text/html",
        content_bytes=content_bytes,
        extracted_text=extracted_text,
        original_filename=None,
        source_url=url.strip(),
    )
    row = await db_create_source(db, source_create)
    return SourceRead.model_validate(row)