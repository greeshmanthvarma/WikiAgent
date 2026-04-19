from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db_models import PageModel, PageVersionModel

# Soft-delete and special handling (delete_page); spawn should create these for every agent.
RESERVED_PAGE_PATHS = frozenset({"index", "log", "schema"})


def normalize_path(raw: str) -> str:
    s = raw.strip().lower().replace("\\", "/") # replace backslashes with forward slashes, strip whitespace
    while "//" in s:
        s = s.replace("//", "/")
    s = s.strip("/") # remove leading/trailing slashes
    if not s:
        raise ValueError("path is empty")

    parts = s.split("/") # split into parts
    for p in parts:
        if p in (".", "..") or not p:
            raise ValueError(f"invalid path segment: {p!r}") # raise error if the path segment is invalid

    if parts[-1].endswith(".md"): # remove .md from the last segment only
        parts[-1] = parts[-1][:-3]
        if not parts[-1]:
            raise ValueError("invalid path: empty name after .md")

    return "/".join(parts)


async def read_page(page_path: str, agent_id: int, db: AsyncSession) -> str:
    """Read current page body; raises ValueError if path is invalid or page missing."""
    path = normalize_path(page_path)
    result = await db.execute(
        select(PageModel).where(PageModel.path == path, PageModel.agent_id == agent_id).where(PageModel.is_active == True)
    )
    page = result.scalar_one_or_none()
    if page is None:
        raise ValueError(f"page not found: {path}")
    return page.content


async def list_pages(agent_id: int, db: AsyncSession) -> list[str]:
    """List all page paths for an agent, sorted for stable tool output."""
    query = select(PageModel.path).where(PageModel.agent_id == agent_id).where(PageModel.is_active == True).order_by(PageModel.path)
    result = await db.execute(query)
    return list(result.scalars().all())

async def write_page(page_path: str, content: str, agent_id: int, db: AsyncSession, *, job_id: int | None = None) -> None:
    """
    Create or replace page body for this path (upsert), and append a PageVersion snapshot.
    """
    path = normalize_path(page_path)
    r = await db.execute(
        select(PageModel).where(PageModel.path == path, PageModel.agent_id == agent_id)
    )
    page = r.scalar_one_or_none()

    if page is None:
        page = PageModel(path=path, content=content, agent_id=agent_id)
        db.add(page)
        await db.flush()
    else:
        page.content = content
        if not page.is_active:
            page.is_active = True

    #get the previous version number. function coalesce returns the first non-null value, or 0 if all are null.
    #scalar_one() returns the single result of the query as a scalar value.
    v_result = await db.execute(
        select(func.coalesce(func.max(PageVersionModel.version), 0)).where(
            PageVersionModel.page_id == page.id
        )
    )
    next_version = int(v_result.scalar_one()) + 1 # convert the result to an integer and add 1 to get the next version number
    db.add(
        PageVersionModel(
            page_id=page.id,
            version=next_version,
            content=content,
            job_id=job_id,
        )
    )
    await db.commit()

def _escape_ilike_pattern(term: str) -> str:
    """Escape % and _ so user input is literal in SQL ILIKE."""
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


async def search_pages(search_query: str, agent_id: int, db: AsyncSession, *, limit: int = 100) -> list[str]:
    """
    Naive substring search over page bodies (case-insensitive).
    Returns matching paths only; use read_page for full text. Empty query returns [].
    """
    q = search_query.strip()
    if not q:
        return []
    pattern = f"%{_escape_ilike_pattern(q)}%"
    query = (
        select(PageModel.path)
        .where(PageModel.agent_id == agent_id)
        .where(PageModel.is_active == True)
        .where(PageModel.content.ilike(pattern, escape="\\"))
        .order_by(PageModel.path)
        .limit(limit)
    )
    result = await db.execute(query)
    return list(result.scalars().all())


async def append_log_entry(log_entry: str, agent_id: int, db: AsyncSession, *, job_id: int | None = None) -> None:
    """
    Append one line/block to the reserved `log` page (creates it if missing).
    Reuses write_page so PageVersion history stays consistent.
    """
    text = log_entry.strip()
    if not text:
        raise ValueError("log entry is empty")

    path = "log"
    r = await db.execute(
        select(PageModel).where(PageModel.path == path, PageModel.agent_id == agent_id)
    )
    page = r.scalar_one_or_none()
    base = (page.content or "").rstrip() if page else ""
    new_content = f"{base}\n\n{text}" if base else text
    await write_page(path, new_content, agent_id, db, job_id=job_id)

async def delete_page(page_path: str, agent_id: int, db: AsyncSession) -> None:
    path = normalize_path(page_path)
    if path in RESERVED_PAGE_PATHS:
        raise ValueError(f"cannot delete reserved page: {path}")

    r = await db.execute(
        select(PageModel).where(PageModel.path == path, PageModel.agent_id == agent_id)
    )
    page = r.scalar_one_or_none()
    if page is None:
        raise ValueError(f"page not found: {path}")
    if not page.is_active:
        return
    page.is_active = False
    await db.commit()

async def page_exists(page_path: str, agent_id: int, db: AsyncSession) -> dict[str, bool | None]:
    # No row -> scalar is None. Row with is_active=False -> False (still exists=True).
    path = normalize_path(page_path)
    r = await db.execute(
        select(PageModel.is_active).where(
            PageModel.path == path, PageModel.agent_id == agent_id
        )
    )
    is_active = r.scalar_one_or_none()
    if is_active is None:
        return {"exists": False, "is_active": None}
    return {"exists": True, "is_active": is_active}

async def get_page_metadata(page_path: str, agent_id: int, db: AsyncSession) -> dict:
    """Get metadata for a page."""
    path = normalize_path(page_path)
    r = await db.execute(
        select(PageModel).where(PageModel.path == path, PageModel.agent_id == agent_id)
    )
    page = r.scalar_one_or_none()
    if page is None:
        raise ValueError(f"page not found: {path}")

    return {
        "path": page.path,
        "description": page.description,
        "is_active": page.is_active,
        "created_at": page.created_at.isoformat() if page.created_at else None,
        "updated_at": page.updated_at.isoformat() if page.updated_at else None,
        "content_length": len(page.content or ""),
    }