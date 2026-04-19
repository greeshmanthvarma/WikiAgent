from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
import os
from urllib.parse import urlparse, urlunparse
from dotenv import load_dotenv
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
DB_ECHO = os.getenv("DB_ECHO", "false").lower() == "true"

_original_url = (os.getenv("DATABASE_URL") or "").strip()
_connect_args = {}
if _original_url:
    _url = urlparse(_original_url)
    if _url.query:
        DATABASE_URL = urlunparse(_url._replace(query=""))
    if "neon.tech" in _original_url or "sslmode=require" in _original_url.lower():
        _connect_args["ssl"] = True

engine = create_async_engine(
    DATABASE_URL,
    echo=DB_ECHO,
    future=True,
    connect_args=_connect_args,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Base class for models
Base = declarative_base()


# Dependency to get database session
async def get_db() -> AsyncSession:
    """Dependency to get database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

