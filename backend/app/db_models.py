import enum

from sqlalchemy import (
    Column,
    DateTime,
    Enum as SQLEnum,
    ForeignKey,
    Integer,
    String,
    Text,
    Boolean,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class MessageRole(str, enum.Enum):
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"


class AgentStatus(str, enum.Enum):
    IDLE = "idle"
    RUNNING = "running"
    FAILED = "failed"


class AgentKind(str, enum.Enum):
    WIKI = "wiki"
    BOOK = "book"
    SYNTHESIS = "synthesis"


class JobType(str, enum.Enum):
    INGEST = "ingest"
    LINT = "lint"


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SynthesisRunKind(str, enum.Enum):
    CONTRADICTION = "contradiction"
    GAP = "gap"
    CONVERGENCE = "convergence"


class UserModel(Base):
    """Registered user; owns agents, jobs, chats, and synthesis runs."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, nullable=False, unique=True)
    password = Column(String, nullable=True)
    profile_picture = Column(String, nullable=True)
    username = Column(String, nullable=False, unique=True)

    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    agents = relationship("AgentModel", back_populates="user", cascade="all, delete-orphan")
    conversations = relationship("ConversationModel", back_populates="user", cascade="all, delete-orphan")
    jobs = relationship("JobModel", back_populates="user", cascade="all, delete-orphan")
    synthesis_runs = relationship("SynthesisRunModel", back_populates="user", cascade="all, delete-orphan")


class AgentModel(Base):
    """A wiki, book, or synthesis agent with pages, jobs, and optional synthesis links."""

    __tablename__ = "agents"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    status = Column(
        SQLEnum(AgentStatus, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=AgentStatus.IDLE,
    )
    kind = Column(
        SQLEnum(AgentKind, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=AgentKind.WIKI,
    )
    last_synthesis_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    user = relationship("UserModel", back_populates="agents", foreign_keys=[user_id])
    pages = relationship("PageModel", back_populates="agent", cascade="all, delete-orphan")
    conversations = relationship("ConversationModel", back_populates="agent", cascade="all, delete-orphan")
    sources = relationship("SourceModel", back_populates="agent", cascade="all, delete-orphan")
    jobs = relationship("JobModel", back_populates="agent", cascade="all, delete-orphan")
    synthesis_runs = relationship(
        "SynthesisRunModel",
        back_populates="synthesis_agent",
        foreign_keys="SynthesisRunModel.synthesis_agent_id",
        cascade="all, delete-orphan",
    )
    synthesis_run_involvements = relationship(
        "SynthesisRunInvolvedAgentModel",
        back_populates="wiki_agent",
        foreign_keys="SynthesisRunInvolvedAgentModel.wiki_agent_id",
        cascade="all, delete-orphan",
    )


class PageModel(Base):
    """Wiki page identified by path; unique per agent."""

    __tablename__ = "pages"
    __table_args__ = (UniqueConstraint("agent_id", "path", name="uq_pages_agent_path"),)

    id = Column(Integer, primary_key=True, index=True)
    path = Column(String, nullable=False, index=True)
    description = Column(String, nullable=True)
    content = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False, index=True)

    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    agent = relationship("AgentModel", back_populates="pages", foreign_keys=[agent_id])
    versions = relationship("PageVersionModel", back_populates="page", cascade="all, delete-orphan")


class PageVersionModel(Base):
    """Historical snapshot of page content; optional job_id links the write to a background job."""

    __tablename__ = "page_versions"
    __table_args__ = (UniqueConstraint("page_id", "version", name="uq_page_versions_page_version"),)

    id = Column(Integer, primary_key=True, index=True)
    page_id = Column(Integer, ForeignKey("pages.id"), nullable=False, index=True)
    version = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    job_id = Column(Integer, ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True, index=True)

    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    page = relationship("PageModel", back_populates="versions")
    job = relationship("JobModel", back_populates="page_versions")


class SourceModel(Base):
    """Uploaded file metadata for ingestion; lineage to pages goes through jobs and page versions."""

    __tablename__ = "sources"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False, index=True)
    storage_key = Column(String, nullable=True)
    original_filename = Column(String, nullable=True)
    content_type = Column(String, nullable=True)
    byte_size = Column(Integer, nullable=True)
    sha256 = Column(String(64), nullable=True, index=True)
    extra = Column(JSONB, nullable=True)

    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    agent = relationship("AgentModel", back_populates="sources")
    jobs = relationship("JobModel", back_populates="source")


class JobModel(Base):
    """Background work (ingest, lint, …); optional source; state_json holds orchestration progress."""

    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False, index=True)
    source_id = Column(Integer, ForeignKey("sources.id"), nullable=True, index=True)

    type = Column(
        SQLEnum(JobType, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    status = Column(
        SQLEnum(JobStatus, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
        default=JobStatus.PENDING,
    )
    state_json = Column(JSONB, nullable=True)
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    user = relationship("UserModel", back_populates="jobs")
    agent = relationship("AgentModel", back_populates="jobs")
    source = relationship("SourceModel", back_populates="jobs")
    page_versions = relationship("PageVersionModel", back_populates="job")


class SynthesisRunModel(Base):
    """One synthesis pass; the synthesis agent writes; wiki_involvements lists other agents read as sources."""

    __tablename__ = "synthesis_runs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    synthesis_agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False, index=True)

    kind = Column(
        SQLEnum(SynthesisRunKind, values_callable=lambda x: [e.value for e in x]),
        nullable=False,
    )
    summary = Column(Text, nullable=True)
    extra = Column(JSONB, nullable=True)

    started_at = Column(DateTime, server_default=func.now(), nullable=False)
    completed_at = Column(DateTime, nullable=True)

    user = relationship("UserModel", back_populates="synthesis_runs")
    synthesis_agent = relationship(
        "AgentModel",
        back_populates="synthesis_runs",
        foreign_keys=[synthesis_agent_id],
    )
    wiki_involvements = relationship(
        "SynthesisRunInvolvedAgentModel",
        back_populates="synthesis_run",
        cascade="all, delete-orphan",
    )


class SynthesisRunInvolvedAgentModel(Base):
    """Join table: which wiki agents were read as sources for a synthesis run."""

    __tablename__ = "synthesis_run_involved_agents"
    __table_args__ = (
        UniqueConstraint("synthesis_run_id", "wiki_agent_id", name="uq_synthesis_run_wiki_agent"),
    )

    id = Column(Integer, primary_key=True, index=True)
    synthesis_run_id = Column(Integer, ForeignKey("synthesis_runs.id"), nullable=False, index=True)
    wiki_agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False, index=True)

    synthesis_run = relationship("SynthesisRunModel", back_populates="wiki_involvements")
    wiki_agent = relationship("AgentModel", back_populates="synthesis_run_involvements")


class MessageModel(Base):
    """A single chat message in a conversation."""

    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    role = Column(SQLEnum(MessageRole, values_callable=lambda x: [e.value for e in x]), nullable=False)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False, index=True)

    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    conversation = relationship("ConversationModel", back_populates="messages")


class ConversationModel(Base):
    """Chat thread for a user against a specific wiki agent."""

    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False, index=True)

    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    user = relationship("UserModel", back_populates="conversations", foreign_keys=[user_id])
    agent = relationship("AgentModel", back_populates="conversations", foreign_keys=[agent_id])
    messages = relationship("MessageModel", back_populates="conversation", cascade="all, delete-orphan")
