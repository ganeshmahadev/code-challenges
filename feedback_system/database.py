"""Database connection and operations."""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import StaticPool
from config import config
from models import Base, Feedback
from schemas import AnalysisResult


# Create async engine
# StaticPool for SQLite to avoid threading issues
engine = create_async_engine(
    config.DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
    echo=False
)

# Create session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)


async def init_db():
    """Initialize database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """Dependency to get database session."""
    async with AsyncSessionLocal() as session:
        yield session


def get_db_session():
    """Get database session as context manager (for CLI usage).

    Returns:
        Async context manager for database session
    """
    return AsyncSessionLocal()


async def save_feedback(
    db: AsyncSession,
    feedback_text: str,
    analysis: AnalysisResult
) -> Feedback:
    """Save feedback and analysis to database.

    Args:
        db: Database session
        feedback_text: Original feedback text
        analysis: Analysis result

    Returns:
        Saved Feedback model
    """
    feedback = Feedback(
        text=feedback_text,
        sentiment=analysis.sentiment,
        topic=analysis.topic,
        confidence_score=analysis.confidence_score,
        alert_triggered=analysis.alert_triggered,
        processing_method=analysis.processing_method
    )

    db.add(feedback)
    await db.commit()
    await db.refresh(feedback)

    return feedback
