"""Main FastAPI application for feedback analysis."""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Header, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from config import config
from database import init_db, get_db, save_feedback
from schemas import FeedbackRequest, FeedbackResponse, AnalysisResult
from ai_analyzer import AIAnalyzer
from ml_fallback_analyzer import MLFallbackAnalyzer
from cache import FeedbackCache
from alerting import AlertService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize services
ai_analyzer = AIAnalyzer()
fallback_analyzer = MLFallbackAnalyzer()  # Using ML-based fallback
cache = FeedbackCache()
alert_service = AlertService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    logger.info("Initializing database...")
    await init_db()
    logger.info("Application started successfully")
    yield
    # Shutdown
    logger.info("Application shutting down")


app = FastAPI(
    title="Customer Feedback Analysis API",
    description="AI-powered customer feedback sentiment and topic analysis",
    version="1.0.0",
    lifespan=lifespan
)


async def verify_api_key(x_api_key: str = Header(...)) -> None:
    """Verify API key authentication.

    Stubbed authentication per assignment requirements.
    In production, this would validate against a proper auth system.
    """
    if x_api_key != config.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )


@app.post("/feedback", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED)
async def analyze_feedback(
    request: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(verify_api_key)
):
    """Analyze customer feedback and store results.

    This endpoint:
    1. Checks cache for identical feedback
    2. Attempts AI analysis (with timeout)
    3. Falls back to rule-based analysis if AI fails
    4. Stores results in database
    5. Triggers alerts if needed
    6. Returns results within 500ms

    Args:
        request: Feedback request with text
        db: Database session
        _: API key verification

    Returns:
        FeedbackResponse with analysis results
    """
    feedback_text = request.text.strip()

    # Step 1: Check cache
    cached_result = cache.get(feedback_text)
    if cached_result:
        logger.info("Cache hit for feedback")
        # Still save to database even if cached (for analytics)
        feedback = await save_feedback(db, feedback_text, cached_result)

        # Alert if needed (idempotent check)
        if cached_result.alert_triggered:
            await alert_service.send_alert(feedback.id, feedback_text, cached_result)

        return FeedbackResponse(
            id=feedback.id,
            sentiment=cached_result.sentiment,
            topic=cached_result.topic,
            confidence_score=cached_result.confidence_score,
            alert_triggered=cached_result.alert_triggered,
            processing_method=cached_result.processing_method,
            created_at=feedback.created_at.isoformat()
        )

    # Step 2: Try AI analysis
    analysis_result: AnalysisResult
    try:
        if config.AI_PROVIDER_ENABLED:
            logger.info("Attempting AI analysis")
            analysis_result = await ai_analyzer.analyze(feedback_text)
            logger.info(f"AI analysis successful: {analysis_result.sentiment}/{analysis_result.topic}")
        else:
            raise Exception("AI provider disabled in config")

    except Exception as e:
        # Step 3: Fallback to rule-based analysis
        logger.warning(f"AI analysis failed: {e}. Using fallback analyzer")
        analysis_result = fallback_analyzer.analyze(feedback_text)
        logger.info(f"Fallback analysis: {analysis_result.sentiment}/{analysis_result.topic}")

    # Step 4: Cache the result
    cache.set(feedback_text, analysis_result)

    # Step 5: Save to database
    feedback = await save_feedback(db, feedback_text, analysis_result)

    # Step 6: Send alert if needed
    if analysis_result.alert_triggered:
        await alert_service.send_alert(feedback.id, feedback_text, analysis_result)

    return FeedbackResponse(
        id=feedback.id,
        sentiment=analysis_result.sentiment,
        topic=analysis_result.topic,
        confidence_score=analysis_result.confidence_score,
        alert_triggered=analysis_result.alert_triggered,
        processing_method=analysis_result.processing_method,
        created_at=feedback.created_at.isoformat()
    )


@app.get("/health")
async def health_check():
    """Health check endpoint.

    Returns system status including AI availability and cache stats.
    """
    ai_status = "healthy" if config.AI_PROVIDER_ENABLED and ai_analyzer.client else "degraded"

    return {
        "status": "healthy",
        "ai_provider": ai_status,
        "cache_stats": cache.get_stats()
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Customer Feedback Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "POST /feedback",
            "health": "GET /health"
        }
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unexpected errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )
