#!/usr/bin/env python3
"""Interactive CLI for testing the feedback analysis system.

This allows users to:
1. Enter feedback directly in the terminal
2. See analysis results immediately
3. See visual alert notifications
4. Test the system without making HTTP requests
"""
import asyncio
import sys
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.prompt import Prompt

from config import config
from database import init_db, get_db_session, save_feedback
from ai_analyzer import AIAnalyzer
from ml_fallback_analyzer import MLFallbackAnalyzer
from cache import FeedbackCache
from alerting import AlertService


console = Console()


class InteractiveFeedbackSystem:
    """Interactive feedback analysis system."""

    def __init__(self):
        """Initialize the system."""
        self.ai_analyzer = AIAnalyzer()
        self.fallback_analyzer = MLFallbackAnalyzer()
        self.cache = FeedbackCache()
        self.alert_service = AlertService()

    async def analyze_feedback(self, feedback_text: str):
        """Analyze feedback and return results.

        Args:
            feedback_text: The feedback to analyze

        Returns:
            Analysis result
        """
        # Check cache
        cached_result = self.cache.get(feedback_text)
        if cached_result:
            console.print("[cyan]💾 Result from cache[/cyan]")
            return cached_result, True

        # Try AI analysis
        try:
            if config.AI_PROVIDER_ENABLED:
                console.print("[yellow]🤖 Analyzing with AI...[/yellow]")
                result = await self.ai_analyzer.analyze(feedback_text)
            else:
                raise Exception("AI disabled")
        except Exception as e:
            console.print(f"[yellow] AI unavailable ({str(e)}), using ML fallback...[/yellow]")
            result = self.fallback_analyzer.analyze(feedback_text)

        # Cache the result
        self.cache.set(feedback_text, result)
        return result, False

    def display_result(self, result, cached=False):
        """Display analysis results in a nice format.

        Args:
            result: Analysis result
            cached: Whether result was from cache
        """
        # Create results table
        table = Table(
            title="📊 Analysis Results",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta"
        )

        table.add_column("Attribute", style="cyan", width=20)
        table.add_column("Value", style="green")

        
        table.add_row("Sentiment", f"{result.sentiment}")
        table.add_row("Topic", result.topic)
        table.add_row("Confidence", result.confidence_score)
        table.add_row("Processing Method", result.processing_method)
        table.add_row("From Cache", "Yes" if cached else "No")

        console.print(table)

        # Show alert if triggered
        if result.alert_triggered:
            self.display_alert(result)

    def display_alert(self, result):
        """Display alert notification.

        Args:
            result: Analysis result that triggered alert
        """
        alert_content = f"""
[bold red] ALERT TRIGGERED![/bold red]

This feedback indicates customer dissatisfaction!

[bold]Why:[/bold] Sentiment is {result.sentiment} (customer is unhappy)
[bold]Sentiment:[/bold] {result.sentiment}
[bold]Topic:[/bold] {result.topic}
[bold]Confidence:[/bold] {result.confidence_score}

[bold yellow]Action Required:[/bold yellow]
→ Review this feedback immediately
→ Contact customer within 24 hours
→ Escalate to appropriate team
        """

        panel = Panel(
            alert_content,
            title="🚨 ALERT 🚨",
            border_style="bold red",
            box=box.DOUBLE,
            padding=(1, 2)
        )

        console.print()
        console.print(panel)
        console.print()

    async def save_to_database(self, feedback_text, result):
        """Save feedback to database.

        Args:
            feedback_text: Original feedback
            result: Analysis result

        Returns:
            Saved feedback ID
        """
        async with get_db_session() as db:
            feedback = await save_feedback(db, feedback_text, result)
            return feedback.id

    def display_welcome(self):
        """Display welcome message."""
        welcome = """
[bold cyan]Customer Feedback Analysis System[/bold cyan]
[dim]Interactive CLI Mode[/dim]

This system will analyze your feedback for:
  • Sentiment (positive, negative, neutral, mixed)
  • Topic (billing, technical, support, etc.)
  • Alert triggers

Using:
  [green]✓[/green] AI: GPT-3.5-turbo (if available)
  [green]✓[/green] Fallback: RoBERTa sentiment model
        """

        panel = Panel(
            welcome,
            border_style="bold blue",
            box=box.DOUBLE,
            padding=(1, 2)
        )

        console.print(panel)
        console.print()

    async def run_interactive(self):
        """Run the interactive CLI loop."""
        self.display_welcome()

        while True:
            console.print()
            console.print("[bold]Enter feedback to analyze[/bold] (or 'quit' to exit):")
            console.print()

            feedback_text = Prompt.ask("Your feedback")

            if feedback_text.lower() in ['quit', 'exit', 'q']:
                console.print("\n[cyan]Goodbye![/cyan]\n")
                break

            if not feedback_text.strip():
                console.print("[red]⚠️  Feedback cannot be empty[/red]")
                continue

            console.print()
            console.print("[bold]Processing your feedback...[/bold]")
            console.print()

            # Analyze
            result, cached = await self.analyze_feedback(feedback_text)

            # Display results
            self.display_result(result, cached)

            # Save to database
            try:
                feedback_id = await self.save_to_database(feedback_text, result)
                console.print(f"\n[dim]💾 Saved to database with ID: {feedback_id}[/dim]")

                # Send Slack alert if triggered
                if result.alert_triggered:
                    try:
                        await self.alert_service.send_alert(feedback_id, feedback_text, result)
                        console.print(f"[green]✅ Slack alert sent to webhook[/green]")
                    except Exception as alert_error:
                        console.print(f"[yellow]⚠️  Could not send Slack alert: {alert_error}[/yellow]")
            except Exception as e:
                console.print(f"\n[yellow]⚠️  Could not save to database: {e}[/yellow]")

            # Show cache stats
            stats = self.cache.get_stats()
            console.print(
                f"\n[dim]Cache: {stats['hits']} hits, "
                f"{stats['misses']} misses, "
                f"{stats['size']} entries[/dim]"
            )


async def main():
    """Main entry point."""
    # Initialize database
    console.print("[cyan]Initializing database...[/cyan]")
    await init_db()

    # Run interactive system
    system = InteractiveFeedbackSystem()

    try:
        await system.run_interactive()
    except KeyboardInterrupt:
        console.print("\n\n[cyan] Goodbye![/cyan]\n")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
