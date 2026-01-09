#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table


console = Console()


def get_video_info(video_path: Path) -> dict:
    """Get video metadata using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        console.print(f"[red]Error reading video: {result.stderr}[/red]")
        sys.exit(1)

    return json.loads(result.stdout)


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}m {secs:.1f}s"
    elif minutes > 0:
        return f"{minutes}m {secs:.1f}s"
    else:
        return f"{secs:.1f}s"


def format_filesize(bytes_size: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.1f} TB"


def display_video_info(video_path: Path, info: dict) -> None:
    """Display video information in a nice table."""
    video_stream = next(
        (s for s in info.get("streams", []) if s.get("codec_type") == "video"),
        None
    )
    audio_stream = next(
        (s for s in info.get("streams", []) if s.get("codec_type") == "audio"),
        None
    )
    format_info = info.get("format", {})

    table = Table(title="Video Information", show_header=False, border_style="cyan")
    table.add_column("Property", style="bold")
    table.add_column("Value", style="green")

    # File info
    table.add_row("File", video_path.name)

    # Duration
    duration = float(format_info.get("duration", 0))
    table.add_row("Duration", format_duration(duration))

    # File size
    size = int(format_info.get("size", 0))
    table.add_row("Size", format_filesize(size))

    # Format
    format_name = format_info.get("format_long_name", format_info.get("format_name", "Unknown"))
    table.add_row("Format", format_name)

    if video_stream:
        # Resolution
        width = video_stream.get("width", "?")
        height = video_stream.get("height", "?")
        table.add_row("Resolution", f"{width}x{height}")

        # Video codec
        codec = video_stream.get("codec_long_name", video_stream.get("codec_name", "Unknown"))
        table.add_row("Video Codec", codec)

        # Frame rate
        fps = video_stream.get("r_frame_rate", "")
        if fps and "/" in fps:
            num, den = fps.split("/")
            fps_val = float(num) / float(den) if float(den) != 0 else 0
            table.add_row("Frame Rate", f"{fps_val:.2f} fps")

        # Bitrate
        bitrate = video_stream.get("bit_rate")
        if bitrate:
            table.add_row("Video Bitrate", f"{int(bitrate) // 1000} kbps")

    if audio_stream:
        audio_codec = audio_stream.get("codec_long_name", audio_stream.get("codec_name", "Unknown"))
        table.add_row("Audio Codec", audio_codec)

        sample_rate = audio_stream.get("sample_rate")
        if sample_rate:
            table.add_row("Sample Rate", f"{sample_rate} Hz")

    console.print()
    console.print(table)
    console.print()


def speed_up_video(input_path: Path, speed: float) -> Path:
    """Speed up video using ffmpeg."""
    stem = input_path.stem
    suffix = input_path.suffix

    # Format speed for filename (e.g., 2x, 1.5x)
    speed_str = f"{speed}x" if speed != int(speed) else f"{int(speed)}x"
    output_path = input_path.parent / f"{stem}-{speed_str}{suffix}"

    # Calculate the atempo filter chain for audio
    # atempo only accepts values between 0.5 and 2.0, so we chain them for higher speeds
    audio_filters = []
    remaining_speed = speed
    while remaining_speed > 2.0:
        audio_filters.append("atempo=2.0")
        remaining_speed /= 2.0
    if remaining_speed < 0.5:
        remaining_speed = 0.5
    audio_filters.append(f"atempo={remaining_speed}")
    audio_filter_str = ",".join(audio_filters)

    # Video filter: setpts divides timestamps to speed up
    video_filter = f"setpts={1/speed}*PTS"

    cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-filter:v", video_filter,
        "-filter:a", audio_filter_str,
        "-y",  # Overwrite output file if exists
        str(output_path)
    ]

    console.print(f"\n[cyan]Speeding up video by {speed_str}...[/cyan]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing video...", total=None)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        progress.update(task, completed=True)

    if result.returncode != 0:
        console.print(f"[red]Error processing video:[/red]")
        console.print(f"[red]{result.stderr}[/red]")
        sys.exit(1)

    return output_path


def main():
    console.print(Panel.fit(
        "[bold cyan]Video Speed Tool[/bold cyan]\n"
        "[dim]Speed up your videos with ease[/dim]",
        border_style="cyan"
    ))

    if len(sys.argv) < 2:
        console.print("[red]Usage: python main.py <video_path>[/red]")
        sys.exit(1)

    video_path = Path(sys.argv[1])

    if not video_path.exists():
        console.print(f"[red]Error: File not found: {video_path}[/red]")
        sys.exit(1)

    if not video_path.is_file():
        console.print(f"[red]Error: Not a file: {video_path}[/red]")
        sys.exit(1)

    # Get and display video info
    info = get_video_info(video_path)
    display_video_info(video_path, info)

    # Ask for speed
    while True:
        speed_input = Prompt.ask(
            "[bold]Enter speed multiplier[/bold]",
            default="2"
        )

        try:
            speed = float(speed_input)
            if speed <= 0:
                console.print("[red]Speed must be greater than 0[/red]")
                continue
            if speed > 100:
                console.print("[red]Speed must be 100 or less[/red]")
                continue
            break
        except ValueError:
            console.print("[red]Please enter a valid number[/red]")

    # Process video
    output_path = speed_up_video(video_path, speed)

    # Show result
    console.print()
    console.print(Panel.fit(
        f"[green]Video saved to:[/green]\n[bold]{output_path}[/bold]",
        border_style="green",
        title="Complete"
    ))


if __name__ == "__main__":
    main()
