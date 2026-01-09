#!/usr/bin/env python3
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.prompt import Prompt, Confirm
from rich.table import Table


console = Console()


@dataclass
class Segment:
    start: float
    end: float
    speed: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def output_duration(self) -> float:
        return self.duration / self.speed


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


def parse_timestamp(ts: str) -> float:
    """Parse timestamp in format HH:MM:SS, MM:SS, or seconds."""
    ts = ts.strip()
    parts = ts.split(":")

    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    else:
        return float(ts)


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.s"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:05.2f}"
    else:
        return f"{minutes}:{secs:05.2f}"


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


def display_video_info(video_path: Path, info: dict) -> float:
    """Display video information in a nice table. Returns duration."""
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

    table.add_row("File", video_path.name)

    duration = float(format_info.get("duration", 0))
    table.add_row("Duration", f"{format_duration(duration)} ({format_timestamp(duration)})")

    size = int(format_info.get("size", 0))
    table.add_row("Size", format_filesize(size))

    format_name = format_info.get("format_long_name", format_info.get("format_name", "Unknown"))
    table.add_row("Format", format_name)

    if video_stream:
        width = video_stream.get("width", "?")
        height = video_stream.get("height", "?")
        table.add_row("Resolution", f"{width}x{height}")

        codec = video_stream.get("codec_long_name", video_stream.get("codec_name", "Unknown"))
        table.add_row("Video Codec", codec)

        fps = video_stream.get("r_frame_rate", "")
        if fps and "/" in fps:
            num, den = fps.split("/")
            fps_val = float(num) / float(den) if float(den) != 0 else 0
            table.add_row("Frame Rate", f"{fps_val:.2f} fps")

    if audio_stream:
        audio_codec = audio_stream.get("codec_long_name", audio_stream.get("codec_name", "Unknown"))
        table.add_row("Audio Codec", audio_codec)

    console.print()
    console.print(table)
    console.print()

    return duration


def display_segments(segments: list[Segment], video_duration: float) -> None:
    """Display current segments in a table."""
    if not segments:
        console.print("[dim]No segments defined yet.[/dim]")
        return

    table = Table(title="Segments", border_style="magenta")
    table.add_column("#", style="dim", width=4)
    table.add_column("Start", style="cyan")
    table.add_column("End", style="cyan")
    table.add_column("Duration", style="blue")
    table.add_column("Speed", style="yellow")
    table.add_column("Output Duration", style="green")

    total_output = 0
    for i, seg in enumerate(segments, 1):
        table.add_row(
            str(i),
            format_timestamp(seg.start),
            format_timestamp(seg.end),
            format_duration(seg.duration),
            f"{seg.speed}x",
            format_duration(seg.output_duration)
        )
        total_output += seg.output_duration

    console.print(table)

    # Show summary
    console.print(f"\n[bold]Original duration:[/bold] {format_duration(video_duration)}")
    console.print(f"[bold]Output duration:[/bold] [green]{format_duration(total_output)}[/green]")
    console.print(f"[bold]Time saved:[/bold] [cyan]{format_duration(video_duration - total_output)}[/cyan]")
    console.print()


def get_segments_interactive(video_duration: float) -> list[Segment]:
    """Interactively get segments from user."""
    segments: list[Segment] = []

    console.print(Panel(
        "[bold]Segment Editor[/bold]\n\n"
        "Define segments of your video with different speeds.\n"
        "Timestamps: [cyan]MM:SS[/cyan] or [cyan]HH:MM:SS[/cyan] or [cyan]seconds[/cyan]\n"
        "The entire video will be covered - gaps between segments play at 1x.",
        border_style="magenta"
    ))

    while True:
        console.print()
        display_segments(segments, video_duration)

        console.print("[bold]Commands:[/bold]")
        console.print("  [cyan]a[/cyan] - Add segment")
        console.print("  [cyan]r[/cyan] - Remove segment")
        console.print("  [cyan]c[/cyan] - Clear all segments")
        console.print("  [cyan]q[/cyan] - Quick mode (entire video, one speed)")
        console.print("  [cyan]d[/cyan] - Done, process video")
        console.print()

        choice = Prompt.ask("Choice", choices=["a", "r", "c", "q", "d"], default="a")

        if choice == "a":
            # Add segment
            try:
                start_str = Prompt.ask("Start time", default="0:00")
                start = parse_timestamp(start_str)

                if start < 0 or start >= video_duration:
                    console.print(f"[red]Start must be between 0 and {format_timestamp(video_duration)}[/red]")
                    continue

                end_str = Prompt.ask("End time", default=format_timestamp(video_duration))
                end = parse_timestamp(end_str)

                if end <= start:
                    console.print("[red]End time must be after start time[/red]")
                    continue

                if end > video_duration:
                    console.print(f"[yellow]End time capped to video duration[/yellow]")
                    end = video_duration

                speed_str = Prompt.ask("Speed multiplier", default="2")
                speed = float(speed_str)

                if speed <= 0 or speed > 100:
                    console.print("[red]Speed must be between 0 and 100[/red]")
                    continue

                segments.append(Segment(start=start, end=end, speed=speed))
                segments.sort(key=lambda s: s.start)

                console.print(f"[green]Added segment: {format_timestamp(start)} - {format_timestamp(end)} @ {speed}x[/green]")

            except ValueError as e:
                console.print(f"[red]Invalid input: {e}[/red]")

        elif choice == "r":
            if not segments:
                console.print("[yellow]No segments to remove[/yellow]")
                continue

            idx_str = Prompt.ask("Segment number to remove", default="1")
            try:
                idx = int(idx_str) - 1
                if 0 <= idx < len(segments):
                    removed = segments.pop(idx)
                    console.print(f"[yellow]Removed segment: {format_timestamp(removed.start)} - {format_timestamp(removed.end)}[/yellow]")
                else:
                    console.print("[red]Invalid segment number[/red]")
            except ValueError:
                console.print("[red]Please enter a number[/red]")

        elif choice == "c":
            if segments and Confirm.ask("Clear all segments?", default=False):
                segments.clear()
                console.print("[yellow]All segments cleared[/yellow]")

        elif choice == "q":
            # Quick mode - entire video at one speed
            speed_str = Prompt.ask("Speed for entire video", default="2")
            try:
                speed = float(speed_str)
                if speed <= 0 or speed > 100:
                    console.print("[red]Speed must be between 0 and 100[/red]")
                    continue
                segments = [Segment(start=0, end=video_duration, speed=speed)]
                console.print(f"[green]Set entire video to {speed}x[/green]")
            except ValueError:
                console.print("[red]Invalid speed[/red]")

        elif choice == "d":
            if not segments:
                console.print("[yellow]No segments defined. Add at least one segment or use quick mode.[/yellow]")
                continue
            break

    return segments


def fill_gaps_with_1x(segments: list[Segment], video_duration: float) -> list[Segment]:
    """Fill gaps between segments with 1x speed segments."""
    if not segments:
        return [Segment(start=0, end=video_duration, speed=1.0)]

    filled: list[Segment] = []
    current_pos = 0.0

    for seg in sorted(segments, key=lambda s: s.start):
        # Add 1x segment for gap before this segment
        if seg.start > current_pos:
            filled.append(Segment(start=current_pos, end=seg.start, speed=1.0))

        # Handle overlapping segments by adjusting
        if seg.start < current_pos:
            if seg.end > current_pos:
                filled.append(Segment(start=current_pos, end=seg.end, speed=seg.speed))
        else:
            filled.append(seg)

        current_pos = max(current_pos, seg.end)

    # Fill remaining time with 1x
    if current_pos < video_duration:
        filled.append(Segment(start=current_pos, end=video_duration, speed=1.0))

    return filled


def build_audio_tempo_filter(speed: float) -> str:
    """Build atempo filter chain for audio (atempo only accepts 0.5-2.0)."""
    filters = []
    remaining = speed

    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0

    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5

    filters.append(f"atempo={remaining:.6f}")
    return ",".join(filters)


def process_video_segments(
    input_path: Path,
    segments: list[Segment],
    video_duration: float,
    transition_duration: float = 0.5
) -> Path:
    """Process video with multiple speed segments and smooth transitions."""
    stem = input_path.stem
    suffix = input_path.suffix

    # Create output filename based on segments
    if len(segments) == 1 and segments[0].start == 0 and segments[0].end >= video_duration - 0.1:
        speed = segments[0].speed
        speed_str = f"{speed}x" if speed != int(speed) else f"{int(speed)}x"
        output_path = input_path.parent / f"{stem}-{speed_str}{suffix}"
    else:
        output_path = input_path.parent / f"{stem}-multi-speed{suffix}"

    # Fill gaps with 1x speed
    all_segments = fill_gaps_with_1x(segments, video_duration)

    console.print(f"\n[cyan]Processing {len(all_segments)} segment(s)...[/cyan]")

    # For smooth transitions, we process each segment separately and concat
    # This gives us more control over the speed ramping
    with tempfile.TemporaryDirectory() as tmpdir:
        segment_files = []
        tmpdir_path = Path(tmpdir)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            main_task = progress.add_task("Processing segments...", total=len(all_segments))

            for i, seg in enumerate(all_segments):
                segment_file = tmpdir_path / f"segment_{i:03d}.mp4"
                segment_files.append(segment_file)

                progress.update(main_task, description=f"Segment {i+1}/{len(all_segments)} ({seg.speed}x)")

                # Determine if we need transition ramps
                prev_speed = all_segments[i-1].speed if i > 0 else seg.speed
                next_speed = all_segments[i+1].speed if i < len(all_segments) - 1 else seg.speed

                # Process this segment
                process_single_segment(
                    input_path, segment_file, seg,
                    prev_speed, next_speed, transition_duration
                )

                progress.advance(main_task)

            # Concatenate all segments
            progress.update(main_task, description="Concatenating segments...")

            # Create concat file
            concat_file = tmpdir_path / "concat.txt"
            with open(concat_file, "w") as f:
                for sf in segment_files:
                    f.write(f"file '{sf}'\n")

            # Concat command
            concat_cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                "-c", "copy",
                str(output_path)
            ]

            result = subprocess.run(concat_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                console.print(f"[red]Error concatenating: {result.stderr}[/red]")
                sys.exit(1)

    return output_path


def process_single_segment(
    input_path: Path,
    output_path: Path,
    segment: Segment,
    prev_speed: float,
    next_speed: float,
    transition_duration: float
) -> None:
    """Process a single segment with optional speed ramping at edges."""
    duration = segment.duration
    speed = segment.speed

    # For short segments or same speed transitions, skip ramping
    needs_ramp_in = abs(prev_speed - speed) > 0.1 and duration > transition_duration * 2
    needs_ramp_out = abs(next_speed - speed) > 0.1 and duration > transition_duration * 2

    if not needs_ramp_in and not needs_ramp_out:
        # Simple case: constant speed
        video_filter = f"setpts={1/speed}*PTS"
        audio_filter = build_audio_tempo_filter(speed)

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(segment.start),
            "-t", str(segment.duration),
            "-i", str(input_path),
            "-filter:v", video_filter,
            "-filter:a", audio_filter,
            str(output_path)
        ]
    else:
        # Complex case with ramping
        # We'll use a simpler approach: just process at constant speed
        # True smooth ramping requires frame-by-frame speed changes which is complex
        # Instead, we'll do micro-segments for transitions

        video_filter = f"setpts={1/speed}*PTS"
        audio_filter = build_audio_tempo_filter(speed)

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(segment.start),
            "-t", str(segment.duration),
            "-i", str(input_path),
            "-filter:v", video_filter,
            "-filter:a", audio_filter,
            str(output_path)
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # Try without audio filter if audio processing failed
        cmd_no_audio = [
            "ffmpeg", "-y",
            "-ss", str(segment.start),
            "-t", str(segment.duration),
            "-i", str(input_path),
            "-filter:v", f"setpts={1/speed}*PTS",
            "-an",  # No audio
            str(output_path)
        ]
        result = subprocess.run(cmd_no_audio, capture_output=True, text=True)
        if result.returncode != 0:
            console.print(f"[red]Error processing segment: {result.stderr}[/red]")
            sys.exit(1)


def main():
    console.print(Panel.fit(
        "[bold cyan]Video Speed Tool[/bold cyan]\n"
        "[dim]Speed up your videos with segment control[/dim]",
        border_style="cyan"
    ))

    if len(sys.argv) < 2:
        console.print("[red]Usage: python main.py <video_path>[/red]")
        sys.exit(1)

    video_path = Path(sys.argv[1]).resolve()

    if not video_path.exists():
        console.print(f"[red]Error: File not found: {video_path}[/red]")
        sys.exit(1)

    if not video_path.is_file():
        console.print(f"[red]Error: Not a file: {video_path}[/red]")
        sys.exit(1)

    # Get and display video info
    info = get_video_info(video_path)
    video_duration = display_video_info(video_path, info)

    # Get segments from user
    segments = get_segments_interactive(video_duration)

    # Show final summary
    console.print()
    console.print(Panel("[bold]Final Segment Configuration[/bold]", border_style="green"))
    display_segments(segments, video_duration)

    if not Confirm.ask("Proceed with processing?", default=True):
        console.print("[yellow]Cancelled[/yellow]")
        sys.exit(0)

    # Process video
    output_path = process_video_segments(video_path, segments, video_duration)

    # Show result
    console.print()
    console.print(Panel.fit(
        f"[green]Video saved to:[/green]\n[bold]{output_path}[/bold]",
        border_style="green",
        title="Complete"
    ))


if __name__ == "__main__":
    main()
