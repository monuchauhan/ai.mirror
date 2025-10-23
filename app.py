"""
AI Mirror VTON - Main CLI Application

A production-ready virtual try-on pipeline with video extraction and VITON-HD processing.
Supports fallback to warp/blend algorithm when model is unavailable.

Usage:
    python -m app extract-frames --video input.mp4 --output frames/ --fps 20
    python -m app process-batch --frames frames/ --garment-front shirt.jpg --output results/
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.logging import RichHandler

# Import pipeline modules
from src.frames import extract_frames
from src.batch_worker import process_batch_parallel
from src.utils import setup_logging, validate_paths

app = typer.Typer(
    name="ai-mirror-vton",
    help="AI Mirror Virtual Try-On Pipeline",
    add_completion=False
)
console = Console()


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from YAML file with environment variable substitution.
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config_str = f.read()
            # Replace environment variables
            for key, value in os.environ.items():
                config_str = config_str.replace(f"${{{key}}}", value)
            config = yaml.safe_load(config_str)
        return config
    except FileNotFoundError:
        console.print(f"[yellow]Warning: Config file {config_path} not found. Using defaults.[/yellow]")
        return {}


@app.command()
def extract_frames_cmd(
    video: str = typer.Option(..., "--video", help="Input video file path"),
    output: str = typer.Option(..., "--output", help="Output directory for frames"),
    fps: int = typer.Option(30, "--fps", help="Target frames per second"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging")
):
    """
    Extract frames from video at specified FPS.
    
    Example:
        python -m app extract-frames --video input.mp4 --output frames/ --fps 20
    """
    # Setup logging
    log_level = logging.DEBUG if debug else logging.INFO
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Extracting frames from {video} at {fps} FPS")
    
    try:
        # Validate input
        if not os.path.exists(video):
            console.print(f"[red]Error: Video file not found: {video}[/red]")
            raise typer.Exit(code=1)
        
        # Create output directory
        os.makedirs(output, exist_ok=True)
        
        # Extract frames
        frame_paths = extract_frames(video, output, target_fps=fps)
        
        console.print(f"[green]✓[/green] Extracted {len(frame_paths)} frames to {output}")
        logger.info(f"Frame extraction complete: {len(frame_paths)} frames")
        
    except Exception as e:
        console.print(f"[red]Error during frame extraction: {e}[/red]")
        logger.exception("Frame extraction failed")
        raise typer.Exit(code=1)


@app.command()
def process_batch_cmd(
    frames: str = typer.Option(..., "--frames", help="Directory containing frames"),
    garment_front: str = typer.Option(..., "--garment-front", help="Front garment image"),
    garment_back: Optional[str] = typer.Option(None, "--garment-back", help="Back garment image (optional)"),
    output: str = typer.Option(..., "--output", help="Output directory"),
    viton_model_path: Optional[str] = typer.Option(None, "--viton-model-path", help="Path to VITON-HD checkpoint"),
    parallel: int = typer.Option(1, "--parallel", help="Number of parallel workers"),
    refine: bool = typer.Option(False, "--refine", help="Enable SDXL refinement"),
    debug: bool = typer.Option(False, "--debug", help="Save intermediate outputs"),
    config_file: str = typer.Option("config.yaml", "--config", help="Configuration file path")
):
    """
    Process batch of frames with virtual try-on.
    
    Example:
        python -m app process-batch \\
            --frames frames/ \\
            --garment-front shirt_front.jpg \\
            --garment-back shirt_back.jpg \\
            --output results/ \\
            --viton-model-path models/viton_hd \\
            --parallel 4 \\
            --refine \\
            --debug
    """
    # Load configuration
    config = load_config(config_file)
    
    # Setup logging
    log_level = logging.DEBUG if debug else logging.INFO
    setup_logging(log_level)
    logger = logging.getLogger(__name__)
    
    # Determine model path (CLI arg > env var > config)
    model_path = viton_model_path or os.getenv("MODEL_PATH") or config.get("model", {}).get("viton_hd_path")
    
    logger.info(f"Processing frames from {frames}")
    logger.info(f"Model path: {model_path or 'None (using fallback)'}")
    logger.info(f"Parallel workers: {parallel}")
    logger.info(f"Refinement: {refine}")
    logger.info(f"Debug mode: {debug}")
    
    try:
        # Validate inputs
        if not os.path.exists(frames):
            console.print(f"[red]Error: Frames directory not found: {frames}[/red]")
            raise typer.Exit(code=1)
        
        if not os.path.exists(garment_front):
            console.print(f"[red]Error: Garment front image not found: {garment_front}[/red]")
            raise typer.Exit(code=1)
        
        if garment_back and not os.path.exists(garment_back):
            console.print(f"[red]Error: Garment back image not found: {garment_back}[/red]")
            raise typer.Exit(code=1)
        
        # Create output directory
        os.makedirs(output, exist_ok=True)
        
        if debug:
            debug_dir = os.path.join(output, "debug")
            os.makedirs(debug_dir, exist_ok=True)
        else:
            debug_dir = None
        
        # Process batch
        console.print("[cyan]Starting batch processing...[/cyan]")
        
        results = process_batch_parallel(
            frames_dir=frames,
            garment_front_path=garment_front,
            garment_back_path=garment_back,
            output_dir=output,
            model_path=model_path,
            num_workers=parallel,
            enable_refine=refine,
            debug_dir=debug_dir,
            config=config
        )
        
        console.print(f"[green]✓[/green] Processed {len(results)} frames successfully")
        logger.info(f"Batch processing complete: {len(results)} frames")
        
    except Exception as e:
        console.print(f"[red]Error during batch processing: {e}[/red]")
        logger.exception("Batch processing failed")
        raise typer.Exit(code=1)


@app.command()
def info():
    """Display system and configuration information."""
    import torch
    
    console.print("\n[bold cyan]AI Mirror VTON - System Information[/bold cyan]\n")
    
    console.print(f"[bold]Python:[/bold] {sys.version.split()[0]}")
    console.print(f"[bold]PyTorch:[/bold] {torch.__version__}")
    console.print(f"[bold]CUDA Available:[/bold] {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        console.print(f"[bold]CUDA Version:[/bold] {torch.version.cuda}")
        console.print(f"[bold]GPU Count:[/bold] {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            console.print(f"  [bold]GPU {i}:[/bold] {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            console.print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
    else:
        console.print("[yellow]No CUDA devices detected[/yellow]")
    
    console.print(f"\n[bold]Model Path (env):[/bold] {os.getenv('MODEL_PATH', 'Not set')}")
    
    config = load_config()
    if config:
        console.print(f"[bold]Config loaded:[/bold] {len(config)} sections")
    
    console.print()


if __name__ == "__main__":
    app()
