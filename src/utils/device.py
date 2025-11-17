#!/usr/bin/env python3
"""Device detection and management utilities.

This module provides automatic detection of available compute devices (CUDA, MPS, CPU)
for optimal hardware utilization across different platforms.
"""

import logging
from typing import Literal

logger = logging.getLogger(__name__)

DeviceType = Literal["cuda", "mps", "cpu"]


def detect_device() -> DeviceType:
    """Automatically detect the best available compute device.

    Detection order:
    1. CUDA (NVIDIA GPUs) - if PyTorch with CUDA is available
    2. MPS (Apple Silicon) - if running on macOS with MPS support
    3. CPU - fallback for all other cases

    Returns:
        Device type as string: "cuda", "mps", or "cpu"

    Examples:
        >>> device = detect_device()
        >>> print(f"Using device: {device}")
        Using device: cuda
    """
    # Try CUDA first (NVIDIA GPUs)
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA detected: {device_name}")
            return "cuda"
    except (ImportError, RuntimeError) as e:
        logger.debug(f"CUDA not available: {e}")

    # Try MPS (Apple Silicon)
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("MPS (Apple Silicon) detected")
            return "mps"
    except (ImportError, RuntimeError, AttributeError) as e:
        logger.debug(f"MPS not available: {e}")

    # Fallback to CPU
    logger.info("Using CPU (no GPU acceleration available)")
    return "cpu"


def get_device_info() -> dict:
    """Get detailed information about the current device.

    Returns:
        Dictionary containing device information:
        - device: Device type (cuda/mps/cpu)
        - name: Device name (e.g., "NVIDIA RTX 3090")
        - memory: Available memory in GB (for GPUs)
        - count: Number of available devices

    Examples:
        >>> info = get_device_info()
        >>> print(info)
        {'device': 'cuda', 'name': 'NVIDIA GeForce RTX 3090', 'memory': 24.0, 'count': 1}
    """
    device = detect_device()
    info = {
        "device": device,
        "name": None,
        "memory": None,
        "count": 1
    }

    try:
        import torch

        if device == "cuda":
            info["name"] = torch.cuda.get_device_name(0)
            info["memory"] = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
            info["count"] = torch.cuda.device_count()

        elif device == "mps":
            info["name"] = "Apple Silicon (MPS)"
            # MPS doesn't expose memory info directly
            info["memory"] = None
            info["count"] = 1

        else:  # cpu
            import platform
            info["name"] = platform.processor() or "CPU"
            import psutil
            info["memory"] = psutil.virtual_memory().total / 1e9  # GB
            info["count"] = 1

    except Exception as e:
        logger.warning(f"Could not get detailed device info: {e}")

    return info


def set_device_for_embedding_model(config: dict) -> str:
    """Set the optimal device for embedding models based on auto-detection.

    This function checks if GPU auto-detection should override the config setting.

    Args:
        config: Configuration dictionary containing embeddings settings

    Returns:
        Device string to use for embeddings

    Examples:
        >>> config = {'embeddings': {'device': 'auto'}}
        >>> device = set_device_for_embedding_model(config)
        >>> print(device)
        cuda
    """
    embeddings_config = config.get('embeddings', {})
    device_setting = embeddings_config.get('device', 'cpu')

    # If explicitly set to cuda/mps/cpu, respect the setting
    if device_setting in ['cuda', 'mps', 'cpu']:
        logger.info(f"Using explicitly configured device: {device_setting}")
        return device_setting

    # If set to 'auto', detect automatically
    if device_setting == 'auto':
        detected = detect_device()
        logger.info(f"Auto-detected device: {detected}")
        return detected

    # For any other value, auto-detect
    logger.warning(f"Unknown device setting '{device_setting}', auto-detecting...")
    return detect_device()


def format_device_info() -> str:
    """Format device information as a human-readable string.

    Returns:
        Formatted string with device information

    Examples:
        >>> print(format_device_info())
        Device: CUDA (NVIDIA GeForce RTX 3090)
        Memory: 24.0 GB
        Count: 1
    """
    info = get_device_info()

    lines = [f"Device: {info['device'].upper()}"]

    if info['name']:
        lines.append(f"Name: {info['name']}")

    if info['memory']:
        lines.append(f"Memory: {info['memory']:.1f} GB")

    if info['count'] > 1:
        lines.append(f"Count: {info['count']}")

    return "\n".join(lines)


if __name__ == "__main__":
    # Simple CLI for testing
    print("Device Detection Report")
    print("=" * 50)
    print(format_device_info())
    print("=" * 50)

    info = get_device_info()
    print(f"\nRecommended for embeddings: {info['device']}")
