"""
Version compatibility checking between vLLM and vLLM-Omni.

This module provides utilities to check if the installed vLLM version
is compatible with the current vLLM-Omni version to prevent startup
failures due to version mismatches.
"""

import importlib.metadata
import logging
import os
import sys
from typing import Literal

logger = logging.getLogger(__name__)


def parse_major_version(version: str) -> tuple[int, int]:
    """
    Extract major version number from a version string.

    Handles various version formats:
    - Standard: "0.14.0" -> (0, 14)
    - With device suffix: "0.14.0+cuda" -> (0, 14)
    - Dev versions: "0.14.1.dev23+g1a2b3c4" -> (0, 14)
    - With device on dev: "0.14.1.dev23+g1a2b3c4.rocm" -> (0, 14)
    - Future versions: "1.2.3" -> (1, 2)

    Args:
        version: Version string to parse

    Returns:
        Tuple of (major, minor) version numbers. For semantic versioning X.Y.Z,
        returns (X, Y) to properly handle both 0.x.y and future 1.x.y versions.

    Raises:
        ValueError: If version string cannot be parsed
    """
    try:
        # Remove device suffix if present (e.g., "+cuda", ".rocm")
        base_version = version.split("+")[0]

        # Remove dev suffix if present (e.g., ".dev23")
        base_version = base_version.split(".dev")[0]

        # Split version components
        parts = base_version.split(".")

        if len(parts) < 2:
            raise ValueError(f"Invalid version format: {version}")

        # Return (major, minor) tuple to handle both 0.x.y and 1.x.y
        return (int(parts[0]), int(parts[1]))

    except (IndexError, ValueError) as e:
        raise ValueError(f"Failed to parse version '{version}': {e}") from e


def check_vllm_compatibility(action: Literal["warn", "error"] = "error") -> bool:
    """
    Check if the installed vLLM version is compatible with vLLM-Omni.

    Compares the major versions of vLLM and vLLM-Omni. A mismatch in major
    versions typically indicates incompatible APIs and can lead to startup
    failures or runtime errors.

    Args:
        action: Action to take on version mismatch:
            - "error": Log error and exit with status code 1
            - "warn": Log warning but continue execution

    Returns:
        True if versions are compatible or check is skipped, False if mismatch detected

    Environment Variables:
        VLLM_OMNI_VERSION_CHECK: Override the action parameter
            ("error" or "warn")
    """
    # Allow environment variable to override action
    action = os.getenv("VLLM_OMNI_VERSION_CHECK", action)  # type: ignore

    try:
        # Get vLLM version
        try:
            vllm_version = importlib.metadata.version("vllm")
        except importlib.metadata.PackageNotFoundError:
            logger.error(
                "vLLM is not installed. vLLM-Omni requires vLLM to be installed.\n"
                "Please install vLLM first: pip install vllm"
            )
            if action == "error":
                sys.exit(1)
            return False

        # Get vLLM-Omni version
        from vllm_omni.version import __version__ as omni_version

        # Parse major versions
        try:
            vllm_major = parse_major_version(vllm_version)
            omni_major = parse_major_version(omni_version)
        except ValueError as e:
            logger.warning(f"Failed to parse version numbers: {e}")
            return True  # Don't block on parse failure

        # Check compatibility
        if vllm_major != omni_major:
            msg = (
                f"\n{'=' * 70}\n"
                f"VERSION MISMATCH DETECTED!\n"
                f"{'=' * 70}\n"
                f"  vLLM version:       {vllm_version} (major: {vllm_major[0]}.{vllm_major[1]})\n"
                f"  vLLM-Omni version:  {omni_version} (major: {omni_major[0]}.{omni_major[1]})\n"
                f"\n"
                f"Major version mismatch may cause startup failures or runtime errors.\n"
                f"Please install matching versions:\n"
                f"  - For vLLM {vllm_major[0]}.{vllm_major[1]}.x: pip install vllm-omni~={vllm_major[0]}.{vllm_major[1]}.0\n"
                f"  - For vLLM-Omni {omni_major[0]}.{omni_major[1]}.x: pip install vllm~={omni_major[0]}.{omni_major[1]}.0\n"
                f"\n"
                f"To bypass this check (not recommended):\n"
                f"  export VLLM_OMNI_VERSION_CHECK=warn  # Show warning only\n"
                f"{'=' * 70}\n"
            )

            if action == "error":
                logger.error(msg)
                sys.exit(1)
            else:
                logger.warning(msg)
            return False

        logger.info(f"Version compatibility check passed: vLLM {vllm_version}, vLLM-Omni {omni_version}")
        return True

    except Exception as e:
        # Don't block startup on unexpected errors in version checking
        logger.warning(f"Failed to check vLLM compatibility: {e}")
        return True


__all__ = ["check_vllm_compatibility", "parse_major_version"]
