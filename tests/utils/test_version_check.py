# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for version compatibility checking.
"""

import importlib.metadata
from unittest.mock import MagicMock, patch

import pytest

from vllm_omni.utils.version_check import check_vllm_compatibility, parse_major_version


@pytest.mark.core_model
@pytest.mark.cpu
class TestParseMajorVersion:
    """Test cases for parse_major_version function."""

    def test_standard_version(self) -> None:
        """Test parsing standard version format."""
        assert parse_major_version("0.14.0") == (0, 14)
        assert parse_major_version("0.15.2") == (0, 15)
        assert parse_major_version("0.16.0") == (0, 16)

    def test_future_version(self) -> None:
        """Test parsing future 1.x.y versions."""
        assert parse_major_version("1.0.0") == (1, 0)
        assert parse_major_version("1.2.3") == (1, 2)
        assert parse_major_version("2.5.1") == (2, 5)

    def test_version_with_cuda_suffix(self) -> None:
        """Test parsing version with CUDA device suffix."""
        assert parse_major_version("0.14.0+cuda") == (0, 14)
        assert parse_major_version("0.15.1+cuda") == (0, 15)
        assert parse_major_version("1.0.0+cuda") == (1, 0)

    def test_version_with_rocm_suffix(self) -> None:
        """Test parsing version with ROCm device suffix."""
        assert parse_major_version("0.14.0+rocm") == (0, 14)
        assert parse_major_version("0.15.2.rocm") == (0, 15)

    def test_dev_version(self) -> None:
        """Test parsing development version."""
        assert parse_major_version("0.14.1.dev23") == (0, 14)
        assert parse_major_version("0.15.0.dev5") == (0, 15)
        assert parse_major_version("1.2.3.dev10") == (1, 2)

    def test_dev_version_with_git_hash(self) -> None:
        """Test parsing dev version with git hash."""
        assert parse_major_version("0.14.1.dev23+g1a2b3c4") == (0, 14)
        assert parse_major_version("0.15.2.dev10+gabcdef0") == (0, 15)

    def test_dev_version_with_device_suffix(self) -> None:
        """Test parsing dev version with device suffix."""
        assert parse_major_version("0.14.1.dev23+g1a2b3c4.rocm") == (0, 14)
        assert parse_major_version("0.15.0.dev5.npu") == (0, 15)

    def test_invalid_version_format(self) -> None:
        """Test that invalid version formats raise ValueError."""
        with pytest.raises(ValueError, match="Invalid version format"):
            parse_major_version("invalid")

        with pytest.raises(ValueError, match="Invalid version format"):
            parse_major_version("1")

        with pytest.raises(ValueError, match="Failed to parse version"):
            parse_major_version("0.x.y")


@pytest.mark.core_model
@pytest.mark.cpu
class TestCheckVllmCompatibility:
    """Test cases for check_vllm_compatibility function."""

    @patch("vllm_omni.utils.version_check.importlib.metadata.version")
    @patch("vllm_omni.version.__version__", "0.14.5+cuda")
    @patch("vllm_omni.utils.version_check.logger")
    def test_compatible_versions(self, mock_logger: MagicMock, mock_version: MagicMock) -> None:
        """Test that compatible versions pass the check."""
        # Mock vLLM version
        mock_version.return_value = "0.14.0"

        result = check_vllm_compatibility(action="error")

        assert result is True
        mock_logger.info.assert_called_once()
        assert "Version compatibility check passed" in mock_logger.info.call_args[0][0]

    @patch("vllm_omni.utils.version_check.importlib.metadata.version")
    @patch("vllm_omni.version.__version__", "0.15.0+cuda")
    @patch("vllm_omni.utils.version_check.logger")
    @patch("vllm_omni.utils.version_check.sys.exit")
    def test_incompatible_versions_error_action(
        self, mock_exit: MagicMock, mock_logger: MagicMock, mock_version: MagicMock
    ) -> None:
        """Test that incompatible versions with error action exit."""
        # Mock vLLM version
        mock_version.return_value = "0.14.0"

        result = check_vllm_compatibility(action="error")

        assert result is False
        mock_logger.error.assert_called_once()
        assert "VERSION MISMATCH DETECTED" in mock_logger.error.call_args[0][0]
        mock_exit.assert_called_once_with(1)

    @patch("vllm_omni.utils.version_check.importlib.metadata.version")
    @patch("vllm_omni.version.__version__", "0.15.0+cuda")
    @patch("vllm_omni.utils.version_check.logger")
    def test_incompatible_versions_warn_action(self, mock_logger: MagicMock, mock_version: MagicMock) -> None:
        """Test that incompatible versions with warn action only warn."""
        # Mock vLLM version
        mock_version.return_value = "0.14.0"

        result = check_vllm_compatibility(action="warn")

        assert result is False
        mock_logger.warning.assert_called_once()
        assert "VERSION MISMATCH DETECTED" in mock_logger.warning.call_args[0][0]

    @patch("vllm_omni.utils.version_check.importlib.metadata.version")
    @patch("vllm_omni.utils.version_check.logger")
    @patch("vllm_omni.utils.version_check.sys.exit")
    def test_vllm_not_installed(self, mock_exit: MagicMock, mock_logger: MagicMock, mock_version: MagicMock) -> None:
        """Test behavior when vLLM is not installed."""
        # Mock vLLM not being installed
        mock_version.side_effect = importlib.metadata.PackageNotFoundError()

        result = check_vllm_compatibility(action="error")

        assert result is False
        mock_logger.error.assert_called_once()
        assert "vLLM is not installed" in mock_logger.error.call_args[0][0]
        mock_exit.assert_called_once_with(1)

    @patch("vllm_omni.utils.version_check.importlib.metadata.version")
    @patch("vllm_omni.utils.version_check.logger")
    def test_vllm_not_installed_warn_action(self, mock_logger: MagicMock, mock_version: MagicMock) -> None:
        """Test behavior when vLLM is not installed with warn action."""
        # Mock vLLM not being installed
        mock_version.side_effect = importlib.metadata.PackageNotFoundError()

        result = check_vllm_compatibility(action="warn")

        assert result is False
        mock_logger.error.assert_called_once()
        assert "vLLM is not installed" in mock_logger.error.call_args[0][0]

    @patch("vllm_omni.utils.version_check.importlib.metadata.version")
    @patch("vllm_omni.version.__version__", "0.14.0")
    @patch("vllm_omni.utils.version_check.logger")
    def test_version_parse_failure(self, mock_logger: MagicMock, mock_version: MagicMock) -> None:
        """Test that version parse failures don't block startup."""
        # Mock vLLM version with invalid format
        mock_version.return_value = "invalid-version"

        result = check_vllm_compatibility(action="error")

        # Should return True (don't block on parse failure)
        assert result is True
        mock_logger.warning.assert_called()
        assert "Failed to parse version" in mock_logger.warning.call_args[0][0]

    @patch.dict("os.environ", {"VLLM_OMNI_VERSION_CHECK": "warn"})
    @patch("vllm_omni.utils.version_check.importlib.metadata.version")
    @patch("vllm_omni.version.__version__", "0.15.0")
    @patch("vllm_omni.utils.version_check.logger")
    def test_environment_variable_override(self, mock_logger: MagicMock, mock_version: MagicMock) -> None:
        """Test that environment variable overrides action parameter."""
        # Mock incompatible versions
        mock_version.return_value = "0.14.0"

        # Pass action="error" but env var should override to "warn"
        result = check_vllm_compatibility(action="error")

        assert result is False
        # Should warn instead of error due to env var
        mock_logger.warning.assert_called_once()
        mock_logger.error.assert_not_called()

    @patch("vllm_omni.utils.version_check.importlib.metadata.version")
    @patch("vllm_omni.version.__version__", "0.14.5.dev10+gabcdef0.cuda")
    @patch("vllm_omni.utils.version_check.logger")
    def test_dev_versions_compatibility(self, mock_logger: MagicMock, mock_version: MagicMock) -> None:
        """Test compatibility check with development versions."""
        # Mock vLLM dev version
        mock_version.return_value = "0.14.1.dev23+g1a2b3c4"

        result = check_vllm_compatibility(action="error")

        assert result is True
        mock_logger.info.assert_called_once()

    @patch("vllm_omni.utils.version_check.importlib.metadata.version")
    @patch("vllm_omni.utils.version_check.logger")
    def test_unexpected_exception(self, mock_logger: MagicMock, mock_version: MagicMock) -> None:
        """Test that unexpected exceptions don't block startup."""
        # Mock an unexpected exception
        mock_version.side_effect = RuntimeError("Unexpected error")

        result = check_vllm_compatibility(action="error")

        # Should return True (don't block on unexpected errors)
        assert result is True
        mock_logger.warning.assert_called()
        assert "Failed to check vLLM compatibility" in mock_logger.warning.call_args[0][0]
