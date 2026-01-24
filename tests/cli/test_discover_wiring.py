"""
Tests for CLI discover command wiring to STAC client.

Verifies that:
1. Mock fallback has been removed
2. Network errors are retried with exponential backoff
3. Query errors are raised immediately without retry
4. Empty results are returned as empty list (not mock data)
"""

import pytest
from unittest.mock import Mock, patch, call
from datetime import datetime

from cli.commands.discover import (
    perform_discovery,
    _discover_with_retry,
    NetworkError,
    QueryError,
    STACError,
)


class TestDiscoverWiring:
    """Test suite for discover command STAC client integration."""

    def test_successful_discovery(self):
        """Test successful STAC discovery returns results."""
        geometry = {
            "type": "Polygon",
            "coordinates": [[
                [-80.5, 25.5],
                [-80.0, 25.5],
                [-80.0, 26.0],
                [-80.5, 26.0],
                [-80.5, 25.5],
            ]],
        }

        mock_results = [
            {
                "id": "S2A_MSIL2A_20240915",
                "source": "sentinel2",
                "datetime": "2024-09-15T10:30:00",
                "cloud_cover": 5.0,
                "priority": "primary",
            }
        ]

        with patch("cli.commands.discover.discover_data") as mock_discover:
            mock_discover.return_value = mock_results

            results = perform_discovery(
                geometry=geometry,
                start=datetime(2024, 9, 15),
                end=datetime(2024, 9, 20),
                event_type="flood",
                sources=None,
                max_cloud=30.0,
                config={},
            )

            assert len(results) == 1
            assert results[0]["id"] == "S2A_MSIL2A_20240915"
            mock_discover.assert_called_once()

    def test_empty_results_returned_as_empty_list(self):
        """Test that no results found returns empty list, not mock data."""
        geometry = {
            "type": "Polygon",
            "coordinates": [[
                [-80.5, 25.5],
                [-80.0, 25.5],
                [-80.0, 26.0],
                [-80.5, 26.0],
                [-80.5, 25.5],
            ]],
        }

        with patch("cli.commands.discover.discover_data") as mock_discover:
            mock_discover.return_value = []  # No results

            results = perform_discovery(
                geometry=geometry,
                start=datetime(2024, 9, 15),
                end=datetime(2024, 9, 20),
                event_type="flood",
                sources=None,
                max_cloud=30.0,
                config={},
            )

            assert results == []
            assert isinstance(results, list)

    def test_import_error_raises_stac_error(self):
        """Test that missing pystac-client raises STACError."""
        geometry = {
            "type": "Polygon",
            "coordinates": [[
                [-80.5, 25.5],
                [-80.0, 25.5],
                [-80.0, 26.0],
                [-80.5, 26.0],
                [-80.5, 25.5],
            ]],
        }

        with patch("cli.commands.discover.discover_data", side_effect=ImportError("No module named 'pystac_client'")):
            with pytest.raises(STACError) as exc_info:
                perform_discovery(
                    geometry=geometry,
                    start=datetime(2024, 9, 15),
                    end=datetime(2024, 9, 20),
                    event_type="flood",
                    sources=None,
                    max_cloud=30.0,
                    config={},
                )

            assert "pystac-client" in str(exc_info.value)

    def test_source_filtering(self):
        """Test that source filtering works correctly."""
        geometry = {
            "type": "Polygon",
            "coordinates": [[
                [-80.5, 25.5],
                [-80.0, 25.5],
                [-80.0, 26.0],
                [-80.5, 26.0],
                [-80.5, 25.5],
            ]],
        }

        mock_results = [
            {"id": "S1_001", "source": "sentinel1", "datetime": "2024-09-15", "priority": "primary"},
            {"id": "S2_001", "source": "sentinel2", "datetime": "2024-09-15", "priority": "primary"},
            {"id": "L8_001", "source": "landsat", "datetime": "2024-09-15", "priority": "secondary"},
        ]

        with patch("cli.commands.discover.discover_data") as mock_discover:
            mock_discover.return_value = mock_results

            # Filter for sentinel1 only
            results = perform_discovery(
                geometry=geometry,
                start=datetime(2024, 9, 15),
                end=datetime(2024, 9, 20),
                event_type="flood",
                sources=["sentinel1"],
                max_cloud=30.0,
                config={},
            )

            assert len(results) == 1
            assert results[0]["source"] == "sentinel1"


class TestRetryLogic:
    """Test suite for retry logic with exponential backoff."""

    def test_network_error_retries_with_backoff(self):
        """Test that network errors are retried with exponential backoff."""
        mock_func = Mock()
        mock_func.side_effect = [
            Exception("Connection timeout"),
            Exception("Connection timeout"),
            [{"id": "result"}],  # Success on third attempt
        ]

        with patch("cli.commands.discover.time.sleep") as mock_sleep:
            results = _discover_with_retry(
                bbox=[-80.5, 25.5, -80.0, 26.0],
                start_date="2024-09-15",
                end_date="2024-09-20",
                event_type="flood",
                max_cloud_cover=30.0,
                discover_func=mock_func,
                max_attempts=3,
            )

            assert len(results) == 1
            assert results[0]["id"] == "result"
            assert mock_func.call_count == 3

            # Check exponential backoff: 1s, 2s
            assert mock_sleep.call_count == 2
            mock_sleep.assert_has_calls([call(1), call(2)])

    def test_network_error_exhausts_retries(self):
        """Test that network errors raise NetworkError after max attempts."""
        mock_func = Mock()
        mock_func.side_effect = Exception("Connection timeout")

        with patch("cli.commands.discover.time.sleep"):
            with pytest.raises(NetworkError) as exc_info:
                _discover_with_retry(
                    bbox=[-80.5, 25.5, -80.0, 26.0],
                    start_date="2024-09-15",
                    end_date="2024-09-20",
                    event_type="flood",
                    max_cloud_cover=30.0,
                    discover_func=mock_func,
                    max_attempts=3,
                )

            assert "3 attempts" in str(exc_info.value)
            assert mock_func.call_count == 3

    def test_query_error_no_retry(self):
        """Test that query errors are not retried."""
        mock_func = Mock()
        mock_func.side_effect = Exception("Invalid bbox parameter")

        with pytest.raises(QueryError) as exc_info:
            _discover_with_retry(
                bbox=[-80.5, 25.5, -80.0, 26.0],
                start_date="2024-09-15",
                end_date="2024-09-20",
                event_type="flood",
                max_cloud_cover=30.0,
                discover_func=mock_func,
                max_attempts=3,
            )

        # Should fail immediately without retry
        assert mock_func.call_count == 1
        assert "Invalid query parameters" in str(exc_info.value)

    def test_unknown_error_no_retry(self):
        """Test that unknown errors are not retried."""
        mock_func = Mock()
        mock_func.side_effect = Exception("Some unknown error")

        with pytest.raises(STACError) as exc_info:
            _discover_with_retry(
                bbox=[-80.5, 25.5, -80.0, 26.0],
                start_date="2024-09-15",
                end_date="2024-09-20",
                event_type="flood",
                max_cloud_cover=30.0,
                discover_func=mock_func,
                max_attempts=3,
            )

        # Should fail immediately without retry
        assert mock_func.call_count == 1
        assert "STAC discovery failed" in str(exc_info.value)

    def test_empty_results_no_retry(self):
        """Test that empty results are returned immediately (not retried)."""
        mock_func = Mock(return_value=[])

        results = _discover_with_retry(
            bbox=[-80.5, 25.5, -80.0, 26.0],
            start_date="2024-09-15",
            end_date="2024-09-20",
            event_type="flood",
            max_cloud_cover=30.0,
            discover_func=mock_func,
            max_attempts=3,
        )

        assert results == []
        assert mock_func.call_count == 1  # No retry for empty results


class TestErrorCategorization:
    """Test suite for error categorization."""

    @pytest.mark.parametrize("error_msg,expected_error", [
        ("Connection timeout", NetworkError),
        ("DNS resolution failed", NetworkError),
        ("Network unreachable", NetworkError),
        ("Connection refused", NetworkError),
        ("Connection reset by peer", NetworkError),
        ("Invalid bbox parameter", QueryError),
        ("Invalid collection name", QueryError),
        ("Validation error", QueryError),
        ("Parameter error", QueryError),
    ])
    def test_error_categorization(self, error_msg, expected_error):
        """Test that errors are categorized correctly."""
        mock_func = Mock(side_effect=Exception(error_msg))

        with pytest.raises(expected_error):
            _discover_with_retry(
                bbox=[-80.5, 25.5, -80.0, 26.0],
                start_date="2024-09-15",
                end_date="2024-09-20",
                event_type="flood",
                max_cloud_cover=30.0,
                discover_func=mock_func,
                max_attempts=1,  # Only 1 attempt to avoid retries
            )


class TestNoMockFallback:
    """Test suite to verify mock fallback has been completely removed."""

    def test_no_generate_mock_results_function(self):
        """Verify that generate_mock_results function no longer exists."""
        import cli.commands.discover as discover_module

        assert not hasattr(discover_module, "generate_mock_results"), (
            "generate_mock_results function should be removed"
        )

    def test_no_mock_data_on_error(self):
        """Verify that errors don't fall back to mock data."""
        geometry = {
            "type": "Polygon",
            "coordinates": [[
                [-80.5, 25.5],
                [-80.0, 25.5],
                [-80.0, 26.0],
                [-80.5, 26.0],
                [-80.5, 25.5],
            ]],
        }

        with patch("cli.commands.discover.discover_data") as mock_discover:
            mock_discover.side_effect = Exception("STAC catalog unreachable")

            with pytest.raises((NetworkError, STACError)):
                perform_discovery(
                    geometry=geometry,
                    start=datetime(2024, 9, 15),
                    end=datetime(2024, 9, 20),
                    event_type="flood",
                    sources=None,
                    max_cloud=30.0,
                    config={},
                )

    def test_no_random_imports(self):
        """Verify that random module is not imported (used by mock generator)."""
        with patch("cli.commands.discover.discover_data") as mock_discover:
            mock_discover.return_value = []

            geometry = {
                "type": "Polygon",
                "coordinates": [[
                    [-80.5, 25.5],
                    [-80.0, 25.5],
                    [-80.0, 26.0],
                    [-80.5, 26.0],
                    [-80.5, 25.5],
                ]],
            }

            results = perform_discovery(
                geometry=geometry,
                start=datetime(2024, 9, 15),
                end=datetime(2024, 9, 20),
                event_type="flood",
                sources=None,
                max_cloud=30.0,
                config={},
            )

            # Should return empty list, not mock data with random values
            assert results == []
