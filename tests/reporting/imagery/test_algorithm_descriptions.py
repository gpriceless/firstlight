"""
Tests for Algorithm Descriptions Module

Validates that all algorithms in the codebase have descriptions
and that description functions work correctly.
"""

import pytest
from core.reporting.imagery.algorithm_descriptions import (
    ALGORITHM_DESCRIPTIONS,
    SENSOR_DESCRIPTIONS,
    get_algorithm_description,
    get_sensor_description,
    get_detection_method_text,
    format_algorithm_summary,
)


class TestAlgorithmDescriptionsCoverage:
    """Verify coverage of all algorithms in the codebase."""

    def test_all_flood_algorithms_covered(self):
        """All flood algorithms have descriptions."""
        required_algorithms = [
            "flood.baseline.threshold_sar",
            "flood.baseline.ndwi",
            "flood.baseline.change_detection",
            "flood.baseline.hand_model",
            "flood.advanced.ensemble_fusion",
            "flood.advanced.unet_segmentation",
        ]

        for algo_id in required_algorithms:
            assert algo_id in ALGORITHM_DESCRIPTIONS, f"Missing description for {algo_id}"

    def test_all_wildfire_algorithms_covered(self):
        """All wildfire algorithms have descriptions."""
        required_algorithms = [
            "wildfire.baseline.thermal_anomaly",
            "wildfire.baseline.nbr_differenced",
            "wildfire.baseline.ba_classifier",
        ]

        for algo_id in required_algorithms:
            assert algo_id in ALGORITHM_DESCRIPTIONS, f"Missing description for {algo_id}"

    def test_all_storm_algorithms_covered(self):
        """All storm algorithms have descriptions."""
        required_algorithms = [
            "storm.baseline.wind_damage",
            "storm.baseline.structural_damage",
        ]

        for algo_id in required_algorithms:
            assert algo_id in ALGORITHM_DESCRIPTIONS, f"Missing description for {algo_id}"


class TestGetAlgorithmDescription:
    """Test get_algorithm_description function."""

    def test_exact_match(self):
        """Returns description for exact algorithm ID match."""
        desc = get_algorithm_description("flood.baseline.threshold_sar")
        assert "radar" in desc.lower()
        assert "cloud" in desc.lower()

    def test_wildfire_thermal(self):
        """Returns appropriate description for thermal fire detection."""
        desc = get_algorithm_description("wildfire.baseline.thermal_anomaly")
        assert "fire" in desc.lower()
        assert "thermal" in desc.lower()

    def test_burn_severity(self):
        """Returns appropriate description for burn severity."""
        desc = get_algorithm_description("wildfire.baseline.nbr_differenced")
        assert "burn severity" in desc.lower() or "severity" in desc.lower()
        assert "vegetation" in desc.lower() or "health" in desc.lower()

    def test_unknown_algorithm_returns_fallback(self):
        """Returns generic fallback for unknown algorithms."""
        desc = get_algorithm_description("unknown.algorithm.id")
        assert "analysis" in desc.lower()
        assert len(desc) > 0

    def test_partial_match(self):
        """Handles partial matches gracefully."""
        desc = get_algorithm_description("threshold_sar")
        assert "radar" in desc.lower()


class TestDescriptionQuality:
    """Verify description quality and language guidelines."""

    def test_no_technical_jargon(self):
        """Descriptions avoid unexplained technical terms."""
        jargon_terms = ["backscatter", "radiometric", "orthorectified", "geotiff"]

        for algo_id, description in ALGORITHM_DESCRIPTIONS.items():
            desc_lower = description.lower()
            for term in jargon_terms:
                assert term not in desc_lower, (
                    f"Algorithm {algo_id} description contains jargon: '{term}'"
                )

    def test_descriptions_are_concise(self):
        """Descriptions are one sentence (no periods except at end)."""
        for algo_id, description in ALGORITHM_DESCRIPTIONS.items():
            # Count periods
            period_count = description.count(".")

            # Should have 0 or 1 period (at the end)
            assert period_count <= 1, (
                f"Algorithm {algo_id} description has multiple sentences: '{description}'"
            )

    def test_descriptions_focus_on_what_not_how(self):
        """Descriptions focus on WHAT is detected, not implementation details."""
        implementation_terms = ["function", "module", "algorithm", "class", "method"]

        for algo_id, description in ALGORITHM_DESCRIPTIONS.items():
            desc_lower = description.lower()

            # "algorithm" is okay in the fallback, but shouldn't be in specific descriptions
            if algo_id != "fallback":
                for term in implementation_terms:
                    if term == "algorithm":
                        continue  # "algorithm" is okay in context
                    assert term not in desc_lower, (
                        f"Algorithm {algo_id} description mentions implementation: '{term}'"
                    )

    def test_no_acronyms_without_explanation(self):
        """Acronyms are either avoided or explained."""
        # SAR is explained as "radar"
        for algo_id, description in ALGORITHM_DESCRIPTIONS.items():
            # Check for unexplained uppercase acronyms (3+ letters)
            import re
            acronyms = re.findall(r'\b[A-Z]{3,}\b', description)

            for acronym in acronyms:
                # These are acceptable as they're common or explained
                allowed = ["AI", "SAR"]
                assert acronym in allowed, (
                    f"Algorithm {algo_id} contains unexplained acronym: '{acronym}'"
                )


class TestGetSensorDescription:
    """Test get_sensor_description function."""

    def test_sentinel1_description(self):
        """Returns correct description for Sentinel-1."""
        desc = get_sensor_description("sentinel-1")
        assert "sentinel-1" in desc.lower()
        assert "radar" in desc.lower()

    def test_sentinel2_description(self):
        """Returns correct description for Sentinel-2."""
        desc = get_sensor_description("sentinel-2")
        assert "sentinel-2" in desc.lower()
        assert "optical" in desc.lower()

    def test_generic_sar(self):
        """Returns correct description for generic SAR."""
        desc = get_sensor_description("sar")
        assert "radar" in desc.lower()

    def test_case_insensitive(self):
        """Handles case-insensitive sensor names."""
        desc1 = get_sensor_description("SENTINEL-1")
        desc2 = get_sensor_description("sentinel-1")
        assert desc1 == desc2

    def test_unknown_sensor_returns_as_is(self):
        """Unknown sensors are returned as-is."""
        desc = get_sensor_description("unknown-sensor")
        assert desc == "unknown-sensor"


class TestGetDetectionMethodText:
    """Test get_detection_method_text function."""

    def test_flood_sar_with_sensor(self):
        """Generates correct text for flood SAR detection."""
        text = get_detection_method_text(
            "flood.baseline.threshold_sar",
            "flood",
            "sentinel-1"
        )

        assert "flood" in text.lower()
        assert "radar" in text.lower()
        assert "sentinel-1" in text.lower()
        assert text.endswith(".")

    def test_wildfire_nbr_with_sensor(self):
        """Generates correct text for burn severity mapping."""
        text = get_detection_method_text(
            "wildfire.baseline.nbr_differenced",
            "wildfire",
            "sentinel-2"
        )

        assert "burn severity" in text.lower() or "severity" in text.lower()
        assert "sentinel-2" in text.lower()
        assert text.endswith(".")

    def test_without_sensor(self):
        """Generates text without sensor information."""
        text = get_detection_method_text(
            "flood.baseline.ndwi",
            "flood",
            None
        )

        assert "flood" in text.lower() or "water" in text.lower()
        assert text.endswith(".")
        assert len(text.split(".")) <= 2  # Should be 1 sentence

    def test_text_is_capitalized(self):
        """Generated text starts with capital letter."""
        text = get_detection_method_text(
            "flood.baseline.threshold_sar",
            "flood"
        )

        assert text[0].isupper()

    def test_thermal_fire_detection(self):
        """Generates appropriate text for thermal fire detection."""
        text = get_detection_method_text(
            "wildfire.baseline.thermal_anomaly",
            "wildfire",
            "modis"
        )

        assert "fire" in text.lower() or "thermal" in text.lower()
        assert "modis" in text.lower()


class TestFormatAlgorithmSummary:
    """Test format_algorithm_summary function."""

    def test_summary_with_area(self):
        """Generates complete summary with area statistics."""
        summary = format_algorithm_summary(
            "flood.baseline.threshold_sar",
            "flood",
            "sentinel-1",
            250.5
        )

        assert "flood" in summary.lower()
        assert "sentinel-1" in summary.lower()
        assert "250.5" in summary
        assert "hectares" in summary.lower()

    def test_summary_without_area(self):
        """Generates summary without area statistics."""
        summary = format_algorithm_summary(
            "flood.baseline.threshold_sar",
            "flood",
            "sentinel-1",
            None
        )

        assert "flood" in summary.lower()
        assert "sentinel-1" in summary.lower()
        assert "hectares" not in summary.lower()

    def test_area_formatting(self):
        """Area is formatted to one decimal place."""
        summary = format_algorithm_summary(
            "flood.baseline.ndwi",
            "flood",
            area_detected_ha=123.456
        )

        assert "123.5" in summary

    def test_complete_sentence_structure(self):
        """Summary has proper sentence structure."""
        summary = format_algorithm_summary(
            "wildfire.baseline.thermal_anomaly",
            "wildfire",
            "viirs",
            50.0
        )

        # Should have multiple sentences
        sentences = [s.strip() for s in summary.split(".") if s.strip()]
        assert len(sentences) >= 2

        # First sentence should be capitalized
        assert sentences[0][0].isupper()

        # Should end with period
        assert summary.rstrip().endswith(".")


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_algorithm_id(self):
        """Handles empty algorithm ID gracefully."""
        desc = get_algorithm_description("")
        assert len(desc) > 0  # Should return fallback

    def test_none_sensor(self):
        """Handles None sensor gracefully."""
        text = get_detection_method_text(
            "flood.baseline.threshold_sar",
            "flood",
            None
        )
        assert len(text) > 0

    def test_zero_area(self):
        """Handles zero detected area."""
        summary = format_algorithm_summary(
            "flood.baseline.ndwi",
            "flood",
            area_detected_ha=0.0
        )

        assert "0.0" in summary
        assert "hectares" in summary.lower()

    def test_large_area(self):
        """Handles large area values."""
        summary = format_algorithm_summary(
            "flood.baseline.threshold_sar",
            "flood",
            area_detected_ha=123456.789
        )

        assert "123456.8" in summary

    def test_event_type_with_subtype(self):
        """Handles event types with subtypes (e.g., flood.coastal)."""
        text = get_detection_method_text(
            "flood.baseline.threshold_sar",
            "flood.coastal",
            "sentinel-1"
        )

        assert "flood" in text.lower()
        assert len(text) > 0


class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_flood_report_scenario(self):
        """Complete flood detection report scenario."""
        summary = format_algorithm_summary(
            algorithm_id="flood.baseline.threshold_sar",
            event_type="flood",
            sensor="sentinel-1",
            area_detected_ha=342.7
        )

        # Should have all key information
        assert "flood" in summary.lower()
        assert "radar" in summary.lower()
        assert "cloud" in summary.lower()
        assert "sentinel-1" in summary.lower()
        assert "342.7" in summary
        assert "hectares" in summary.lower()

    def test_wildfire_report_scenario(self):
        """Complete wildfire detection report scenario."""
        summary = format_algorithm_summary(
            algorithm_id="wildfire.baseline.nbr_differenced",
            event_type="wildfire",
            sensor="landsat-8",
            area_detected_ha=1250.3
        )

        # Should have all key information
        assert "severity" in summary.lower() or "burn" in summary.lower()
        assert "landsat-8" in summary.lower()
        assert "1250.3" in summary

    def test_minimal_information_scenario(self):
        """Scenario with minimal information provided."""
        text = get_detection_method_text(
            "flood.advanced.ensemble_fusion",
            "flood"
        )

        # Should still generate meaningful text
        assert "flood" in text.lower()
        assert len(text) > 10
        assert text[0].isupper()
        assert text.endswith(".")
