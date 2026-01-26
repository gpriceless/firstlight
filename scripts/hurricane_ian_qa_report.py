#!/usr/bin/env python3
"""
Generate QA Reports for Hurricane Ian Analysis.

Loads existing analysis data and generates:
- Full QA Report with recommendations (HTML)
- Executive Summary Report (HTML/PDF-ready)
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.quality.reporting.qa_report import (
    QAReportGenerator,
    ReportConfig,
    ReportFormat,
    ReportLevel,
    ReportSection,
    QAReport,
    ReportMetadata,
    QualitySummary,
    CheckReport,
    CrossValidationReport,
    UncertaintySummaryReport,
    GatingReport,
    FlagReport,
    ExpertReviewReport,
    Recommendation,
)

# Configuration
DATA_DIR = Path.home() / 'hurricane_ian_real' / 'products'
OUTPUT_DIR = Path.home() / 'hurricane_ian_real' / 'reports'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EVENT_ID = "hurricane_ian_2022"
PRODUCT_ID = "flood_extent_fort_myers_fl"


def load_data():
    """Load Hurricane Ian analysis data."""
    print("Loading Hurricane Ian data...")

    data = {}

    # Load numpy arrays
    for name in ['flood_extent', 'confidence', 'ndwi', 'pre_flood_sar', 'post_flood_sar']:
        path = DATA_DIR / f'{name}.npy'
        if path.exists():
            data[name] = np.load(path)
            print(f"  Loaded {name}: shape={data[name].shape}, dtype={data[name].dtype}")

    # Load metadata
    metadata_path = DATA_DIR / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path) as f:
            data['metadata'] = json.load(f)
        print(f"  Loaded metadata: {list(data['metadata'].keys())}")

    # Load GeoJSON
    geojson_path = DATA_DIR / 'flood_analysis.geojson'
    if geojson_path.exists():
        with open(geojson_path) as f:
            data['geojson'] = json.load(f)
        print(f"  Loaded GeoJSON")

    return data


def analyze_data_quality(data: dict) -> dict:
    """Perform quality analysis on the data."""
    print("\nAnalyzing data quality...")

    results = {
        'checks': [],
        'metrics': {},
        'issues': [],
        'recommendations': [],
    }

    flood_extent = data.get('flood_extent')
    confidence = data.get('confidence')
    ndwi = data.get('ndwi')
    pre_sar = data.get('pre_flood_sar')
    post_sar = data.get('post_flood_sar')
    metadata = data.get('metadata', {})

    # === Spatial Checks ===
    print("  Running spatial checks...")

    # Check 1: Coverage completeness
    if flood_extent is not None:
        valid_pixels = np.sum(~np.isnan(flood_extent.astype(float)))
        total_pixels = flood_extent.size
        coverage = valid_pixels / total_pixels
        results['metrics']['coverage'] = coverage

        status = "pass" if coverage > 0.95 else "warning" if coverage > 0.8 else "soft_fail"
        results['checks'].append({
            'name': 'spatial_coverage',
            'category': 'spatial',
            'status': status,
            'metric': coverage,
            'threshold': 0.95,
            'details': f'Data coverage: {coverage:.1%} of AOI',
        })

        # Check 2: Spatial coherence (check for isolated pixels)
        flood_pixels = np.sum(flood_extent)
        results['metrics']['flood_pixels'] = int(flood_pixels)

        # Simple connectivity check
        from scipy import ndimage
        labeled, num_features = ndimage.label(flood_extent)
        results['metrics']['flood_regions'] = num_features

        # Check for fragmentation
        if num_features > 100:
            status = "warning"
            details = f"High fragmentation: {num_features} disconnected flood regions"
            results['issues'].append(details)
        else:
            status = "pass"
            details = f"Spatial coherence OK: {num_features} flood regions"

        results['checks'].append({
            'name': 'spatial_coherence',
            'category': 'spatial',
            'status': status,
            'metric': num_features,
            'threshold': 100,
            'details': details,
        })

    # === Value Range Checks ===
    print("  Running value range checks...")

    # Check 3: Confidence values in valid range
    if confidence is not None:
        conf_min = float(np.min(confidence))
        conf_max = float(np.max(confidence))
        conf_mean = float(np.mean(confidence))
        results['metrics']['confidence_mean'] = conf_mean
        results['metrics']['confidence_range'] = (conf_min, conf_max)

        in_range = (conf_min >= 0) and (conf_max <= 1)
        status = "pass" if in_range and conf_mean > 0.7 else "warning" if in_range else "soft_fail"

        results['checks'].append({
            'name': 'confidence_range',
            'category': 'value',
            'status': status,
            'metric': conf_mean,
            'threshold': 0.7,
            'details': f'Confidence range [{conf_min:.2f}, {conf_max:.2f}], mean={conf_mean:.2f}',
        })

    # Check 4: NDWI values
    if ndwi is not None:
        ndwi_min = float(np.min(ndwi))
        ndwi_max = float(np.max(ndwi))
        ndwi_mean = float(np.mean(ndwi))
        results['metrics']['ndwi_mean'] = ndwi_mean

        # NDWI should be in [-1, 1]
        in_range = (ndwi_min >= -1) and (ndwi_max <= 1)
        status = "pass" if in_range else "soft_fail"

        results['checks'].append({
            'name': 'ndwi_range',
            'category': 'value',
            'status': status,
            'metric': ndwi_mean,
            'details': f'NDWI range [{ndwi_min:.2f}, {ndwi_max:.2f}], mean={ndwi_mean:.2f}',
        })

    # Check 5: SAR backscatter range
    if post_sar is not None:
        sar_min = float(np.min(post_sar))
        sar_max = float(np.max(post_sar))
        sar_mean = float(np.mean(post_sar))
        results['metrics']['sar_mean_db'] = sar_mean

        # SAR backscatter typically -30 to 0 dB
        in_range = (sar_min >= -35) and (sar_max <= 5)
        status = "pass" if in_range else "warning"

        results['checks'].append({
            'name': 'sar_backscatter_range',
            'category': 'value',
            'status': status,
            'metric': sar_mean,
            'details': f'SAR backscatter range [{sar_min:.1f}, {sar_max:.1f}] dB, mean={sar_mean:.1f} dB',
        })

    # === Cross-Validation Checks ===
    print("  Running cross-validation checks...")

    if flood_extent is not None and ndwi is not None:
        # Compare SAR-based flood with NDWI-based water
        sar_flood = flood_extent.astype(bool)
        ndwi_water = ndwi > 0.1  # Water threshold

        # Calculate agreement
        intersection = np.sum(sar_flood & ndwi_water)
        union = np.sum(sar_flood | ndwi_water)
        iou = intersection / union if union > 0 else 0
        results['metrics']['sar_ndwi_iou'] = iou

        # Cohen's Kappa
        n = sar_flood.size
        po = (np.sum(sar_flood == ndwi_water)) / n  # Observed agreement
        pe = ((np.sum(sar_flood) * np.sum(ndwi_water)) +
              (np.sum(~sar_flood) * np.sum(~ndwi_water))) / (n * n)  # Expected agreement
        kappa = (po - pe) / (1 - pe) if (1 - pe) != 0 else 0
        results['metrics']['sar_ndwi_kappa'] = kappa

        status = "pass" if iou > 0.6 else "warning" if iou > 0.4 else "soft_fail"

        results['checks'].append({
            'name': 'sar_ndwi_agreement',
            'category': 'cross_validation',
            'status': status,
            'metric': iou,
            'threshold': 0.6,
            'details': f'SAR vs NDWI agreement: IoU={iou:.2%}, Kappa={kappa:.3f}',
        })

        # Disagreement analysis
        sar_only = np.sum(sar_flood & ~ndwi_water)
        ndwi_only = np.sum(~sar_flood & ndwi_water)
        results['metrics']['sar_only_pixels'] = int(sar_only)
        results['metrics']['ndwi_only_pixels'] = int(ndwi_only)

        if sar_only > ndwi_only * 2:
            results['issues'].append(f"SAR detects {sar_only:,} pixels not confirmed by NDWI - possible false positives from wet soil")
        if ndwi_only > sar_only * 2:
            results['issues'].append(f"NDWI detects {ndwi_only:,} water pixels missed by SAR - possible under-detection")

    # === Temporal Consistency ===
    print("  Running temporal checks...")

    if pre_sar is not None and post_sar is not None:
        # Check SAR change magnitude
        change = post_sar - pre_sar
        change_mean = float(np.mean(change))
        change_std = float(np.std(change))
        results['metrics']['sar_change_mean'] = change_mean
        results['metrics']['sar_change_std'] = change_std

        # Flood should show negative change (lower backscatter)
        flood_change = change[flood_extent.astype(bool)]
        flood_change_mean = float(np.mean(flood_change)) if len(flood_change) > 0 else 0
        results['metrics']['flood_area_change'] = flood_change_mean

        # Strong negative change in flood areas is expected
        status = "pass" if flood_change_mean < -5 else "warning" if flood_change_mean < -2 else "soft_fail"

        results['checks'].append({
            'name': 'temporal_sar_change',
            'category': 'temporal',
            'status': status,
            'metric': flood_change_mean,
            'threshold': -5.0,
            'details': f'SAR change in flood areas: {flood_change_mean:.1f} dB (expected < -5 dB)',
        })

    # === Uncertainty Analysis ===
    print("  Running uncertainty analysis...")

    if confidence is not None:
        # High uncertainty areas
        high_unc_mask = confidence < 0.5
        high_unc_pct = np.sum(high_unc_mask) / confidence.size
        results['metrics']['high_uncertainty_pct'] = high_unc_pct

        # Uncertainty in flood areas
        if flood_extent is not None:
            flood_conf = confidence[flood_extent.astype(bool)]
            flood_conf_mean = float(np.mean(flood_conf)) if len(flood_conf) > 0 else 0
            results['metrics']['flood_confidence_mean'] = flood_conf_mean

            status = "pass" if flood_conf_mean > 0.8 else "warning" if flood_conf_mean > 0.6 else "soft_fail"

            results['checks'].append({
                'name': 'flood_confidence',
                'category': 'uncertainty',
                'status': status,
                'metric': flood_conf_mean,
                'threshold': 0.8,
                'details': f'Mean confidence in flood areas: {flood_conf_mean:.1%}',
            })

    # === Generate Recommendations ===
    print("  Generating recommendations...")

    # Based on issues found
    for check in results['checks']:
        if check['status'] == 'soft_fail':
            if 'coverage' in check['name']:
                results['recommendations'].append({
                    'category': 'Data Quality',
                    'priority': 'high',
                    'recommendation': 'Consider acquiring additional imagery to fill coverage gaps',
                    'rationale': check['details'],
                    'impact': 'Improved spatial completeness of flood extent',
                })
            elif 'agreement' in check['name']:
                results['recommendations'].append({
                    'category': 'Methodology',
                    'priority': 'high',
                    'recommendation': 'Review algorithm parameters and consider ensemble approach',
                    'rationale': 'Low agreement between SAR and optical methods',
                    'impact': 'More robust flood detection with reduced false positives',
                })
            elif 'confidence' in check['name']:
                results['recommendations'].append({
                    'category': 'Uncertainty',
                    'priority': 'high',
                    'recommendation': 'Flag low-confidence regions for manual review',
                    'rationale': check['details'],
                    'impact': 'Improved product reliability for decision-making',
                })
        elif check['status'] == 'warning':
            if 'coherence' in check['name']:
                results['recommendations'].append({
                    'category': 'Post-Processing',
                    'priority': 'medium',
                    'recommendation': 'Apply morphological filtering to reduce spatial fragmentation',
                    'rationale': check['details'],
                    'impact': 'Cleaner flood extent boundaries',
                })

    # Standard recommendations for hurricane flood analysis
    results['recommendations'].append({
        'category': 'Validation',
        'priority': 'medium',
        'recommendation': 'Cross-reference with FEMA flood observations and high-water marks',
        'rationale': 'Independent validation improves product credibility',
        'impact': 'Quantified accuracy metrics for stakeholders',
    })

    results['recommendations'].append({
        'category': 'Documentation',
        'priority': 'low',
        'recommendation': 'Document temporal gap between SAR acquisitions and storm peak',
        'rationale': 'Timing affects flood extent accuracy',
        'impact': 'Better interpretation of results by end users',
    })

    return results


def build_qa_report(data: dict, analysis: dict) -> QAReport:
    """Build QAReport object from analysis results."""
    print("\nBuilding QA report...")

    metadata = data.get('metadata', {})

    # Create report metadata
    report_metadata = ReportMetadata(
        event_id=EVENT_ID,
        product_id=PRODUCT_ID,
        format=ReportFormat.HTML,
        level=ReportLevel.DETAILED,
        sections=list(ReportSection),
    )

    # Build checks list
    checks = []
    passed = 0
    warnings = 0
    failed = 0

    for check in analysis['checks']:
        status = check['status']
        if status == 'pass':
            passed += 1
        elif status == 'warning':
            warnings += 1
        else:
            failed += 1

        checks.append(CheckReport(
            check_name=check['name'],
            category=check['category'],
            status=status,
            metric_value=check.get('metric'),
            threshold=check.get('threshold'),
            details=check['details'],
        ))

    # Determine overall status
    if failed > 2:
        overall_status = "BLOCKED"
    elif failed > 0:
        overall_status = "REVIEW_REQUIRED"
    elif warnings > 0:
        overall_status = "PASS_WITH_WARNINGS"
    else:
        overall_status = "PASS"

    # Calculate confidence score
    confidence_score = max(0.0, 1.0 - (failed * 0.2) - (warnings * 0.05))

    # Key findings
    key_findings = []
    if metadata.get('flood_area_ha'):
        key_findings.append(f"Detected {metadata['flood_area_ha']:,.0f} hectares of flooding in Fort Myers area")
    if analysis['metrics'].get('sar_ndwi_iou'):
        iou = analysis['metrics']['sar_ndwi_iou']
        key_findings.append(f"SAR and optical methods show {iou:.0%} agreement (IoU)")
    if analysis['issues']:
        key_findings.extend(analysis['issues'][:2])

    # Create summary
    summary = QualitySummary(
        overall_status=overall_status,
        confidence_score=confidence_score,
        total_checks=len(checks),
        passed_checks=passed,
        warning_checks=warnings,
        failed_checks=failed,
        key_findings=key_findings,
    )

    # Cross-validation report
    cross_validation = None
    if 'sar_ndwi_iou' in analysis['metrics']:
        cross_validation = CrossValidationReport(
            methods_compared=['SAR_threshold', 'NDWI_optical'],
            agreement_score=analysis['metrics']['sar_ndwi_iou'],
            iou=analysis['metrics']['sar_ndwi_iou'],
            kappa=analysis['metrics'].get('sar_ndwi_kappa'),
            consensus_method='SAR_primary_NDWI_validation',
        )

    # Uncertainty report
    uncertainty = None
    if 'high_uncertainty_pct' in analysis['metrics']:
        uncertainty = UncertaintySummaryReport(
            mean_uncertainty=1.0 - analysis['metrics'].get('confidence_mean', 0.85),
            max_uncertainty=1.0 - analysis['metrics'].get('confidence_range', (0, 0.15))[0],
            high_uncertainty_percent=analysis['metrics']['high_uncertainty_pct'] * 100,
        )

    # Gating report
    gating = GatingReport(
        status=overall_status,
        rules_evaluated=len(checks),
        rules_passed=passed + warnings,
        warning_rules=[c['name'] for c in analysis['checks'] if c['status'] == 'warning'],
        blocking_rules=[c['name'] for c in analysis['checks'] if c['status'] == 'soft_fail'],
    )

    # Recommendations
    recommendations = [
        Recommendation(
            category=r['category'],
            priority=r['priority'],
            recommendation=r['recommendation'],
            rationale=r['rationale'],
            impact=r['impact'],
        )
        for r in analysis['recommendations']
    ]

    # Expert review (required if there are failed checks)
    expert_review = None
    if failed > 0:
        expert_review = ExpertReviewReport(
            required=True,
            reason=f"{failed} quality checks failed - manual review recommended",
            priority="normal" if failed == 1 else "high",
            status="pending",
        )

    return QAReport(
        metadata=report_metadata,
        summary=summary,
        checks=checks,
        cross_validation=cross_validation,
        uncertainty=uncertainty,
        gating=gating,
        expert_review=expert_review,
        recommendations=recommendations,
    )


def generate_executive_summary(data: dict, qa_report: QAReport) -> str:
    """Generate executive summary HTML report."""
    metadata = data.get('metadata', {})

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Executive Summary: Hurricane Ian Flood Analysis</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 0;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #1a365d 0%, #2c5282 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 2.5em;
        }}
        .header .subtitle {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        .container {{
            max-width: 900px;
            margin: -30px auto 40px auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 0;
            border-bottom: 1px solid #e2e8f0;
        }}
        .summary-card {{
            padding: 25px;
            text-align: center;
            border-right: 1px solid #e2e8f0;
        }}
        .summary-card:last-child {{
            border-right: none;
        }}
        .summary-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #2d3748;
        }}
        .summary-card .label {{
            color: #718096;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        .summary-card.status-pass .value {{ color: #38a169; }}
        .summary-card.status-warning .value {{ color: #d69e2e; }}
        .summary-card.status-fail .value {{ color: #e53e3e; }}
        .section {{
            padding: 30px 40px;
            border-bottom: 1px solid #e2e8f0;
        }}
        .section:last-child {{
            border-bottom: none;
        }}
        .section h2 {{
            color: #2d3748;
            margin: 0 0 20px 0;
            font-size: 1.4em;
            border-bottom: 2px solid #3182ce;
            padding-bottom: 10px;
            display: inline-block;
        }}
        .key-findings {{
            background: #ebf8ff;
            border-left: 4px solid #3182ce;
            padding: 15px 20px;
            margin: 15px 0;
        }}
        .key-findings li {{
            margin: 8px 0;
        }}
        .recommendation {{
            background: #f7fafc;
            border-radius: 6px;
            padding: 15px 20px;
            margin: 15px 0;
            border-left: 4px solid #48bb78;
        }}
        .recommendation.high {{
            border-left-color: #e53e3e;
        }}
        .recommendation.medium {{
            border-left-color: #d69e2e;
        }}
        .recommendation .priority {{
            font-size: 0.75em;
            text-transform: uppercase;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .recommendation.high .priority {{ color: #e53e3e; }}
        .recommendation.medium .priority {{ color: #d69e2e; }}
        .recommendation.low .priority {{ color: #48bb78; }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
        }}
        .metric-item {{
            background: #f7fafc;
            padding: 12px 15px;
            border-radius: 4px;
        }}
        .metric-item .label {{
            font-size: 0.85em;
            color: #718096;
        }}
        .metric-item .value {{
            font-size: 1.2em;
            font-weight: 600;
            color: #2d3748;
        }}
        .footer {{
            background: #f7fafc;
            padding: 20px 40px;
            font-size: 0.85em;
            color: #718096;
            text-align: center;
        }}
        .status-badge {{
            display: inline-block;
            padding: 8px 20px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 1.1em;
        }}
        .status-badge.pass {{ background: #c6f6d5; color: #22543d; }}
        .status-badge.warning {{ background: #fefcbf; color: #744210; }}
        .status-badge.fail {{ background: #fed7d7; color: #742a2a; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Hurricane Ian Flood Analysis</h1>
        <div class="subtitle">Fort Myers, FL &bull; September 28, 2022</div>
    </div>

    <div class="container">
        <div class="summary-cards">
            <div class="summary-card">
                <div class="value">{metadata.get('flood_area_ha', 0):,.0f}</div>
                <div class="label">Hectares Flooded</div>
            </div>
            <div class="summary-card">
                <div class="value">{metadata.get('coverage_pct', 0):.1f}%</div>
                <div class="label">Area Coverage</div>
            </div>
            <div class="summary-card">
                <div class="value">{qa_report.summary.confidence_score:.0%}</div>
                <div class="label">Confidence Score</div>
            </div>
            <div class="summary-card status-{'pass' if qa_report.summary.overall_status == 'PASS' else 'warning' if 'WARNING' in qa_report.summary.overall_status else 'fail'}">
                <div class="value">{qa_report.summary.passed_checks}/{qa_report.summary.total_checks}</div>
                <div class="label">QC Checks Passed</div>
            </div>
        </div>

        <div class="section">
            <h2>Quality Assessment</h2>
            <p style="text-align: center; margin-bottom: 20px;">
                <span class="status-badge {'pass' if qa_report.summary.overall_status == 'PASS' else 'warning' if 'WARNING' in qa_report.summary.overall_status else 'fail'}">
                    {qa_report.summary.overall_status.replace('_', ' ')}
                </span>
            </p>

            <div class="key-findings">
                <strong>Key Findings:</strong>
                <ul>
"""

    for finding in qa_report.summary.key_findings:
        html += f"                    <li>{finding}</li>\n"

    html += """                </ul>
            </div>
        </div>

        <div class="section">
            <h2>Analysis Metrics</h2>
            <div class="metrics-grid">
                <div class="metric-item">
                    <div class="label">Event</div>
                    <div class="value">Hurricane Ian (Cat 4)</div>
                </div>
                <div class="metric-item">
                    <div class="label">Landfall Date</div>
                    <div class="value">September 28, 2022</div>
                </div>
                <div class="metric-item">
                    <div class="label">Analysis Area</div>
                    <div class="value">Fort Myers, FL</div>
                </div>
                <div class="metric-item">
                    <div class="label">Bounding Box</div>
                    <div class="value">[-82.2, 26.3] to [-81.7, 26.8]</div>
                </div>
                <div class="metric-item">
                    <div class="label">SAR Method</div>
                    <div class="value">Threshold-based detection</div>
                </div>
                <div class="metric-item">
                    <div class="label">Optical Method</div>
                    <div class="value">NDWI water index</div>
                </div>
"""

    if qa_report.cross_validation:
        html += f"""                <div class="metric-item">
                    <div class="label">Method Agreement (IoU)</div>
                    <div class="value">{qa_report.cross_validation.iou:.1%}</div>
                </div>
"""

    if qa_report.uncertainty:
        html += f"""                <div class="metric-item">
                    <div class="label">High Uncertainty Areas</div>
                    <div class="value">{qa_report.uncertainty.high_uncertainty_percent:.1f}%</div>
                </div>
"""

    html += """            </div>
        </div>

        <div class="section">
            <h2>Recommendations</h2>
"""

    for rec in qa_report.recommendations:
        html += f"""            <div class="recommendation {rec.priority}">
                <div class="priority">{rec.priority} priority</div>
                <strong>{rec.recommendation}</strong>
                <p style="margin: 8px 0 0 0; color: #718096; font-size: 0.9em;">
                    {rec.rationale}
                </p>
            </div>
"""

    html += f"""        </div>

        <div class="footer">
            Generated by FirstLight Geospatial Intelligence Platform &bull; {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
        </div>
    </div>
</body>
</html>"""

    return html


def main():
    print("=" * 70)
    print("  HURRICANE IAN QA REPORT GENERATION")
    print("=" * 70)

    # Load data
    data = load_data()

    if not data:
        print("ERROR: No data found!")
        return

    # Analyze quality
    analysis = analyze_data_quality(data)

    # Build QA report
    qa_report = build_qa_report(data, analysis)

    # Save full QA report (HTML)
    print("\nSaving reports...")

    qa_html_path = OUTPUT_DIR / 'hurricane_ian_qa_report.html'
    qa_report.save(qa_html_path, ReportFormat.HTML)
    print(f"  Full QA Report (HTML): {qa_html_path}")

    # Save QA report (JSON)
    qa_json_path = OUTPUT_DIR / 'hurricane_ian_qa_report.json'
    qa_report.save(qa_json_path, ReportFormat.JSON)
    print(f"  Full QA Report (JSON): {qa_json_path}")

    # Save QA report (Markdown)
    qa_md_path = OUTPUT_DIR / 'hurricane_ian_qa_report.md'
    qa_report.save(qa_md_path, ReportFormat.MARKDOWN)
    print(f"  Full QA Report (Markdown): {qa_md_path}")

    # Generate and save executive summary
    exec_html = generate_executive_summary(data, qa_report)
    exec_path = OUTPUT_DIR / 'hurricane_ian_executive_summary.html'
    exec_path.write_text(exec_html)
    print(f"  Executive Summary (HTML): {exec_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"\n  Status: {qa_report.summary.overall_status}")
    print(f"  Confidence: {qa_report.summary.confidence_score:.0%}")
    print(f"  Checks: {qa_report.summary.passed_checks} passed, "
          f"{qa_report.summary.warning_checks} warnings, "
          f"{qa_report.summary.failed_checks} failed")

    print(f"\n  Key Findings:")
    for finding in qa_report.summary.key_findings:
        print(f"    - {finding}")

    print(f"\n  Top Recommendations:")
    for rec in qa_report.recommendations[:3]:
        print(f"    [{rec.priority.upper()}] {rec.recommendation}")

    print(f"\n  Reports saved to: {OUTPUT_DIR}")
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
