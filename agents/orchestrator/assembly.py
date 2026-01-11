"""
Final Product Assembly for Orchestrator Agent.

Collects outputs from all agents and assembles final deliverables:
- Collect outputs from Discovery, Pipeline, Quality, and Reporting agents
- Merge provenance records from all stages
- Package final deliverables (GeoTIFF, GeoJSON, reports)
- Generate execution summary with metrics and quality scores

The assembly module ensures:
- All outputs are collected and organized
- Provenance chain is complete and verified
- Products are packaged in requested formats
- Execution summary captures the full workflow
"""

import hashlib
import json
import logging
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union


logger = logging.getLogger(__name__)


class ProductFormat(Enum):
    """Output product formats."""
    GEOTIFF = "geotiff"         # Cloud-optimized GeoTIFF
    GEOJSON = "geojson"         # Vector features
    GEOPARQUET = "geoparquet"   # Columnar vector format
    ZARR = "zarr"               # Chunked array format
    PDF = "pdf"                 # Report document
    HTML = "html"               # Web report
    JSON = "json"               # Structured data
    YAML = "yaml"               # Configuration/metadata


class ProductType(Enum):
    """Types of output products."""
    PRIMARY_RESULT = "primary_result"       # Main analysis output
    CONFIDENCE_MAP = "confidence_map"       # Confidence/uncertainty
    QUALITY_REPORT = "quality_report"       # QA/QC report
    PROVENANCE = "provenance"               # Lineage record
    VISUALIZATION = "visualization"         # Maps/charts
    SUMMARY = "summary"                     # Executive summary
    RAW_OUTPUT = "raw_output"              # Intermediate result
    METADATA = "metadata"                   # Product metadata


@dataclass
class ProductMetadata:
    """
    Metadata for an output product.

    Attributes:
        product_id: Unique product identifier
        event_id: Associated event ID
        product_type: Type of product
        format: Output format
        name: Product name
        description: Product description
        created_at: Creation timestamp
        file_path: Path to product file
        file_size_bytes: File size
        checksum: File checksum (SHA-256)
        spatial_extent: Bounding box [minx, miny, maxx, maxy]
        temporal_extent: Time range [start, end]
        crs: Coordinate reference system
        resolution_m: Spatial resolution
        confidence_score: Overall confidence (0-1)
        quality_flags: Quality indicators
        source_stages: Stages that contributed
        tags: Product tags
        custom: Custom metadata
    """
    product_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_id: str = ""
    product_type: ProductType = ProductType.PRIMARY_RESULT
    format: ProductFormat = ProductFormat.GEOTIFF
    name: str = ""
    description: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    file_path: Optional[str] = None
    file_size_bytes: int = 0
    checksum: Optional[str] = None
    spatial_extent: Optional[List[float]] = None
    temporal_extent: Optional[List[str]] = None
    crs: str = "EPSG:4326"
    resolution_m: Optional[float] = None
    confidence_score: Optional[float] = None
    quality_flags: List[str] = field(default_factory=list)
    source_stages: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    custom: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "product_id": self.product_id,
            "event_id": self.event_id,
            "product_type": self.product_type.value,
            "format": self.format.value,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "file_path": self.file_path,
            "file_size_bytes": self.file_size_bytes,
            "checksum": self.checksum,
            "spatial_extent": self.spatial_extent,
            "temporal_extent": self.temporal_extent,
            "crs": self.crs,
            "resolution_m": self.resolution_m,
            "confidence_score": self.confidence_score,
            "quality_flags": self.quality_flags,
            "source_stages": self.source_stages,
            "tags": self.tags,
            "custom": self.custom,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProductMetadata':
        """Create from dictionary."""
        return cls(
            product_id=data.get("product_id", str(uuid.uuid4())),
            event_id=data.get("event_id", ""),
            product_type=ProductType(data.get("product_type", "primary_result")),
            format=ProductFormat(data.get("format", "geotiff")),
            name=data.get("name", ""),
            description=data.get("description", ""),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(timezone.utc),
            file_path=data.get("file_path"),
            file_size_bytes=data.get("file_size_bytes", 0),
            checksum=data.get("checksum"),
            spatial_extent=data.get("spatial_extent"),
            temporal_extent=data.get("temporal_extent"),
            crs=data.get("crs", "EPSG:4326"),
            resolution_m=data.get("resolution_m"),
            confidence_score=data.get("confidence_score"),
            quality_flags=data.get("quality_flags", []),
            source_stages=data.get("source_stages", []),
            tags=data.get("tags", []),
            custom=data.get("custom", {}),
        )


@dataclass
class ProvenanceRecord:
    """
    Provenance record for tracking data lineage.

    Attributes:
        record_id: Unique record identifier
        event_id: Associated event ID
        stage: Processing stage
        timestamp: Record timestamp
        operation: Operation performed
        inputs: Input data references
        outputs: Output data references
        parameters: Operation parameters
        algorithm: Algorithm used
        algorithm_version: Algorithm version
        execution_time_seconds: Execution duration
        executor: Agent or process that executed
        environment: Execution environment info
        notes: Additional notes
    """
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_id: str = ""
    stage: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    operation: str = ""
    inputs: List[Dict[str, Any]] = field(default_factory=list)
    outputs: List[Dict[str, Any]] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    algorithm: Optional[str] = None
    algorithm_version: Optional[str] = None
    execution_time_seconds: Optional[float] = None
    executor: str = ""
    environment: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "record_id": self.record_id,
            "event_id": self.event_id,
            "stage": self.stage,
            "timestamp": self.timestamp.isoformat(),
            "operation": self.operation,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "parameters": self.parameters,
            "algorithm": self.algorithm,
            "algorithm_version": self.algorithm_version,
            "execution_time_seconds": self.execution_time_seconds,
            "executor": self.executor,
            "environment": self.environment,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProvenanceRecord':
        """Create from dictionary."""
        return cls(
            record_id=data.get("record_id", str(uuid.uuid4())),
            event_id=data.get("event_id", ""),
            stage=data.get("stage", ""),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(timezone.utc),
            operation=data.get("operation", ""),
            inputs=data.get("inputs", []),
            outputs=data.get("outputs", []),
            parameters=data.get("parameters", {}),
            algorithm=data.get("algorithm"),
            algorithm_version=data.get("algorithm_version"),
            execution_time_seconds=data.get("execution_time_seconds"),
            executor=data.get("executor", ""),
            environment=data.get("environment", {}),
            notes=data.get("notes", ""),
        )


@dataclass
class ExecutionSummary:
    """
    Summary of workflow execution.

    Attributes:
        event_id: Event identifier
        orchestrator_id: Orchestrator agent ID
        started_at: Execution start time
        completed_at: Execution completion time
        total_duration_seconds: Total duration
        stages_completed: Number of stages completed
        stages_failed: Number of stages failed
        data_sources_used: Data sources accessed
        algorithms_executed: Algorithms run
        products_generated: Products created
        quality_score: Overall quality score
        degraded_mode: Whether degraded mode was used
        degraded_reasons: Reasons for degradation
        errors_encountered: Number of errors
        warnings: Warning messages
        metrics: Detailed metrics by stage
    """
    event_id: str
    orchestrator_id: str = ""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_duration_seconds: float = 0.0
    stages_completed: int = 0
    stages_failed: int = 0
    data_sources_used: List[str] = field(default_factory=list)
    algorithms_executed: List[str] = field(default_factory=list)
    products_generated: List[str] = field(default_factory=list)
    quality_score: Optional[float] = None
    degraded_mode: bool = False
    degraded_reasons: List[str] = field(default_factory=list)
    errors_encountered: int = 0
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "orchestrator_id": self.orchestrator_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_duration_seconds": self.total_duration_seconds,
            "stages_completed": self.stages_completed,
            "stages_failed": self.stages_failed,
            "data_sources_used": self.data_sources_used,
            "algorithms_executed": self.algorithms_executed,
            "products_generated": self.products_generated,
            "quality_score": self.quality_score,
            "degraded_mode": self.degraded_mode,
            "degraded_reasons": self.degraded_reasons,
            "errors_encountered": self.errors_encountered,
            "warnings": self.warnings,
            "metrics": self.metrics,
        }


@dataclass
class AssemblyResult:
    """
    Result of product assembly.

    Attributes:
        event_id: Event identifier
        success: Whether assembly succeeded
        products: List of assembled products
        provenance: Merged provenance chain
        summary: Execution summary
        output_directory: Path to output directory
        manifest: Product manifest
        errors: Any errors during assembly
    """
    event_id: str
    success: bool = True
    products: List[ProductMetadata] = field(default_factory=list)
    provenance: List[ProvenanceRecord] = field(default_factory=list)
    summary: Optional[ExecutionSummary] = None
    output_directory: Optional[str] = None
    manifest: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "success": self.success,
            "products": [p.to_dict() for p in self.products],
            "provenance": [p.to_dict() for p in self.provenance],
            "summary": self.summary.to_dict() if self.summary else None,
            "output_directory": self.output_directory,
            "manifest": self.manifest,
            "errors": self.errors,
        }


class ProductAssembler:
    """
    Assembles final products from agent outputs.

    Collects outputs from all stages, merges provenance, packages
    deliverables, and generates the execution summary.

    Example:
        assembler = ProductAssembler(output_dir="/path/to/output")

        result = await assembler.assemble(
            event_id="event_001",
            event_spec=event_spec,
            discovery_results=discovery_output,
            pipeline_results=pipeline_output,
            quality_results=quality_output,
            requested_formats=[ProductFormat.GEOTIFF, ProductFormat.GEOJSON],
        )
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        include_provenance: bool = True,
        include_summary: bool = True,
        generate_manifest: bool = True,
    ):
        """
        Initialize product assembler.

        Args:
            output_dir: Base output directory
            include_provenance: Include provenance records
            include_summary: Include execution summary
            generate_manifest: Generate product manifest
        """
        self.output_dir = Path(output_dir) if output_dir else Path("./output")
        self.include_provenance = include_provenance
        self.include_summary = include_summary
        self.generate_manifest = generate_manifest

    async def assemble(
        self,
        event_id: str,
        event_spec: Dict[str, Any],
        discovery_results: Optional[Dict[str, Any]] = None,
        pipeline_results: Optional[Dict[str, Any]] = None,
        quality_results: Optional[Dict[str, Any]] = None,
        reporting_results: Optional[Dict[str, Any]] = None,
        execution_state: Optional[Dict[str, Any]] = None,
        requested_formats: Optional[List[ProductFormat]] = None,
    ) -> AssemblyResult:
        """
        Assemble final products from all agent outputs.

        Args:
            event_id: Event identifier
            event_spec: Original event specification
            discovery_results: Discovery agent output
            pipeline_results: Pipeline agent output
            quality_results: Quality agent output
            reporting_results: Reporting agent output
            execution_state: Orchestrator execution state
            requested_formats: Requested output formats

        Returns:
            AssemblyResult with all products
        """
        result = AssemblyResult(event_id=event_id)

        # Create output directory
        event_output_dir = self.output_dir / event_id
        event_output_dir.mkdir(parents=True, exist_ok=True)
        result.output_directory = str(event_output_dir)

        try:
            # 1. Collect outputs from all stages
            stage_outputs = self._collect_stage_outputs(
                discovery_results,
                pipeline_results,
                quality_results,
                reporting_results,
            )

            # 2. Merge provenance records
            if self.include_provenance:
                result.provenance = self._merge_provenance(
                    event_id,
                    stage_outputs,
                    execution_state,
                )

            # 3. Package primary products
            primary_products = await self._package_primary_products(
                event_id,
                event_output_dir,
                pipeline_results,
                requested_formats or [ProductFormat.GEOTIFF, ProductFormat.GEOJSON],
            )
            result.products.extend(primary_products)

            # 4. Package quality report
            if quality_results:
                quality_products = await self._package_quality_products(
                    event_id,
                    event_output_dir,
                    quality_results,
                )
                result.products.extend(quality_products)

            # 5. Package reports
            if reporting_results:
                report_products = await self._package_reports(
                    event_id,
                    event_output_dir,
                    reporting_results,
                )
                result.products.extend(report_products)

            # 6. Generate execution summary
            if self.include_summary:
                result.summary = self._generate_summary(
                    event_id,
                    event_spec,
                    execution_state,
                    stage_outputs,
                    result.products,
                )

            # 7. Generate manifest
            if self.generate_manifest:
                result.manifest = self._generate_manifest(
                    event_id,
                    event_spec,
                    result.products,
                    result.provenance,
                    result.summary,
                )
                # Write manifest file
                manifest_path = event_output_dir / "manifest.json"
                with open(manifest_path, 'w') as f:
                    json.dump(result.manifest, f, indent=2, default=str)

            # 8. Write provenance file
            if self.include_provenance and result.provenance:
                provenance_path = event_output_dir / "provenance.json"
                with open(provenance_path, 'w') as f:
                    json.dump(
                        [p.to_dict() for p in result.provenance],
                        f, indent=2, default=str
                    )

            result.success = True
            logger.info(f"Assembled {len(result.products)} products for event {event_id}")

        except Exception as e:
            logger.error(f"Assembly failed for event {event_id}: {e}")
            result.success = False
            result.errors.append(str(e))

        return result

    def _collect_stage_outputs(
        self,
        discovery_results: Optional[Dict[str, Any]],
        pipeline_results: Optional[Dict[str, Any]],
        quality_results: Optional[Dict[str, Any]],
        reporting_results: Optional[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Collect outputs from all stages."""
        return {
            "discovery": discovery_results or {},
            "pipeline": pipeline_results or {},
            "quality": quality_results or {},
            "reporting": reporting_results or {},
        }

    def _merge_provenance(
        self,
        event_id: str,
        stage_outputs: Dict[str, Dict[str, Any]],
        execution_state: Optional[Dict[str, Any]],
    ) -> List[ProvenanceRecord]:
        """Merge provenance records from all stages."""
        records = []

        for stage_name, output in stage_outputs.items():
            if not output:
                continue

            # Extract provenance from stage output
            stage_provenance = output.get("provenance", [])
            if isinstance(stage_provenance, list):
                for prov_data in stage_provenance:
                    if isinstance(prov_data, dict):
                        record = ProvenanceRecord.from_dict(prov_data)
                        record.event_id = event_id
                        record.stage = stage_name
                        records.append(record)
                    elif isinstance(prov_data, ProvenanceRecord):
                        prov_data.event_id = event_id
                        prov_data.stage = stage_name
                        records.append(prov_data)

        # Sort by timestamp
        records.sort(key=lambda r: r.timestamp)

        return records

    async def _package_primary_products(
        self,
        event_id: str,
        output_dir: Path,
        pipeline_results: Optional[Dict[str, Any]],
        formats: List[ProductFormat],
    ) -> List[ProductMetadata]:
        """Package primary analysis products."""
        products = []

        if not pipeline_results:
            return products

        # Get primary outputs from pipeline
        outputs = pipeline_results.get("outputs", [])

        for output in outputs:
            output_path = output.get("path")
            output_type = output.get("type", "primary_result")
            output_format = output.get("format", "geotiff")

            if not output_path:
                continue

            # Create product metadata
            product = ProductMetadata(
                event_id=event_id,
                product_type=ProductType(output_type) if output_type in [pt.value for pt in ProductType] else ProductType.PRIMARY_RESULT,
                format=ProductFormat(output_format) if output_format in [pf.value for pf in ProductFormat] else ProductFormat.GEOTIFF,
                name=output.get("name", f"{event_id}_{output_type}"),
                description=output.get("description", ""),
                source_stages=["pipeline"],
                confidence_score=output.get("confidence"),
            )

            # Copy or reference file
            source_path = Path(output_path)
            if source_path.exists():
                dest_path = output_dir / source_path.name
                shutil.copy2(source_path, dest_path)
                product.file_path = str(dest_path)
                product.file_size_bytes = dest_path.stat().st_size
                product.checksum = self._calculate_checksum(dest_path)
            else:
                product.file_path = output_path

            # Copy spatial/temporal metadata
            if "spatial_extent" in output:
                product.spatial_extent = output["spatial_extent"]
            if "temporal_extent" in output:
                product.temporal_extent = output["temporal_extent"]
            if "crs" in output:
                product.crs = output["crs"]
            if "resolution_m" in output:
                product.resolution_m = output["resolution_m"]

            products.append(product)

        return products

    async def _package_quality_products(
        self,
        event_id: str,
        output_dir: Path,
        quality_results: Dict[str, Any],
    ) -> List[ProductMetadata]:
        """Package quality assessment products."""
        products = []

        # Create quality report product
        report_data = quality_results.get("report", quality_results)

        report_path = output_dir / "quality_report.json"
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

        product = ProductMetadata(
            event_id=event_id,
            product_type=ProductType.QUALITY_REPORT,
            format=ProductFormat.JSON,
            name=f"{event_id}_quality_report",
            description="Quality assessment report",
            file_path=str(report_path),
            file_size_bytes=report_path.stat().st_size,
            checksum=self._calculate_checksum(report_path),
            source_stages=["quality"],
            confidence_score=quality_results.get("overall_score"),
            quality_flags=quality_results.get("flags", []),
        )
        products.append(product)

        # Package confidence map if available
        confidence_map_path = quality_results.get("confidence_map_path")
        if confidence_map_path and Path(confidence_map_path).exists():
            dest_path = output_dir / "confidence_map.tif"
            shutil.copy2(confidence_map_path, dest_path)

            confidence_product = ProductMetadata(
                event_id=event_id,
                product_type=ProductType.CONFIDENCE_MAP,
                format=ProductFormat.GEOTIFF,
                name=f"{event_id}_confidence_map",
                description="Confidence/uncertainty map",
                file_path=str(dest_path),
                file_size_bytes=dest_path.stat().st_size,
                checksum=self._calculate_checksum(dest_path),
                source_stages=["quality"],
            )
            products.append(confidence_product)

        return products

    async def _package_reports(
        self,
        event_id: str,
        output_dir: Path,
        reporting_results: Dict[str, Any],
    ) -> List[ProductMetadata]:
        """Package report products."""
        products = []

        # Process each report format
        for report_type, report_data in reporting_results.items():
            if not isinstance(report_data, dict):
                continue

            report_path = report_data.get("path")
            report_format = report_data.get("format", "html")

            if report_path and Path(report_path).exists():
                source_path = Path(report_path)
                dest_path = output_dir / source_path.name
                shutil.copy2(source_path, dest_path)

                product = ProductMetadata(
                    event_id=event_id,
                    product_type=ProductType.SUMMARY if "summary" in report_type.lower() else ProductType.VISUALIZATION,
                    format=ProductFormat(report_format) if report_format in [pf.value for pf in ProductFormat] else ProductFormat.HTML,
                    name=f"{event_id}_{report_type}",
                    description=report_data.get("description", f"{report_type} report"),
                    file_path=str(dest_path),
                    file_size_bytes=dest_path.stat().st_size,
                    checksum=self._calculate_checksum(dest_path),
                    source_stages=["reporting"],
                )
                products.append(product)

        return products

    def _generate_summary(
        self,
        event_id: str,
        event_spec: Dict[str, Any],
        execution_state: Optional[Dict[str, Any]],
        stage_outputs: Dict[str, Dict[str, Any]],
        products: List[ProductMetadata],
    ) -> ExecutionSummary:
        """Generate execution summary."""
        summary = ExecutionSummary(event_id=event_id)

        # Extract timing from execution state
        if execution_state:
            summary.orchestrator_id = execution_state.get("orchestrator_id", "")

            if execution_state.get("started_at"):
                summary.started_at = datetime.fromisoformat(execution_state["started_at"])
            if execution_state.get("completed_at"):
                summary.completed_at = datetime.fromisoformat(execution_state["completed_at"])

            # Calculate duration
            if summary.started_at:
                end_time = summary.completed_at or datetime.now(timezone.utc)
                summary.total_duration_seconds = (end_time - summary.started_at).total_seconds()

            # Count stages
            stages = execution_state.get("stages", {})
            for stage_name, stage_data in stages.items():
                if isinstance(stage_data, dict):
                    status = stage_data.get("status", "")
                    if status == "completed":
                        summary.stages_completed += 1
                    elif status == "failed":
                        summary.stages_failed += 1

            # Degraded mode info
            degraded_mode = execution_state.get("degraded_mode", {})
            if degraded_mode.get("level") and degraded_mode["level"] != "none":
                summary.degraded_mode = True
                summary.degraded_reasons = degraded_mode.get("reasons", [])

            # Errors
            summary.errors_encountered = len(execution_state.get("error_summary", []))

        # Extract data sources from discovery
        discovery = stage_outputs.get("discovery", {})
        if discovery:
            datasets = discovery.get("selected_datasets", [])
            for ds in datasets:
                if isinstance(ds, dict):
                    source = ds.get("provider") or ds.get("source")
                    if source and source not in summary.data_sources_used:
                        summary.data_sources_used.append(source)

        # Extract algorithms from pipeline
        pipeline = stage_outputs.get("pipeline", {})
        if pipeline:
            steps = pipeline.get("steps_executed", [])
            for step in steps:
                if isinstance(step, dict):
                    algo = step.get("algorithm") or step.get("processor")
                    if algo and algo not in summary.algorithms_executed:
                        summary.algorithms_executed.append(algo)

        # Products generated
        summary.products_generated = [p.name for p in products]

        # Quality score
        quality = stage_outputs.get("quality", {})
        if quality:
            summary.quality_score = quality.get("overall_score")

        # Stage-specific metrics
        for stage_name, output in stage_outputs.items():
            if output and "metrics" in output:
                summary.metrics[stage_name] = output["metrics"]

        return summary

    def _generate_manifest(
        self,
        event_id: str,
        event_spec: Dict[str, Any],
        products: List[ProductMetadata],
        provenance: List[ProvenanceRecord],
        summary: Optional[ExecutionSummary],
    ) -> Dict[str, Any]:
        """Generate product manifest."""
        return {
            "manifest_version": "1.0.0",
            "event_id": event_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "event_spec": event_spec,
            "products": [p.to_dict() for p in products],
            "product_count": len(products),
            "provenance_record_count": len(provenance),
            "summary": summary.to_dict() if summary else None,
            "checksums": {
                p.name: p.checksum
                for p in products if p.checksum
            },
        }

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of a file."""
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)

        return sha256_hash.hexdigest()


class ProductPackager:
    """
    Utility class for packaging individual products.

    Provides methods for converting between formats and adding metadata.
    """

    @staticmethod
    def calculate_file_checksum(file_path: Union[str, Path]) -> str:
        """Calculate SHA-256 checksum."""
        sha256_hash = hashlib.sha256()
        path = Path(file_path)

        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)

        return sha256_hash.hexdigest()

    @staticmethod
    def get_file_size(file_path: Union[str, Path]) -> int:
        """Get file size in bytes."""
        return Path(file_path).stat().st_size

    @staticmethod
    def create_product_bundle(
        products: List[ProductMetadata],
        output_path: Union[str, Path],
        bundle_format: str = "zip"
    ) -> str:
        """
        Create a bundle of products.

        Args:
            products: List of products to bundle
            output_path: Output bundle path
            bundle_format: Bundle format (zip, tar.gz)

        Returns:
            Path to created bundle
        """
        import zipfile
        import tarfile

        output_path = Path(output_path)

        if bundle_format == "zip":
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for product in products:
                    if product.file_path and Path(product.file_path).exists():
                        zf.write(product.file_path, Path(product.file_path).name)

        elif bundle_format in ("tar.gz", "tgz"):
            with tarfile.open(output_path, "w:gz") as tf:
                for product in products:
                    if product.file_path and Path(product.file_path).exists():
                        tf.add(product.file_path, Path(product.file_path).name)

        else:
            raise ValueError(f"Unsupported bundle format: {bundle_format}")

        return str(output_path)
