"""
pygeoapi configuration generator.

Generates a pygeoapi-compatible configuration dict at startup by pulling
process definitions from FirstLight's AlgorithmRegistry. The config is
used to initialize the pygeoapi ASGI application which is then mounted
at /oapi in the FastAPI app.

pygeoapi normally reads from a YAML file, but also supports receiving
a dict directly via its API class.
"""

import logging
import os
import tempfile
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


def generate_pygeoapi_config(
    base_url: str = "/oapi",
    title: str = "FirstLight OGC API",
    description: str = (
        "OGC API Processes endpoint for FirstLight geospatial event "
        "intelligence algorithms. Provides standards-based discovery "
        "and execution of analysis algorithms."
    ),
) -> Dict[str, Any]:
    """
    Generate a complete pygeoapi configuration dict.

    This function builds the server/metadata/resources sections needed
    by pygeoapi. Process resources are generated from the AlgorithmRegistry
    via get_processor_config().

    Args:
        base_url: The base URL path where pygeoapi is mounted.
        title: API title for OGC metadata.
        description: API description.

    Returns:
        A dict suitable for pygeoapi configuration.
    """
    from core.ogc.processors.factory import get_processor_config

    resources = get_processor_config()

    config: Dict[str, Any] = {
        "server": {
            "bind": {
                "host": "0.0.0.0",
                "port": 5000,
            },
            "url": base_url,
            "mimetype": "application/json",
            "encoding": "utf-8",
            "languages": ["en-US"],
            "cors": True,
            "pretty_print": True,
            "limit": 100,
            "map": {
                "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
                "attribution": "OpenStreetMap contributors",
            },
        },
        "logging": {
            "level": "WARNING",
            "logfile": "",
        },
        "metadata": {
            "identification": {
                "title": title,
                "description": description,
                "keywords": [
                    "geospatial",
                    "event intelligence",
                    "OGC",
                    "processes",
                    "flood",
                    "wildfire",
                    "storm",
                ],
                "keywords_type": "theme",
                "terms_of_service": "https://firstlight.example.com/tos",
                "url": "https://firstlight.example.com",
            },
            "license": {
                "name": "MIT",
                "url": "https://opensource.org/licenses/MIT",
            },
            "provider": {
                "name": "FirstLight",
                "url": "https://firstlight.example.com",
            },
            "contact": {
                "name": "FirstLight Team",
                "email": "support@firstlight.example.com",
            },
        },
        "resources": resources,
    }

    return config


def write_pygeoapi_config(
    config: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
) -> str:
    """
    Write pygeoapi config to a YAML file.

    If pygeoapi requires a file path (rather than a dict), this function
    writes the config to a temporary file and returns the path.

    Args:
        config: Configuration dict. If None, generates one.
        output_path: Where to write. If None, uses a temp file.

    Returns:
        Path to the written YAML file.
    """
    if config is None:
        config = generate_pygeoapi_config()

    if output_path is None:
        fd, output_path = tempfile.mkstemp(
            suffix=".yml", prefix="pygeoapi_"
        )
        os.close(fd)

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    logger.info("pygeoapi config written to %s", output_path)
    return output_path


def create_pygeoapi_app():
    """
    Create a pygeoapi Starlette/ASGI application.

    Returns None if pygeoapi is not installed.
    """
    try:
        from pygeoapi.flask_app import CONFIG as pygeoapi_config_module
    except ImportError:
        pass

    try:
        from pygeoapi.openapi import get_oas
        from pygeoapi.starlette_app import app as pygeoapi_starlette_app

        # pygeoapi reads PYGEOAPI_CONFIG and PYGEOAPI_OPENAPI env vars.
        # We generate the config file and set the env var before import.
        config = generate_pygeoapi_config()
        config_path = write_pygeoapi_config(config)
        os.environ["PYGEOAPI_CONFIG"] = config_path

        # Generate OpenAPI doc
        openapi_path = config_path.replace(".yml", "_openapi.yml")
        try:
            openapi_doc = get_oas(config)
            with open(openapi_path, "w") as f:
                yaml.dump(openapi_doc, f, default_flow_style=False)
            os.environ["PYGEOAPI_OPENAPI"] = openapi_path
        except Exception as e:
            logger.warning("Could not generate pygeoapi OpenAPI doc: %s", e)

        logger.info("pygeoapi ASGI application created")
        return pygeoapi_starlette_app

    except ImportError:
        logger.info(
            "pygeoapi not installed. OGC API Processes endpoint disabled. "
            "Install with: pip install firstlight[control-plane]"
        )
        return None
    except Exception as e:
        logger.warning("Failed to create pygeoapi app: %s", e)
        return None
