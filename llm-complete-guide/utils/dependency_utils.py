import os
import subprocess
import tempfile
from pathlib import Path
import logging
import shutil

logger = logging.getLogger(__name__)


def is_uv_installed() -> bool:
    """Check if uv is installed on the system.

    Returns:
        bool: True if uv is installed, False otherwise
    """
    return shutil.which("uv") is not None


def compile_requirements(requirements_path: str = "requirements.txt") -> str:
    """Compile requirements using uv to generate a locked requirements file.

    Args:
        requirements_path: Path to the requirements.txt file

    Returns:
        str: The compiled requirements as a string
    """
    if not is_uv_installed():
        logger.warning(
            "uv is not installed. For faster Gradio deployments, install uv first: "
            "'pip install uv'. This will significantly speed up dependency resolution "
            "during deployment. Falling back to standard requirements..."
        )

    try:
        # Create a temporary file to store the compiled requirements
        with tempfile.NamedTemporaryFile(
            mode="w+", delete=False, suffix=".txt"
        ) as tmp:
            # Run uv pip compile and capture output
            logger.info(
                "Compiling requirements with uv for faster deployment..."
            )
            result = subprocess.run(
                ["uv", "pip", "compile", requirements_path],
                capture_output=True,
                text=True,
                check=True,
            )

            # Write the compiled requirements to the temp file
            tmp.write(result.stdout)
            tmp.flush()

            # Read the compiled requirements
            with open(tmp.name, "r") as f:
                compiled_requirements = f.read()

            # Clean up the temporary file
            os.unlink(tmp.name)

            logger.info("Successfully compiled requirements with uv")
            return compiled_requirements

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to compile requirements: {e.stderr}")
        raise RuntimeError(f"Failed to compile requirements: {e.stderr}")
    except Exception as e:
        logger.error(f"Error during requirements compilation: {e}")
        raise
