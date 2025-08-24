"""
Run tests and collect coverage metrics.
"""

import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Annotated, Dict, Optional

from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)


@step
def run_tests(
    workspace_dir: Path,
    tests_dir: Optional[Path],
    label: str = "tests",
) -> Annotated[Dict, "test_results"]:
    """Run tests and collect coverage metrics.
    
    Args:
        workspace_dir: Path to workspace directory
        tests_dir: Path object to tests directory (None if no tests)
        label: Label for this test run
        
    Returns:
        Dictionary containing test results and metrics
    """
    if tests_dir is None:
        logger.info(f"No tests directory provided for {label}, skipping")
        return {
            "label": label,
            "tests_passed": 0,
            "tests_failed": 0,
            "coverage_total": 0.0,
            "coverage_by_file": {},
            "junit_path": None,
            "coverage_path": None,
            "logs_path": None,
            "skipped": True,
        }
    
    logger.info(f"Running {label} tests from {tests_dir}")
    
    # Create output directory for this test run
    output_dir = tempfile.mkdtemp(prefix=f"qualityflow_{label}_results_")
    output_path = Path(output_dir)
    
    junit_file = output_path / "junit.xml"
    coverage_file = output_path / "coverage.xml"
    logs_file = output_path / "test_logs.txt"
    
    # Copy tests to workspace (pytest needs them in PYTHONPATH)
    workspace_tests_dir = Path(workspace_dir) / f"tests_{label}"
    if workspace_tests_dir.exists():
        shutil.rmtree(workspace_tests_dir)
    shutil.copytree(tests_dir, workspace_tests_dir)
    
    try:
        # Run pytest with coverage
        pytest_cmd = [
            "python", "-m", "pytest",
            str(workspace_tests_dir),
            "--junitxml", str(junit_file),
            "--cov", str(workspace_dir),
            "--cov-report", f"xml:{coverage_file}",
            "--cov-report", "term",
            "-v"
        ]
        
        logger.info(f"Running command: {' '.join(pytest_cmd)}")
        logger.info(f"Working directory: {workspace_dir}")
        logger.info(f"Test directory: {workspace_tests_dir}")
        
        # Debug: list test files
        if workspace_tests_dir.exists():
            test_files = list(workspace_tests_dir.glob("*.py"))
            logger.info(f"Test files found: {[f.name for f in test_files]}")
        else:
            logger.warning(f"Test directory does not exist: {workspace_tests_dir}")
        
        result = subprocess.run(
            pytest_cmd,
            cwd=str(workspace_dir),
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        
        # Save logs and also log to console for debugging
        with open(logs_file, 'w') as f:
            f.write(f"Command: {' '.join(pytest_cmd)}\n")
            f.write(f"Return code: {result.returncode}\n\n")
            f.write("STDOUT:\n")
            f.write(result.stdout)
            f.write("\nSTDERR:\n") 
            f.write(result.stderr)
        
        # Also log the pytest output for debugging
        logger.info(f"Pytest return code: {result.returncode}")
        if result.stdout:
            logger.info(f"Pytest stdout: {result.stdout}")
        if result.stderr:
            logger.info(f"Pytest stderr: {result.stderr}")
        
        # Parse results
        test_results = _parse_test_results(
            result, junit_file, coverage_file, logs_file, label
        )
        
        logger.info(f"Test run complete for {label}: {test_results['tests_passed']} passed, {test_results['tests_failed']} failed, {test_results['coverage_total']:.2f}% coverage")
        
        return test_results
        
    except subprocess.TimeoutExpired:
        logger.error(f"Test run for {label} timed out after 5 minutes")
        return {
            "label": label,
            "tests_passed": 0,
            "tests_failed": 1,
            "coverage_total": 0.0,
            "coverage_by_file": {},
            "junit_path": str(junit_file) if junit_file.exists() else None,
            "coverage_path": str(coverage_file) if coverage_file.exists() else None,
            "logs_path": str(logs_file),
            "error": "Test execution timed out",
        }
        
    except Exception as e:
        logger.error(f"Failed to run tests for {label}: {e}")
        return {
            "label": label,
            "tests_passed": 0,
            "tests_failed": 1,
            "coverage_total": 0.0,
            "coverage_by_file": {},
            "junit_path": str(junit_file) if junit_file.exists() else None,
            "coverage_path": str(coverage_file) if coverage_file.exists() else None,
            "logs_path": str(logs_file) if logs_file.exists() else None,
            "error": str(e),
        }
    
    finally:
        # Clean up copied tests
        if workspace_tests_dir.exists():
            shutil.rmtree(workspace_tests_dir, ignore_errors=True)


def _parse_test_results(
    result: subprocess.CompletedProcess,
    junit_file: Path,
    coverage_file: Path, 
    logs_file: Path,
    label: str,
) -> Dict:
    """Parse test execution results."""
    
    # Parse pytest output for basic stats
    tests_passed = 0
    tests_failed = 0
    
    if result.stdout:
        lines = result.stdout.split('\n')
        for line in lines:
            if ' passed' in line and ' failed' in line:
                # Line like "2 failed, 3 passed in 1.23s"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'passed' and i > 0:
                        tests_passed = int(parts[i-1])
                    elif part == 'failed' and i > 0:
                        tests_failed = int(parts[i-1])
            elif ' passed' in line and 'failed' not in line:
                # Line like "5 passed in 1.23s"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'passed' and i > 0:
                        tests_passed = int(parts[i-1])
    
    # Parse coverage from XML if available
    coverage_total = 0.0
    coverage_by_file = {}
    
    if coverage_file.exists():
        coverage_total, coverage_by_file = _parse_coverage_xml(coverage_file)
    
    return {
        "label": label,
        "tests_passed": tests_passed,
        "tests_failed": tests_failed,
        "coverage_total": coverage_total,
        "coverage_by_file": coverage_by_file,
        "junit_path": str(junit_file) if junit_file.exists() else None,
        "coverage_path": str(coverage_file) if coverage_file.exists() else None,
        "logs_path": str(logs_file),
        "return_code": result.returncode,
    }


def _parse_coverage_xml(coverage_file: Path) -> tuple[float, Dict[str, float]]:
    """Parse coverage XML file."""
    try:
        import xml.etree.ElementTree as ET
        
        tree = ET.parse(coverage_file)
        root = tree.getroot()
        
        # Debug: log the XML structure
        logger.info(f"Coverage XML root tag: {root.tag}")
        logger.info(f"Coverage XML root attribs: {root.attrib}")
        
        # Get overall coverage - try different formats
        coverage_total = 0.0
        
        # Modern pytest-cov uses 'coverage' as root element
        if root.tag == 'coverage':
            line_rate = root.get('line-rate', '0')
            if line_rate != '0':
                coverage_total = float(line_rate) * 100
                logger.info(f"Found line-rate in coverage root: {line_rate}")
        else:
            # Try finding coverage element nested
            coverage_element = root.find('.//coverage')
            if coverage_element is not None:
                line_rate = coverage_element.get('line-rate', '0')
                coverage_total = float(line_rate) * 100
                logger.info(f"Found coverage element with line-rate: {line_rate}")
        
        # If still no coverage found, try branches-valid attribute (alternative format)
        if coverage_total == 0.0:
            branches_valid = root.get('branches-valid', '0')
            branches_covered = root.get('branches-covered', '0') 
            lines_valid = root.get('lines-valid', '0')
            lines_covered = root.get('lines-covered', '0')
            
            if lines_valid != '0':
                line_coverage = float(lines_covered) / float(lines_valid)
                coverage_total = line_coverage * 100
                logger.info(f"Calculated coverage from lines: {lines_covered}/{lines_valid} = {coverage_total:.2f}%")
        
        # Get per-file coverage
        coverage_by_file = {}
        for class_elem in root.findall('.//class'):
            filename = class_elem.get('filename', '')
            line_rate = class_elem.get('line-rate', '0')
            if filename:
                coverage_by_file[filename] = float(line_rate) * 100
        
        logger.info(f"Parsed coverage: {coverage_total}% total, {len(coverage_by_file)} files")
        return coverage_total, coverage_by_file
        
    except Exception as e:
        logger.warning(f"Failed to parse coverage XML: {e}")
        return 0.0, {}