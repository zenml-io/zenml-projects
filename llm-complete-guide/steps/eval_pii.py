import re
from collections import defaultdict
from typing import Dict, List, Union

from datasets import Dataset
from zenml import log_artifact_metadata, step


class PIIDetector:
    """A class to detect PII in HuggingFace datasets."""

    def __init__(self):
        # Email regex pattern
        self.email_pattern = re.compile(
            r"""
            (?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")
            @
            (?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])
        """,
            re.VERBOSE | re.IGNORECASE,
        )

        # Phone number patterns (US formats)
        self.phone_pattern = re.compile(
            r"""
            (?:
                # Format: (123) 456-7890 or 123-456-7890
                (?:\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4}))|
                # Format: +1 123-456-7890 or +1 (123) 456-7890
                (?:\+1[-.\s]?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4}))|
                # Format: 1234567890
                (?:[0-9]{10})
            )
        """,
            re.VERBOSE,
        )

        # SSN pattern (XXX-XX-XXXX)
        self.ssn_pattern = re.compile(
            r"""
            (?!000|666|9\d{2})  # SSN cannot start with 000, 666, or 900-999
            ([0-8]\d{2}|7([0-6]\d))
            [-\s]?
            (?!00)              # Cannot have 00 in the middle group
            ([0-9]{2})
            [-\s]?
            (?!0000)            # Cannot end with 0000
            ([0-9]{4})
        """,
            re.VERBOSE,
        )

        # Credit card pattern (major card types)
        self.credit_card_pattern = re.compile(
            r"""
            (?:
                # Visa
                4[0-9]{12}(?:[0-9]{3})?|
                # Mastercard
                (?:5[1-5][0-9]{2}|222[1-9]|22[3-9][0-9]|2[3-6][0-9]{2}|27[01][0-9]|2720)[0-9]{12}|
                # American Express
                3[47][0-9]{13}|
                # Discover
                6(?:011|5[0-9][0-9])[0-9]{12}
            )
        """,
            re.VERBOSE,
        )

        # IP address pattern (IPv4)
        self.ip_pattern = re.compile(
            r"""
            \b
            (?:
                (?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.
                (?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.
                (?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.
                (?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)
            )
            \b
        """,
            re.VERBOSE,
        )

        # Date pattern (common formats)
        self.date_pattern = re.compile(
            r"""
            (?:
                # MM/DD/YYYY or MM-DD-YYYY
                (?:0[1-9]|1[0-2])[/-](?:0[1-9]|[12][0-9]|3[01])[/-](?:19|20)\d\d|
                # YYYY/MM/DD or YYYY-MM-DD
                (?:19|20)\d\d[/-](?:0[1-9]|1[0-2])[/-](?:0[1-9]|[12][0-9]|3[01])|
                # Month DD, YYYY
                (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|
                   Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|
                   Dec(?:ember)?)\s+(?:0[1-9]|[12][0-9]|3[01])(?:,|\s)+(?:19|20)\d\d
            )
        """,
            re.VERBOSE | re.IGNORECASE,
        )

    def find_pii(self, text: str) -> Dict[str, List[str]]:
        """
        Find all PII in a given text.

        Args:
            text (str): The text to search for PII

        Returns:
            Dict[str, List[str]]: Dictionary of PII types and their findings
        """
        if not isinstance(text, str):
            return {
                "emails": [],
                "phones": [],
                "ssns": [],
                "credit_cards": [],
                "dates": [],
                "ips": [],
            }

        return {
            "emails": self.email_pattern.findall(text),
            "phones": self.phone_pattern.findall(text),
            "ssns": self.ssn_pattern.findall(text),
            "credit_cards": self.credit_card_pattern.findall(text),
            "dates": self.date_pattern.findall(text),
            "ips": self.ip_pattern.findall(text),
        }

    def scan_dataset(
        self,
        dataset: Dataset,
        columns: Union[List[str], None] = None,
        max_samples: int = None,
    ) -> Dict[str, Dict]:
        """Scan a HuggingFace dataset for PII (currently only emails).

        Args:
            dataset (Dataset): HuggingFace dataset to scan
            columns (List[str], optional): Specific columns to scan. If None, scans all string columns
            max_samples (int, optional): Maximum number of samples to scan. If None, scans entire dataset

        Returns:
            Dict[str, Dict]: Dictionary containing:
                - 'statistics': Overall statistics about the scan
                - 'findings': Detailed findings per column
        """
        # Initialize results
        results = {
            "statistics": {
                "total_samples_scanned": 0,
                "columns_scanned": 0,
                "total_findings": {
                    "emails": 0,
                    "phones": 0,
                    "ssns": 0,
                    "credit_cards": 0,
                    "dates": 0,
                    "ips": 0,
                },
            },
            "findings": defaultdict(list),
        }

        # Determine which columns to scan
        if columns is None:
            # Get all columns that contain string data
            columns = [
                col
                for col in dataset.column_names
                if dataset.features[col].dtype in ["string", "str"]
            ]

        results["statistics"]["columns_scanned"] = len(columns)

        # Determine number of samples to scan
        n_samples = (
            len(dataset)
            if max_samples is None
            else min(max_samples, len(dataset))
        )
        results["statistics"]["total_samples_scanned"] = n_samples

        # Scan the dataset
        for idx in range(n_samples):
            sample = dataset[idx]

            for column in columns:
                if column not in sample:
                    continue

                text = sample[column]
                pii_findings = self.find_pii(text)

                # Check if any PII was found
                if any(findings for findings in pii_findings.values()):
                    # Update statistics
                    for pii_type, findings in pii_findings.items():
                        results["statistics"]["total_findings"][pii_type] += (
                            len(findings)
                        )

                    # Record detailed findings
                    results["findings"][column].append(
                        {"index": idx, "findings": pii_findings}
                    )

        return results


@step
def eval_pii(train_dataset: Dataset, test_dataset: Dataset) -> None:
    detector = PIIDetector()
    train_results = detector.scan_dataset(
        dataset=train_dataset,
        columns=[
            "text"
        ],  # specify columns to scan, or None for all string columns
        max_samples=1000,  # optional: limit number of samples to scan
    )
    test_results = detector.scan_dataset(
        dataset=test_dataset, columns=["text"], max_samples=1000
    )
    # Log train results
    train_metadata = {
        "samples_scanned": train_results["statistics"][
            "total_samples_scanned"
        ],
        "emails_found": train_results["statistics"]["total_findings"][
            "emails"
        ],
        "phones_found": train_results["statistics"]["total_findings"][
            "phones"
        ],
        "ssns_found": train_results["statistics"]["total_findings"]["ssns"],
        "credit_cards_found": train_results["statistics"]["total_findings"][
            "credit_cards"
        ],
        "dates_found": train_results["statistics"]["total_findings"]["dates"],
        "ips_found": train_results["statistics"]["total_findings"]["ips"],
    }
    log_artifact_metadata(
        metadata=train_metadata, artifact_name="train_pii_results"
    )

    # Log test results
    test_metadata = {
        "samples_scanned": test_results["statistics"]["total_samples_scanned"],
        "emails_found": test_results["statistics"]["total_findings"]["emails"],
        "phones_found": test_results["statistics"]["total_findings"]["phones"],
        "ssns_found": test_results["statistics"]["total_findings"]["ssns"],
        "credit_cards_found": test_results["statistics"]["total_findings"][
            "credit_cards"
        ],
        "dates_found": test_results["statistics"]["total_findings"]["dates"],
        "ips_found": test_results["statistics"]["total_findings"]["ips"],
    }
    log_artifact_metadata(
        metadata=test_metadata, artifact_name="test_pii_results"
    )

    return train_results, test_results
