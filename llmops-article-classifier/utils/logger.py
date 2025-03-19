# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2025. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# ruff: noqa: E501

import logging

from colorama import Fore, Style
from colorama import init as colorama_init

colorama_init(autoreset=True)

COLORS = {
    "SUCCESS": Fore.LIGHTGREEN_EX,
    "PROCESS": Fore.CYAN,
    "BOLD_PROCESS": Style.BRIGHT + Fore.LIGHTMAGENTA_EX,
    "ALERT": Fore.YELLOW,
    "ERROR": Fore.RED,
    "ACCEPTED": Fore.LIGHTGREEN_EX,
    "REJECTED": Fore.LIGHTRED_EX,
    "YELLOW": Fore.YELLOW,
    "NUMBERS": Fore.LIGHTBLUE_EX + Style.BRIGHT,
    "CHECKPOINT": Fore.LIGHTBLUE_EX,
}


class CustomFormatter(logging.Formatter):
    """Custom formatter for non-specialized log messages."""

    def format(self, record):
        message = record.getMessage()
        record.args = ()

        if record.levelno == logging.ERROR:
            message = f"{COLORS['ERROR']}{Style.BRIGHT}ERROR: {message}{Style.RESET_ALL}"
        elif record.levelno == logging.WARNING:
            message = f"{COLORS['ALERT']}{Style.BRIGHT}WARNING: {message}{Style.RESET_ALL}"

        record.msg = message
        return super().format(record)


class StyledLogger(logging.Logger):
    """Custom formatter to add colors and bold formatting to log messages with enhanced JSON formatting."""

    def log_classification_type(self, value=None):
        """Log the classification type."""
        self.info(
            f"{COLORS['BOLD_PROCESS']}Loading dataset for {COLORS['YELLOW']}{value}{Style.RESET_ALL}"
        )

    def log_classification_config(
        self,
        remaining_articles,
        parallel_config=None,
        checkpoint_config=None,
        batch_config=None,
        total_articles=None,
    ):
        """
        Logs classification process details and configuration.

        Args:
            remaining_articles: List of articles to be processed
            parallel_config: Parallel processing configuration
            checkpoint_config: Checkpoint configuration
            batch_config: Batch processing configuration
            total_articles: Total number of articles in the batch (if different from remaining)
        """
        num_remaining = len(remaining_articles)
        use_parallel = parallel_config and parallel_config.enabled

        if total_articles is not None:
            num_processed = total_articles - num_remaining
            progress_pct = (num_processed / total_articles * 100) if total_articles > 0 else 0

            self.info(
                f"{COLORS['BOLD_PROCESS']}Classifying {COLORS['NUMBERS']}{num_remaining}{COLORS['BOLD_PROCESS']} articles "
                f"{COLORS['YELLOW']}{'in parallel' if use_parallel else 'sequentially'}{COLORS['BOLD_PROCESS']} "
                f"({COLORS['NUMBERS']}{num_processed}/{total_articles}{COLORS['BOLD_PROCESS']} already processed - "
                f"{COLORS['NUMBERS']}{progress_pct:.1f}%{COLORS['BOLD_PROCESS']} complete){Style.RESET_ALL}"
            )
        else:
            self.info(
                f"{COLORS['BOLD_PROCESS']}Classifying {COLORS['NUMBERS']}{num_remaining}{COLORS['BOLD_PROCESS']} articles "
                f"{COLORS['YELLOW']}{'in parallel' if use_parallel else 'sequentially'}{Style.RESET_ALL}"
            )

        if parallel_config and parallel_config.enabled:
            self.info(
                f"{COLORS['BOLD_PROCESS']}Number of workers: {COLORS['NUMBERS']}{parallel_config.workers}{Style.RESET_ALL}"
            )

        if checkpoint_config and checkpoint_config.frequency > 0:
            self.info(
                f"{COLORS['BOLD_PROCESS']}Checkpointing enabled with frequency: {COLORS['NUMBERS']}{checkpoint_config.frequency}{Style.RESET_ALL}"
            )

        if batch_config:
            if total_articles and len(remaining_articles) < total_articles:
                # resuming from a checkpoint
                self.info(
                    f"{COLORS['BOLD_PROCESS']}Processing batch: start={COLORS['NUMBERS']}{batch_config.batch_start}{Style.RESET_ALL}"
                    f"{COLORS['BOLD_PROCESS']}, size={COLORS['NUMBERS']}{batch_config.batch_size}{Style.RESET_ALL}"
                    f"{COLORS['BOLD_PROCESS']} (continuing from index {COLORS['NUMBERS']}{batch_config.batch_start + total_articles - len(remaining_articles)}{Style.RESET_ALL}"
                    f"{COLORS['BOLD_PROCESS']} to {COLORS['NUMBERS']}{batch_config.batch_start + batch_config.batch_size - 1}{Style.RESET_ALL})"
                )
            else:
                # fresh runs
                self.info(
                    f"{COLORS['BOLD_PROCESS']}Processing batch: start={COLORS['NUMBERS']}{batch_config.batch_start}{Style.RESET_ALL}"
                    f"{COLORS['BOLD_PROCESS']}, size={COLORS['NUMBERS']}{batch_config.batch_size}{Style.RESET_ALL}"
                )

    def log_classification(self, is_accepted: bool, reason: str, title: str, url: str = None):
        status = "ACCEPTED" if is_accepted else "REJECTED"
        self.info(f"{COLORS[status]}[{status}] {Style.BRIGHT}{title}{Style.RESET_ALL}")
        self.info(f"{COLORS[status]}reason: {reason}{Style.RESET_ALL}")

    def log_output_file(self, file_path: str, file_type: str = "Dataset"):
        """Log the output file path."""
        self.info(
            f"{COLORS['SUCCESS']}{Style.BRIGHT}{file_type} saved to disk at: {COLORS['SUCCESS']}{file_path}{Style.RESET_ALL}."
        )

    def log_process(self, message):
        self.info(f"{COLORS['PROCESS']}{message}{Style.RESET_ALL}")

    def log_success(self, message):
        self.info(f"{COLORS['SUCCESS']}{message}{Style.RESET_ALL}")

    def log_warning(self, message):
        self.info(f"{COLORS['ALERT']}{Style.BRIGHT}{message}{Style.RESET_ALL}")

    def log_checkpoint(self, message, value=None):
        """Log checkpoint-related information with a dedicated style."""
        self.info(f"{COLORS['CHECKPOINT']}{message}{Style.RESET_ALL}")


def setup_formatted_logger() -> StyledLogger:
    """Set up and configure the styled logger."""
    logging.setLoggerClass(StyledLogger)
    logger = logging.getLogger(__name__)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(CustomFormatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    return logger


logger = setup_formatted_logger()
