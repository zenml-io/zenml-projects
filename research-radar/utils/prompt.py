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

import re

from bs4 import BeautifulSoup
from transformers import AutoTokenizer

from utils import logger


def load_system_prompt():
    with open("prompts/system_prompt.txt", "r") as f:
        return f.read()


def load_user_prompt():
    with open("prompts/user_prompt.txt", "r") as f:
        return f.read()


SYSTEM_PROMPT = load_system_prompt()
USER_PROMPT = load_user_prompt()
DEEPSEEK_PROMPT = f"{SYSTEM_PROMPT}\n\n{USER_PROMPT}"

TOKENIZER = AutoTokenizer.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
)
PROMPT_MARGIN = 2500


def truncate_text_by_token_limit(text: str, max_tokens: int) -> str:
    """
    Truncate text to a maximum number of tokens using the model's tokenizer.
    Uses a rough character cutoff (average ~4 characters per token) before tokenizing.

    Args:
        text: The text to truncate
        max_tokens: The maximum number of tokens to truncate to

    Returns:
        The truncated text
    """
    char_limit = max_tokens * 4
    text = text[:char_limit]
    tokens = TOKENIZER.encode(text)
    truncated_tokens = tokens[:max_tokens]
    return TOKENIZER.decode(truncated_tokens, skip_special_tokens=True)


def clean_text(text: str) -> str:
    """
    Remove HTML artifacts, URLs, extra newlines, and collapse whitespace.

    Args:
        text: The text to clean

    Returns:
        The cleaned text
    """
    soup = BeautifulSoup(text, "html.parser")
    clean = soup.get_text()
    clean = re.sub(r"https?://\S+", "", clean)
    clean = re.sub(r"\n\s*\n", "\n", clean)
    clean = re.sub(r"\s+", " ", clean)
    return clean.strip()


def preprocess_article(text: str, article_max_tokens: int) -> str:
    """
    Preprocess the article text by first truncating it by token count,
    then cleaning up the resulting text.

    Args:
        text: The text to preprocess
        article_max_tokens: The maximum number of tokens to truncate to

    Returns:
        The preprocessed text
    """
    truncated_text = truncate_text_by_token_limit(text, article_max_tokens)
    return clean_text(truncated_text)


def format_prompt_for_deepseek(article_text: str, max_tokens: int) -> str:
    """
    Create a prompt that includes a base prompt and preprocessed article text.

    Args:
        article_text: The text of the article to evaluate
        max_tokens: The maximum number of tokens to truncate to

    Returns:
        The formatted prompt
    """
    base_prompt_tokens = len(TOKENIZER(DEEPSEEK_PROMPT)["input_ids"])
    available_article_tokens = max_tokens - base_prompt_tokens - PROMPT_MARGIN

    processed_text = preprocess_article(article_text, available_article_tokens)
    final_prompt = DEEPSEEK_PROMPT.replace("{{article_text}}", processed_text)

    final_token_count = len(TOKENIZER(final_prompt)["input_ids"])

    if final_token_count > max_tokens:
        logger.log_warning(
            f"Final token count {final_token_count} exceeds max allowed {max_tokens}"
        )

    return final_prompt
