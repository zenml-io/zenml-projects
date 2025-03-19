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

import asyncio
import time
from typing import List

from anthropic import AsyncAnthropic
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from schemas import ClaudeResponse
from utils import (
    SYSTEM_PROMPT,
    USER_PROMPT,
    clean_text,
    logger,
    try_extract_json_from_text,
)


class ClaudeEvaluator:
    """
    Class for evaluating the performance of the Claude model
    """

    def __init__(self, api_key: str, batch_size: int = 3, concurrent_requests: int = 1):
        self.client = AsyncAnthropic(api_key=api_key)
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(concurrent_requests)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4),
        retry=retry_if_exception_type((Exception)),
        reraise=True,
    )
    async def _get_raw_claude_response(self, text: str) -> dict:
        """
        Get the response from the Claude model

        Args:
            text (str): The text to evaluate

        Returns:
            dict: The response from the Claude model
        """

        async with self.semaphore:
            response = await self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                temperature=0.2,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": USER_PROMPT.replace("{{article_text}}", text)}
                ],
            )
        return response

    async def evaluate_single(self, text: str) -> ClaudeResponse:
        """
        Evaluate a single text using the Claude model

        Args:
            text (str): The text to evaluate

        Returns:
            ClaudeResponse: The response from the Claude model
        """
        try:
            start_time = time.time()
            cleaned_text = clean_text(text)
            response = await self._get_raw_claude_response(cleaned_text)

            raw_response = response.content[0].text
            logger.log_process(raw_response)

            _, json_data = try_extract_json_from_text(raw_response)
            if not json_data:
                raise ValueError("Failed to parse JSON response")

            result = ClaudeResponse(
                prediction=1 if json_data.get("is_accepted", False) else 0,
                latency=time.time() - start_time,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                raw_response=raw_response,
                confidence=json_data.get("confidence", -1),
                reason=json_data.get("reason", ""),
            )
            return result

        except Exception as e:
            return ClaudeResponse(
                prediction=-1,
                latency=-1,
                input_tokens=-1,
                output_tokens=-1,
                raw_response=str(e),
                error=str(e),
            )

    async def evaluate_batch(self, texts: List[str]) -> List[ClaudeResponse]:
        """
        Evaluate a batch of texts using the Claude model

        Args:
            texts (List[str]): The texts to evaluate

        Returns:
            List[ClaudeResponse]: The responses from the Claude model
        """

        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_results = await asyncio.gather(*[self.evaluate_single(text) for text in batch])
            results.extend(batch_results)
            await asyncio.sleep(3)

        return results
