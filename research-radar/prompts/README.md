# Prompts Directory

This directory contains the prompts used for article classification by DeepSeek R1 and Claude Haiku models.

## Files

### `system_prompt.txt`

Contains the system context for the classification task, providing detailed criteria for what constitutes a valid GenAI case study. The prompt includes:

- Real-world deployment evidence requirements
- Operational implementation details criteria
- Production experience validation points
- Clear rejection criteria

Reference: [system_prompt.txt](./system_prompt.txt)

### `user_prompt.txt`

The base prompt used for both DeepSeek R1 and Claude Haiku classifications. This prompt:

- Requests evaluation of articles for GenAI case study validity
- Specifies the exact JSON structure for responses

Reference: [user_prompt.txt](./user_prompt.txt)

### `room_to_think_prompt.txt`

An experimental variant of the user prompt that includes explicit instructions for the model to:

1. Take time to reason internally before providing an answer
2. Use chain-of-thought processing while keeping the output clean
3. Provide a final, considered response

This prompt was used to test whether giving DeepSeek R1 explicit "thinking time" instructions would improve classification accuracy.

The prompt is not used in the final pipeline.

Reference: [room_to_think_prompt.txt](./room_to_think_prompt.txt)
