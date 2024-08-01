import os
import openai
from openai import OpenAI

client = OpenAI(
    api_key="api_token",
    base_url="https://adb-4040967622284013.13.azuredatabricks.net/serving-endpoints"
)

response = client.chat.completions.create(
    model="ai-doctor",
    messages="""Hello doctor, i have an acne, what should i do?""",
    max_tokens=100,
    temperature=0.9,
    top_p=0.95,
    stop=[".\n"],
)
