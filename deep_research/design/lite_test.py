#!/usr/bin/env python3

import requests

# Configuration
LITELLM_BASE_URL = "https://litellm-service-5ikaahlouq-uc.a.run.app"
API_KEY = "zenmllitellm"


def test_model(model_name, prompt, max_tokens=100):
    """Test a specific model with a prompt"""

    url = f"{LITELLM_BASE_URL}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }

    try:
        print(f"\n🤖 Testing {model_name}...")
        print(f"📝 Prompt: {prompt}")
        print("⏳ Waiting for response...")

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()

        result = response.json()
        content = result["choices"][0]["message"]["content"]
        usage = result["usage"]

        print(f"✅ Response: {content}")
        print(
            f"📊 Tokens: {usage['prompt_tokens']} prompt + {usage['completion_tokens']} completion = {usage['total_tokens']} total"
        )

        return True

    except requests.exceptions.RequestException as e:
        print(f"❌ Error with {model_name}: {e}")
        return False
    except KeyError as e:
        print(f"❌ Unexpected response format: {e}")
        return False


def main():
    """Test multiple models with different prompts"""

    print("🚀 Testing LiteLLM Deployment")
    print("=" * 50)

    # Test cases: (model, prompt)
    test_cases = [
        ("gpt-4o", "Tell me a short joke about programming."),
        ("gpt-4o-mini", "What is the capital of Japan?"),
        ("claude-3-5-sonnet", "Explain quantum computing in one sentence."),
        ("claude-3-5-haiku", "Write a haiku about artificial intelligence."),
        ("gpt-3.5-turbo", "What's 15 * 23?"),
    ]

    successful_tests = 0
    total_tests = len(test_cases)

    for model, prompt in test_cases:
        if test_model(model, prompt):
            successful_tests += 1

    print("\n" + "=" * 50)
    print(f"🎯 Results: {successful_tests}/{total_tests} tests passed")

    if successful_tests == total_tests:
        print("🎉 All models are working perfectly!")
    else:
        print("⚠️  Some models may need attention.")


if __name__ == "__main__":
    main()
