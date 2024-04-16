# Apache Software License 2.0
#
# Copyright (c) ZenML GmbH 2024. All rights reserved.
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

import os
import re
import string

from openai import OpenAI


def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    return preprocess_text(text).split()


def retrieve_relevant_chunks(query, corpus, top_n=2):
    query_tokens = set(tokenize(query))
    similarities = []
    for chunk in corpus:
        chunk_tokens = set(tokenize(chunk))
        similarity = len(query_tokens.intersection(chunk_tokens)) / len(
            query_tokens.union(chunk_tokens)
        )
        similarities.append((chunk, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in similarities[:top_n]]


def answer_question(query, corpus, top_n=2):
    relevant_chunks = retrieve_relevant_chunks(query, corpus, top_n)
    if not relevant_chunks:
        return "I don't have enough information to answer the question."

    context = "\n".join(relevant_chunks)
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f"Based on the provided context, answer the following question: {query}\n\nContext:\n{context}",
            },
            {
                "role": "user",
                "content": query,
            },
        ],
        model="gpt-3.5-turbo",
    )

    return chat_completion.choices[0].message.content.strip()


# Sci-fi themed corpus about "ZenML World"
corpus = [
    "The luminescent forests of ZenML World are inhabited by glowing Zenbots that emit a soft, pulsating light as they roam the enchanted landscape.",
    "In the neon skies of ZenML World, Cosmic Butterflies flutter gracefully, their iridescent wings leaving trails of stardust in their wake.",
    "Telepathic Treants, ancient sentient trees, communicate through the quantum neural network that spans the entire surface of ZenML World, sharing wisdom and knowledge.",
    "Deep within the melodic caverns of ZenML World, Fractal Fungi emit pulsating tones that resonate through the crystalline structures, creating a symphony of otherworldly sounds.",
    "Near the ethereal waterfalls of ZenML World, Holographic Hummingbirds hover effortlessly, their translucent wings refracting the prismatic light into mesmerizing patterns.",
    "Gravitational Geckos, masters of anti-gravity, traverse the inverted cliffs of ZenML World, defying the laws of physics with their extraordinary abilities.",
    "Plasma Phoenixes, majestic creatures of pure energy, soar above the chromatic canyons of ZenML World, their fiery trails painting the sky in a dazzling display of colors.",
    "Along the prismatic shores of ZenML World, Crystalline Crabs scuttle and burrow, their transparent exoskeletons refracting the light into a kaleidoscope of hues.",
]

corpus = [preprocess_text(sentence) for sentence in corpus]

# Evaluation data
eval_data = [
    {
        "question": "What creatures inhabit the luminescent forests of ZenML World?",
        "expected_answer": "The luminescent forests of ZenML World are inhabited by glowing Zenbots.",
    },
    {
        "question": "What do Fractal Fungi do in the melodic caverns of ZenML World?",
        "expected_answer": "Fractal Fungi emit pulsating tones that resonate through the crystalline structures, creating a symphony of otherworldly sounds in the melodic caverns of ZenML World.",
    },
    {
        "question": "Where do Gravitational Geckos live in ZenML World?",
        "expected_answer": "Gravitational Geckos traverse the inverted cliffs of ZenML World.",
    },
]


def evaluate_retrieval(question, expected_answer, corpus, top_n=2):
    relevant_chunks = retrieve_relevant_chunks(question, corpus, top_n)
    print(f"Question: {question}")
    print(f"Expected Answer: {expected_answer}")
    print("Retrieved Chunks:")
    for chunk in relevant_chunks:
        print(f"- {chunk}")
    score = any(
        any(word in chunk for word in tokenize(expected_answer))
        for chunk in relevant_chunks
    )
    print(f"Retrieval Score: {score}")
    print()
    return score


def evaluate_generation(question, expected_answer, generated_answer):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an evaluation judge. Given a question, an expected answer, and a generated answer, your task is to determine if the generated answer is relevant and accurate. Respond with 'YES' if the generated answer is satisfactory, or 'NO' if it is not.",
            },
            {
                "role": "user",
                "content": f"Question: {question}\nExpected Answer: {expected_answer}\nGenerated Answer: {generated_answer}\nIs the generated answer relevant and accurate?",
            },
        ],
        model="gpt-4",
    )

    judgment = chat_completion.choices[0].message.content.strip().lower()
    print(f"Question: {question}")
    print(f"Expected Answer: {expected_answer}")
    print(f"Generated Answer: {generated_answer}")
    print(f"Judgment: {judgment}")
    print()
    return judgment == "yes"


retrieval_scores = []
generation_scores = []

for item in eval_data:
    retrieval_score = evaluate_retrieval(
        item["question"], item["expected_answer"], corpus
    )
    retrieval_scores.append(retrieval_score)

    generated_answer = answer_question(item["question"], corpus)
    generation_score = evaluate_generation(
        item["question"], item["expected_answer"], generated_answer
    )
    generation_scores.append(generation_score)

retrieval_accuracy = sum(retrieval_scores) / len(retrieval_scores)
generation_accuracy = sum(generation_scores) / len(generation_scores)

print(f"Retrieval Accuracy: {retrieval_accuracy:.2f}")
print(f"Generation Accuracy: {generation_accuracy:.2f}")
