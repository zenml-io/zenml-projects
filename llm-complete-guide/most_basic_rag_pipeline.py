import os
import re
import string

import numpy as np
from openai import OpenAI


def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    return preprocess_text(text).split()


def split_into_chunks(sentence, chunk_size=5):
    words = sentence.split()
    return [
        " ".join(words[i : i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]


def build_vocab(corpus):
    vocab = {}
    for chunk in corpus:
        for word in chunk:
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab


def train_word2vec(corpus, vocab, vector_size, learning_rate, epochs):
    embeddings = np.random.uniform(-1, 1, (len(vocab), vector_size))
    for _ in range(epochs):
        for chunk in corpus:
            for center_word in chunk:
                center_word_index = vocab[center_word]
                context_words = [word for word in chunk if word != center_word]
                for context_word in context_words:
                    context_word_index = vocab[context_word]
                    center_word_vec = embeddings[center_word_index]
                    context_word_vec = embeddings[context_word_index]
                    error = np.dot(center_word_vec, context_word_vec)
                    embeddings[center_word_index] -= (
                        learning_rate * error * context_word_vec
                    )
                    embeddings[context_word_index] -= (
                        learning_rate * error * center_word_vec
                    )
    return embeddings


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def answer_question(query, corpus, vocab, embeddings, top_n=2):
    query_tokens = tokenize(query)
    query_embeddings = []
    for token in query_tokens:
        if token in vocab:
            token_embedding = embeddings[vocab[token]]
            query_embeddings.append(token_embedding)
    if not query_embeddings:
        return "I don't have enough information to answer the question."

    query_embedding = np.mean(query_embeddings, axis=0)
    similarities = []
    for chunk in corpus:
        chunk_tokens = tokenize(" ".join(chunk))
        if chunk_embeddings := [
            embeddings[vocab[token]]
            for token in chunk_tokens
            if token in vocab
        ]:
            chunk_embedding = np.mean(chunk_embeddings, axis=0)
            similarity = cosine_similarity(query_embedding, chunk_embedding)
            similarities.append((chunk, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)

    if similarities:
        top_chunks = [" ".join(chunk[0]) for chunk in similarities[:top_n]]
        context = "\n".join(top_chunks)

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

        return chat_completion["choices"][0]["message"]["content"].strip()
    else:
        return "I don't have enough information to answer the question."


# Sci-fi themed corpus about "ZenML World" with longer sentences
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

print("Starting preprocessing stage...")
# Split the corpus into smaller chunks
chunk_size = 45
corpus_chunks = []
for sentence in corpus:
    chunks = split_into_chunks(preprocess_text(sentence), chunk_size)
    corpus_chunks.extend(chunks)

print("Starting vocabulary building stage...")
# Build vocabulary
vocab = build_vocab(corpus_chunks)

print("Starting training stage...")
# Train word2vec model
vector_size = 50
learning_rate = 0.01
epochs = 200
embeddings = train_word2vec(
    corpus_chunks, vocab, vector_size, learning_rate, epochs
)

print("Model training complete!")
print("Model evaluation:")
# Ask questions
question1 = "What are Plasma Phoenixes?"
answer1 = answer_question(question1, corpus_chunks, vocab, embeddings)
print(f"Question: {question1}")
print(f"Answer: {answer1}")

question2 = (
    "What kinds of creatures live on the prismatic shores of ZenML World?"
)
answer2 = answer_question(question2, corpus_chunks, vocab, embeddings)
print(f"Question: {question2}")
print(f"Answer: {answer2}")
