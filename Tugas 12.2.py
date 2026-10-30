
from transformers import pipeline

# ==========================================
# 1. TEXT CLASSIFICATION (SENTIMENT ANALYSIS)
# ==========================================

print("\n" + "=" * 60)
print("1. TEXT CLASSIFICATION - SENTIMENT ANALYSIS")
print("=" * 60)

classifier = pipeline(
    task="text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1
)

texts = [
    "This LLM API is incredibly fast and easy to use.",
    "The documentation is confusing and the latency is terrible.",
    "The model works fine for most tasks."
]

results = classifier(texts)

for text, result in zip(texts, results):
    print(
        f"[{result['label']} | {result['score']:.3f}] "
        f"{text[:60]}"
    )


# ==========================================
# 2. NAMED ENTITY RECOGNITION (NER)
# ==========================================

print("\n" + "=" * 60)
print("2. NAMED ENTITY RECOGNITION")
print("=" * 60)

ner = pipeline(
    task="ner",
    model="dbmdz/bert-large-cased-finetuned-conll03-english",
    aggregation_strategy="simple",
    device=-1
)

text = (
    "Anthropic was founded in San Francisco by "
    "Dario Amodei and Daniela Amodei."
)

entities = ner(text)

for ent in entities:
    print(
        f"{ent['word']!r:25s} -> "
        f"{ent['entity_group']} "
        f"({ent['score']:.3f})"
    )


# ==========================================
# 3. TEXT GENERATION
# ==========================================

print("\n" + "=" * 60)
print("3. TEXT GENERATION")
print("=" * 60)

generator = pipeline(
    task="text-generation",
    model="distilgpt2",
    device=-1
)

output = generator(
    "The future of AI agents is",
    max_new_tokens=50,
    num_return_sequences=2,
    temperature=0.8,
    do_sample=True,
    pad_token_id=50256
)

for i, seq in enumerate(output):
    print(f"\nSequence {i + 1}:")
    print(seq["generated_text"])


# ==========================================
# 4. TEXT SUMMARIZATION
# ==========================================

print("\n" + "=" * 60)
print("4. TEXT SUMMARIZATION")
print("=" * 60)

summariser = pipeline(
    task="summarization",
    model="facebook/bart-large-cnn",
    device=-1
)

article = """
Large language models have fundamentally changed how software
is built.

Developers now reach for AI APIs as naturally as they reach
for database clients.

The shift from rule-based systems to neural language models
has accelerated since the release of GPT-3 in 2020, and the
pace has only increased with instruction-tuned models like
ChatGPT, Claude, and Gemini.

These models can write code, explain concepts, extract
information, and reason through complex problems - tasks
that previously required specialised rule-based systems
or significant human effort.
"""

summary = summariser(
    article,
    max_length=60,
    min_length=20,
    do_sample=False
)

print("\nSummary:")
print(summary[0]["summary_text"])


# ==========================================
# SELESAI
# ==========================================

print("\n" + "=" * 60)
print("SEMUA TASK MODUL 12.2 SELESAI")
print("=" * 60)
