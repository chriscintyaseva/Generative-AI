
from transformers import AutoTokenizer

# Load tokenizer for GPT-2
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Set padding token (GPT-2 does not have a default padding token)
tokenizer.pad_token = tokenizer.eos_token

# Text to tokenize
text = "Retrieval-Augmented Generation (RAG) combines search with LLMs."

# Encode text into token IDs
tokens = tokenizer.encode(text)

print(f"Token count: {len(tokens)}")
print(f"Token IDs: {tokens}")

# Decode token IDs back to text
decoded = tokenizer.decode(tokens)
print(f"Decoded: {decoded}")

# Convert token IDs into individual tokens
token_strings = tokenizer.convert_ids_to_tokens(tokens)
print(f"Tokens: {token_strings}")

# Batch encoding with padding and truncation
batch = [
    "What is RAG?",
    "Explain vector databases in detail."
]

encoded = tokenizer(
    batch,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)

print(f"\nBatch input_ids shape: {encoded['input_ids'].shape}")
print(f"Attention mask shape: {encoded['attention_mask'].shape}")

# Function to count tokens before sending a prompt to a model
def count_tokens(text: str, model: str = "gpt2") -> int:
    tok = AutoTokenizer.from_pretrained(model)
    return len(tok.encode(text))

# Count tokens in a sample prompt
prompt = "Summarise the following document: " + "word " * 500

print(f"\nPrompt token count: {count_tokens(prompt)}")
