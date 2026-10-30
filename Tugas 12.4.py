
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load a small model for local inference (CPU-friendly)
model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set padding token
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float32
)

# Set model to evaluation mode
model.eval()

# Input prompt
prompt = "The key advantage of local LLMs is"

# Tokenize input
inputs = tokenizer(
    prompt,
    return_tensors="pt",
    padding=True
)

# Generate text without calculating gradients
with torch.no_grad():
    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=40,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

# Decode generated tokens into text
generated = tokenizer.decode(
    output_ids[0],
    skip_special_tokens=True
)

# Display generated text
print("Generated Text:")
print(generated)
