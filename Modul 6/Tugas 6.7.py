from abc import ABC, abstractmethod
from dataclasses import dataclass
import requests


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class ChatResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str


class BaseLLMClient(ABC):
    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.7
    ) -> ChatResponse:
        pass


class OllamaClient(BaseLLMClient):
    def __init__(self, model: str = "qwen2.5:3b"):
        self.model = model
        self.url = "http://localhost:11434/api/chat"

    def chat(
        self,
        messages,
        system="",
        max_tokens=1024,
        temperature=0.7
    ) -> ChatResponse:

        api_messages = []

        if system:
            api_messages.append({
                "role": "system",
                "content": system
            })

        api_messages += [
            {
                "role": m.role,
                "content": m.content
            }
            for m in messages
        ]

        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "messages": api_messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
        )

        response.raise_for_status()

        result = response.json()

        text = result["message"]["content"]

        input_tokens = result.get("prompt_eval_count", 0)
        output_tokens = result.get("eval_count", 0)

        return ChatResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self.model
        )



client: BaseLLMClient = OllamaClient()

msgs = [
    ChatMessage(
        role="user",
        content="What is a vector database?"
    )
]

result = client.chat(
    msgs,
    system="Be concise."
)

print("Model:", result.model)
print("Response:", result.text)
print("Input tokens:", result.input_tokens)
print("Output tokens:", result.output_tokens)