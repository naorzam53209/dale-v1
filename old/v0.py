import sys
import os 

sys.stdout.reconfigure(encoding="utf-8")  # so emoji don't crash the Windows console

from openai import OpenAI

# One client. Point it at any OpenAI-compatible server (Ollama here).
client = OpenAI(base_url=os.getenv("OLLAMA_URL", "http://localhost:11434/v1"),
    api_key="ollama")
MODEL = "qwen3.5:4b"
print("using:", client.base_url)

def chat(user_message: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful personal assistant."},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content or ""


if __name__ == "__main__":
    print(f"v0 assistant ({MODEL}) - ctrl+c to quit")
    try:
        while True:
            question = input("\nwrite a messege... ")
            print("\nassistant:", chat(question))
    except (EOFError, KeyboardInterrupt):
        print("\nbye")