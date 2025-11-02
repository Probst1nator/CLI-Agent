import asyncio

from anthropic import Anthropic

# No API key needed - uses your Claude Code subscription auth
client = Anthropic()

async def custom_experiment():
    message = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "Write a hello world in Rust"}
        ]
    )
    print(message.content[0].text)

asyncio.run(custom_experiment())