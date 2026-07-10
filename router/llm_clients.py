import os
import time
from groq import Groq

# Map our model IDs → actual Groq model strings
GROQ_MODEL_MAP = {
    "claude":  "llama-3.3-70b-versatile",
    "gpt4":    "mixtral-8x7b-32768",
    "gemini":  "gemma2-9b-it",
    "mistral": "llama-3.1-8b-instant",
    "llama":   "llama-3.3-70b-versatile",
}

MODEL_REGISTRY = {
    "claude": {
        "id": "claude",
        "name": "Llama 3.3 70B",
        "provider": "Meta via Groq",
        "model_string": "llama-3.3-70b-versatile",
        "color": "#00e5ff",
        "icon": "🔮",
        "specialties": ["Reasoning", "Analysis", "Creative", "Code"],
        "cost_per_1k": 0.0,
        "speed_ms": 400,
        "context_window": 128000,
        "description": "Most capable open model on Groq"
    },
    "gpt4": {
        "id": "gpt4",
        "name": "Mixtral 8x7B",
        "provider": "Mistral via Groq",
        "model_string": "mixtral-8x7b-32768",
        "color": "#69ff47",
        "icon": "⚡",
        "specialties": ["Code", "Math", "Instructions"],
        "cost_per_1k": 0.0,
        "speed_ms": 350,
        "context_window": 32768,
        "description": "Mixture-of-experts powerhouse"
    },
    "gemini": {
        "id": "gemini",
        "name": "Gemma2 9B",
        "provider": "Google via Groq",
        "model_string": "gemma2-9b-it",
        "color": "#ff9100",
        "icon": "💎",
        "specialties": ["Factual", "Concise", "Efficient"],
        "cost_per_1k": 0.0,
        "speed_ms": 300,
        "context_window": 8192,
        "description": "Google's Gemma — fast and factual"
    },
    "mistral": {
        "id": "mistral",
        "name": "Llama 3.1 8B",
        "provider": "Meta via Groq",
        "model_string": "llama-3.1-8b-instant",
        "color": "#e040fb",
        "icon": "🌪️",
        "specialties": ["Speed", "Chat", "Efficient"],
        "cost_per_1k": 0.0,
        "speed_ms": 150,
        "context_window": 128000,
        "description": "Fastest model — instant responses"
    },
    "llama": {
        "id": "llama",
        "name": "Llama 3.3 70B",
        "provider": "Meta via Groq",
        "model_string": "llama-3.3-70b-versatile",
        "color": "#ff4081",
        "icon": "🦙",
        "specialties": ["Open Source", "Versatile", "Strong"],
        "cost_per_1k": 0.0,
        "speed_ms": 400,
        "context_window": 128000,
        "description": "Meta's flagship open-source model"
    }
}


class LLMClients:
    def __init__(self):
        self.client = None
        self._init_client()

    def _init_client(self):
        api_key = os.environ.get("GROQ_API_KEY")
        if api_key:
            try:
                self.client = Groq(api_key=api_key)
                print("  ✅ Groq client initialized")
            except Exception as e:
                print(f"  ⚠️  Groq init error: {e}")
        else:
            print("  ⚠️  No GROQ_API_KEY found — using simulated responses")

    def query(self, model_id: str, prompt: str, stream: bool = False):
        start = time.time()
        groq_model = GROQ_MODEL_MAP.get(model_id, "llama-3.3-70b-versatile")

        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=groq_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1024,
                    temperature=0.7
                )
                text = response.choices[0].message.content
                elapsed = (time.time() - start) * 1000
                return text, elapsed
            except Exception as e:
                print(f"  ⚠️  Groq query error ({model_id}): {e}")
                return self._simulate_response(model_id, prompt)
        else:
            return self._simulate_response(model_id, prompt)

    def _stream_model(self, model_id: str, prompt: str):
        """Stream tokens from Groq."""
        groq_model = GROQ_MODEL_MAP.get(model_id, "llama-3.3-70b-versatile")

        if not self.client:
            response, _ = self._simulate_response(model_id, prompt)
            for char in response:
                yield char
                time.sleep(0.008)
            return

        try:
            with self.client.chat.completions.stream(
                model=groq_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.7
            ) as stream:
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
        except Exception as e:
            yield f"\n[Groq streaming error: {str(e)}]"

    # Called by engine.py for the primary model
    def _stream_claude(self, prompt: str):
        return self._stream_model("claude", prompt)

    def _simulate_response(self, model_id: str, prompt: str) -> tuple:
        m = MODEL_REGISTRY.get(model_id, MODEL_REGISTRY["claude"])
        prompt_lower = prompt.lower()

        if any(w in prompt_lower for w in ["code", "function", "python", "def ", "class "]):
            demo = f"""```python
def solution(input_data):
    \"\"\"
    Generated by {m['name']} ({m['provider']})
    Add GROQ_API_KEY to .env for real responses!
    \"\"\"
    result = []
    for item in input_data:
        result.append(item)
    return result

output = solution(["example"])
print(output)
```
*Set GROQ_API_KEY in your .env for live responses.*"""
        else:
            demo = f"""**{m['name']}** ({m['provider']}) — Simulated Response

To enable real AI responses, add your Groq API key to `.env`:
```
GROQ_API_KEY=gsk_your_key_here
```
Get a free key at console.groq.com

Model: {m['context_window']:,} token context · {m['speed_ms']}ms · FREE
Specialties: {', '.join(m['specialties'])}"""

        return demo, None
