import time
import threading
from router.llm_clients import LLMClients, MODEL_REGISTRY

# Capability matrix: model → intent → base_score (0-1)
CAPABILITY_MATRIX = {
    "claude":  {"CODE": 0.90, "MATH": 0.85, "CREATIVE": 0.97, "ANALYTICAL": 0.96, "FACTUAL": 0.84, "CONVERSATIONAL": 0.88},
    "gpt4":    {"CODE": 0.96, "MATH": 0.91, "CREATIVE": 0.85, "ANALYTICAL": 0.88, "FACTUAL": 0.85, "CONVERSATIONAL": 0.86},
    "gemini":  {"CODE": 0.78, "MATH": 0.93, "CREATIVE": 0.80, "ANALYTICAL": 0.82, "FACTUAL": 0.94, "CONVERSATIONAL": 0.83},
    "mistral": {"CODE": 0.72, "MATH": 0.70, "CREATIVE": 0.75, "ANALYTICAL": 0.73, "FACTUAL": 0.78, "CONVERSATIONAL": 0.92},
    "llama":   {"CODE": 0.68, "MATH": 0.65, "CREATIVE": 0.72, "ANALYTICAL": 0.70, "FACTUAL": 0.72, "CONVERSATIONAL": 0.85},
}

# Speed scores (inverted latency, higher = faster)
SPEED_SCORES = {"claude": 0.70, "gpt4": 0.75, "gemini": 0.80, "mistral": 0.95, "llama": 0.90}

# Cost efficiency (inverted cost, higher = cheaper)
COST_SCORES = {"claude": 0.60, "gpt4": 0.40, "gemini": 0.70, "mistral": 0.85, "llama": 0.90}

# Routing mode weights: [capability, speed, cost]
MODE_WEIGHTS = {
    "smart":  [0.70, 0.20, 0.10],
    "speed":  [0.40, 0.50, 0.10],
    "budget": [0.40, 0.20, 0.40],
    "arena":  [0.70, 0.20, 0.10],  # same as smart, but queries all
}


class RoutingEngine:
    def __init__(self):
        self.clients = LLMClients()
        self.query_history = []
        self.model_stats = {m: {"queries": 0, "total_latency": 0, "errors": 0} for m in MODEL_REGISTRY}

    def route(self, query: str, intent: dict, mode: str = "smart") -> dict:
        """Compute routing decision with full score breakdown."""
        primary_intent = intent["primary"]
        complexity = intent["complexity"]
        weights = MODE_WEIGHTS.get(mode, MODE_WEIGHTS["smart"])

        scores = {}
        breakdown = {}

        for model_id in MODEL_REGISTRY:
            cap = CAPABILITY_MATRIX[model_id][primary_intent]
            spd = SPEED_SCORES[model_id]
            cst = COST_SCORES[model_id]

            # Complexity adjustment: high complexity slightly favors Claude/GPT4
            if complexity > 70:
                if model_id in ["claude", "gpt4"]:
                    cap = min(cap * 1.05, 1.0)
                elif model_id in ["mistral", "llama"]:
                    cap = cap * 0.95

            # Weighted composite score
            composite = (
                weights[0] * cap +
                weights[1] * spd +
                weights[2] * cst
            )

            scores[model_id] = round(composite * 100, 1)
            breakdown[model_id] = {
                "capability": round(cap * 100, 1),
                "speed": round(spd * 100, 1),
                "cost_efficiency": round(cst * 100, 1),
                "composite": round(composite * 100, 1)
            }

        winner = max(scores, key=scores.get)
        runner_up = sorted(scores, key=scores.get, reverse=True)[1]

        return {
            "winner": winner,
            "runner_up": runner_up,
            "scores": scores,
            "breakdown": breakdown,
            "mode": mode,
            "reasoning": self._generate_reasoning(winner, primary_intent, breakdown[winner], complexity)
        }

    def _generate_reasoning(self, model: str, intent: str, breakdown: dict, complexity: int) -> str:
        model_info = MODEL_REGISTRY[model]
        reasons = []

        if breakdown["capability"] >= 90:
            reasons.append(f"top-tier {intent.lower()} capability ({breakdown['capability']}%)")
        elif breakdown["capability"] >= 80:
            reasons.append(f"strong {intent.lower()} capability ({breakdown['capability']}%)")

        if breakdown["speed"] >= 85:
            reasons.append(f"ultra-fast response time")

        if breakdown["cost_efficiency"] >= 80:
            reasons.append(f"excellent cost efficiency")

        if complexity > 70:
            reasons.append(f"handles high complexity queries ({complexity}/100)")

        reason_str = ", ".join(reasons[:2]) if reasons else "best overall match"
        return f"{model_info['name']} selected for {reason_str}. Specializes in {', '.join(model_info['specialties'][:2])}."

    def get_scores(self, intent: dict) -> dict:
        """Return full scoring table for visualization."""
        primary = intent["primary"]
        result = {}
        for model_id in MODEL_REGISTRY:
            result[model_id] = {
                "capability": round(CAPABILITY_MATRIX[model_id][primary] * 100, 1),
                "speed": round(SPEED_SCORES[model_id] * 100, 1),
                "cost": round(COST_SCORES[model_id] * 100, 1),
                **MODEL_REGISTRY[model_id]
            }
        return result

    def stream_response(self, query: str, model_id: str):
        """Stream response from a model."""
        start = time.time()
        try:
            for chunk in self.clients._stream_claude(query) if model_id == "claude" else self._mock_stream(model_id, query):
                yield chunk

            elapsed = (time.time() - start) * 1000
            self.model_stats[model_id]["queries"] += 1
            self.model_stats[model_id]["total_latency"] += elapsed
        except Exception as e:
            yield f"\n[Error: {str(e)}]"

    def _mock_stream(self, model_id: str, query: str):
        """Simulate streaming for non-Claude models."""
        response, _ = self.clients._simulate_response(model_id, query)
        for char in response:
            yield char
            time.sleep(0.01)

    def arena_query(self, query: str, model_ids: list) -> dict:
        """Query multiple models in parallel."""
        results = {}
        errors = {}
        threads = []
        lock = threading.Lock()

        def fetch(model_id):
            start = time.time()
            try:
                response, _ = self.clients.query(model_id, query)
                elapsed = round((time.time() - start) * 1000, 1)
                with lock:
                    results[model_id] = {
                        "response": response,
                        "latency_ms": elapsed,
                        "model": MODEL_REGISTRY[model_id]
                    }
            except Exception as e:
                with lock:
                    errors[model_id] = str(e)

        for model_id in model_ids:
            t = threading.Thread(target=fetch, args=(model_id,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=30)

        return {"responses": results, "errors": errors}

    def get_system_stats(self) -> dict:
        """Return aggregate system performance stats."""
        total_queries = sum(s["queries"] for s in self.model_stats.values())
        stats = {}
        for model_id, data in self.model_stats.items():
            avg_latency = (data["total_latency"] / data["queries"]) if data["queries"] > 0 else MODEL_REGISTRY[model_id]["speed_ms"]
            stats[model_id] = {
                "queries": data["queries"],
                "avg_latency_ms": round(avg_latency, 1),
                "error_rate": round(data["errors"] / max(data["queries"], 1) * 100, 1)
            }
        return {"total_queries": total_queries, "models": stats}
