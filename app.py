from flask import Flask, render_template, request, jsonify, Response, stream_with_context, session, redirect, url_for
from router.classifier import QueryClassifier
from router.engine import RoutingEngine
from router.llm_clients import MODEL_REGISTRY
import json, os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "alka2026")

classifier = QueryClassifier()
engine = RoutingEngine()

# Simple in-memory user store (demo project)
users = {}   # email → {name, password}

# ─── Auth Routes ──────────────────────────────────────────────────

@app.route("/")
def root():
    if "user" in session:
        return redirect("/chat")
    return redirect("/login")

@app.route("/login")
def login_page():
    if "user" in session:
        return redirect("/chat")
    return render_template("auth.html")

@app.route("/auth/login", methods=["POST"])
def do_login():
    data = request.get_json()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"ok": False, "error": "Email and password are required."})

    # Demo: accept any credentials; if email exists check password
    if email in users:
        if users[email]["password"] != password:
            return jsonify({"ok": False, "error": "Incorrect password."})
    else:
        # Auto-register on first login (demo mode)
        name = email.split("@")[0].capitalize()
        users[email] = {"name": name, "password": password}

    session["user"] = {"email": email, "name": users[email]["name"]}
    return jsonify({"ok": True})

@app.route("/auth/signup", methods=["POST"])
def do_signup():
    data = request.get_json()
    first = (data.get("first") or "").strip()
    last  = (data.get("last") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not first or not email or len(password) < 6:
        return jsonify({"ok": False, "error": "Please fill in all fields correctly."})

    if email in users:
        return jsonify({"ok": False, "error": "An account with this email already exists."})

    name = f"{first} {last}".strip()
    users[email] = {"name": name, "password": password}
    session["user"] = {"email": email, "name": name}
    return jsonify({"ok": True})

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

# ─── Chat Route ───────────────────────────────────────────────────

@app.route("/chat")
def chat():
    if "user" not in session:
        return redirect("/login")
    user = session["user"]
    models_data = {k: {
        "name": v["name"], "provider": v["provider"],
        "color": v["color"], "icon": v["icon"],
        "specialties": v["specialties"], "description": v["description"],
        "cost_per_1k": v["cost_per_1k"], "speed_ms": v["speed_ms"],
        "context_window": v["context_window"]
    } for k, v in MODEL_REGISTRY.items()}
    return render_template("index.html", models=models_data, user=user)

# ─── API Routes ───────────────────────────────────────────────────

@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json()
    query = data.get("query", "").strip()
    mode  = data.get("mode", "smart")
    if not query:
        return jsonify({"error": "Empty query"}), 400
    intent  = classifier.classify(query)
    routing = engine.route(query, intent, mode)
    scores  = engine.get_scores(intent)
    return jsonify({"intent": intent, "routing": routing, "scores": scores, "mode": mode})

@app.route("/api/stream", methods=["POST"])
def stream():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    data     = request.get_json()
    query    = data.get("query", "").strip()
    model_id = data.get("model_id", "claude")
    if not query:
        return jsonify({"error": "Empty query"}), 400

    def generate():
        yield f"data: {json.dumps({'type':'start','model':model_id})}\n\n"
        tokens = 0
        try:
            for chunk in engine.stream_response(query, model_id):
                tokens += len(chunk.split())
                yield f"data: {json.dumps({'type':'chunk','content':chunk,'tokens':tokens})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type':'error','message':str(e)})}\n\n"
        yield f"data: {json.dumps({'type':'done','total_tokens':tokens})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@app.route("/api/arena", methods=["POST"])
def arena():
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    data      = request.get_json()
    query     = data.get("query", "").strip()
    model_ids = data.get("models", list(MODEL_REGISTRY.keys()))
    if not query:
        return jsonify({"error": "Empty query"}), 400
    return jsonify(engine.arena_query(query, model_ids))

@app.route("/api/stats")
def stats():
    return jsonify(engine.get_system_stats())

# ─── Run ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "═"*48)
    print("  NEXUS  Multi-LLM Router  v1.0")
    print("═"*48)
    print("  🌐  http://localhost:5000")
    print("  🔐  Login/Signup enabled")
    print("  🤖  Models: 5 via Groq")
    print("═"*48 + "\n")
    app.run(debug=True, port=5000, threaded=True)
