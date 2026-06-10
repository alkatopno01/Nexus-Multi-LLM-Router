# NEXUS — Multi-LLM Adaptive Router

A smart multi-LLM routing system that automatically sends your query
to the best AI model based on intent, complexity, and routing mode.

## Quick Start (3 steps)

1. Get a FREE Groq API key at https://console.groq.com
2. Rename `.env.example` to `.env` and paste your key
3. Run the setup commands below

## Setup Commands

```bash
# Windows
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Then open: http://localhost:5000

## Features

- Login / Signup system
- 5 AI models via Groq (all FREE)
- 4 routing modes: Smart, Speed, Budget, Arena
- Real-time streaming responses
- Arena mode: compare all models side by side
- Query history in sidebar
- Intent detection: CODE, MATH, CREATIVE, ANALYTICAL, FACTUAL

## Models Used (all free via Groq)

| Slot      | Model               |
|-----------|---------------------|
| L3        | Llama 3.3 70B       |
| Mx        | Mixtral 8x7B        |
| G2        | Gemma2 9B           |
| L1        | Llama 3.1 8B Instant|
| Lm        | Llama 3.3 70B       |

## Project Structure

```
nexus/
├── app.py                  Flask app + auth routes
├── router/
│   ├── classifier.py       Intent detection
│   ├── engine.py           Routing scoring engine
│   └── llm_clients.py      Groq API integration
├── templates/
│   ├── auth.html           Login / Signup UI
│   └── index.html          Main chat UI
├── .env.example            API key template
├── requirements.txt        Dependencies
└── README.md               This file
```

## Minor Project Info

Built for B.Tech CSE Minor Project
Tech stack: Python, Flask, Groq API, Vanilla JS, Plus Jakarta Sans
