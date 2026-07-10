import re
import math
from collections import Counter

class QueryClassifier:
    def __init__(self):
        self.intent_patterns = {
            "CODE": {
                "keywords": ["code", "function", "debug", "error", "python", "javascript", "java", "c++",
                             "sql", "api", "bug", "fix", "implement", "class", "algorithm", "syntax",
                             "compile", "runtime", "exception", "variable", "loop", "array", "list",
                             "dict", "object", "method", "library", "framework", "script", "program",
                             "html", "css", "react", "flask", "django", "nodejs", "typescript"],
                "patterns": [r"```", r"def\s+\w+", r"import\s+\w+", r"console\.log",
                              r"print\(", r"#include", r"<[a-z]+>", r"\(\)\s*{"],
                "weight": 1.5
            },
            "MATH": {
                "keywords": ["calculate", "solve", "equation", "formula", "integral", "derivative",
                             "matrix", "vector", "probability", "statistics", "compute", "sum",
                             "average", "mean", "variance", "polynomial", "algebra", "geometry",
                             "theorem", "proof", "graph", "topology", "calculus", "differential"],
                "patterns": [r"\d+\s*[\+\-\*/\^]\s*\d+", r"=\s*\d", r"\d+%",
                              r"sqrt|log|sin|cos|tan|exp"],
                "weight": 1.4
            },
            "CREATIVE": {
                "keywords": ["write", "story", "poem", "creative", "imagine", "fiction", "narrative",
                             "character", "plot", "dialogue", "essay", "blog", "content", "caption",
                             "script", "lyrics", "metaphor", "describe", "brainstorm", "idea",
                             "design", "concept", "innovative", "artistic", "fantasy", "novel"],
                "patterns": [r"write\s+a", r"create\s+a\s+story", r"imagine\s+if", r"once\s+upon"],
                "weight": 1.3
            },
            "ANALYTICAL": {
                "keywords": ["analyze", "analyse", "compare", "evaluate", "assess", "explain why",
                             "pros and cons", "advantages", "disadvantages", "impact", "effect",
                             "cause", "reason", "opinion", "think", "believe", "argue", "debate",
                             "critique", "review", "research", "study", "investigate", "hypothesis",
                             "conclusion", "inference", "implication", "strategy", "decision"],
                "patterns": [r"why\s+is", r"how\s+does", r"what\s+are\s+the", r"explain\s+the",
                              r"analyze", r"compare\s+and"],
                "weight": 1.2
            },
            "FACTUAL": {
                "keywords": ["what is", "who is", "when did", "where is", "history", "definition",
                             "meaning", "fact", "information", "tell me about", "describe",
                             "overview", "summary", "wikipedia", "news", "current", "latest",
                             "update", "recent", "today", "2024", "2025", "capital", "population"],
                "patterns": [r"what\s+is\s+the", r"who\s+(is|was|are)", r"when\s+(did|was|is)",
                              r"where\s+is", r"how\s+many"],
                "weight": 1.1
            },
            "CONVERSATIONAL": {
                "keywords": ["hello", "hi", "hey", "how are you", "thanks", "thank you",
                             "what do you think", "help me", "can you", "could you", "please",
                             "opinion", "suggest", "recommend", "chat", "talk", "discuss"],
                "patterns": [r"^(hi|hello|hey)\b", r"how\s+are\s+you", r"thank\s+you"],
                "weight": 0.9
            }
        }

    def classify(self, query: str) -> dict:
        query_lower = query.lower()
        words = query_lower.split()
        
        scores = {}
        features = {}
        
        for intent, config in self.intent_patterns.items():
            kw_score = 0
            pat_score = 0
            matched_keywords = []
            matched_patterns = []
            
            for kw in config["keywords"]:
                if kw in query_lower:
                    kw_score += 1
                    matched_keywords.append(kw)
            
            for pattern in config["patterns"]:
                if re.search(pattern, query, re.IGNORECASE):
                    pat_score += 2
                    matched_patterns.append(pattern)
            
            raw = (kw_score + pat_score) * config["weight"]
            scores[intent] = raw
            features[intent] = {
                "keywords": matched_keywords[:5],
                "patterns": len(matched_patterns),
                "raw_score": raw
            }
        
        # Normalize scores to confidence percentages
        total = sum(scores.values()) or 1
        confidences = {k: round((v / total) * 100, 1) for k, v in scores.items()}
        
        primary = max(scores, key=scores.get)
        if scores[primary] == 0:
            primary = "CONVERSATIONAL"
            confidences["CONVERSATIONAL"] = 100.0
        
        # Query complexity score (0-100)
        complexity = self._compute_complexity(query)
        
        return {
            "primary": primary,
            "confidences": confidences,
            "features": features,
            "complexity": complexity,
            "query_length": len(words),
            "has_code": bool(re.search(r'```|def |import |function ', query)),
            "has_numbers": bool(re.search(r'\d+', query)),
            "sentiment": self._basic_sentiment(query_lower)
        }

    def _compute_complexity(self, query: str) -> int:
        """Score query complexity 0-100."""
        factors = 0
        words = query.split()
        
        # Length factor
        factors += min(len(words) / 2, 20)
        
        # Technical vocabulary
        tech_words = sum(1 for w in words if len(w) > 8)
        factors += min(tech_words * 3, 25)
        
        # Question depth
        if query.count('?') > 1:
            factors += 10
        
        # Code presence
        if re.search(r'```|def |import |class ', query):
            factors += 25
            
        # Multi-part query
        if any(w in query.lower() for w in ['and', 'also', 'additionally', 'furthermore', 'compare']):
            factors += 10
            
        return min(int(factors), 100)

    def _basic_sentiment(self, query: str) -> str:
        positive = ['great', 'good', 'awesome', 'love', 'help', 'please', 'thanks']
        negative = ['error', 'bug', 'wrong', 'broken', 'fix', 'problem', 'issue', 'fail']
        
        pos = sum(1 for w in positive if w in query)
        neg = sum(1 for w in negative if w in query)
        
        if pos > neg:
            return "positive"
        elif neg > pos:
            return "negative"
        return "neutral"
