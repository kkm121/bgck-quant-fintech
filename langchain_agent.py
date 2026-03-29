"""
Microservice: Agentic Advisory Copilot
Architecture Component: LLM Interface
GITHUB RELEASE UPDATE: Hardcoded to Gemini 2.5 Flash for enterprise-grade 
latency. Expanded telemetry payload to include backend execution speeds 
for deeper user interactivity. Added strict Rationale Rules for 0% and Equal Weights.
"""
import json
import numpy as np

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage
except ImportError:
    pass

class LangchainAgent:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self.api_key = api_key
        # Explicitly enforcing Gemini 2.5 Flash for high-speed terminal polling
        self.model = model
        
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=self.model,
                google_api_key=self.api_key,
                temperature=0.2
            )
        except Exception as e:
            self.llm = None
            print(f"Failed to boot LLM: {e}")

    def _clean_numpy(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def _minify_payload(self, payload: dict) -> str:
        """ Aggressively trims the multi-dimensional math telemetry while ensuring full-universe visibility. """
        if not payload:
            return "No valid telemetry available."
            
        # EXPANDED: Now includes the entire scanned universe for comparative analysis (e.g. Apollo vs Asian Paints)
        all_tickers = payload.get("sorted_tickers", [])
        weights = payload.get("koopman_alpha", {}).get("weights", {})
        vpin = payload.get("vpin_toxicity", {})
        returns = payload.get("expected_returns", {})
        times = payload.get("execution_times", {})
        
        # Structure as a compact table-like dictionary for LLM efficiency
        asset_details = {}
        for t in all_tickers:
            asset_details[t] = {
                "W": f"{weights.get(t, 0)*100:.1f}%",
                "Tox": f"{vpin.get(t, 0)*100:.1f}%",
                "E": f"{returns.get(t, 0)*100:.1f}%"
            }
        
        minified = {
            "Systemic_Crash_Imminent": payload.get("topology_risk", {}).get("crash_imminent", False),
            "Market_Betti_1_Risk": payload.get("topology_risk", {}).get("betti_1_l2_derivative", 0.0),
            "Z3_Compliance": payload.get("z3_compliance", {}).get("status", "Unknown"),
            "Full_Scanned_Universe_Telemetry": asset_details,
            "Pipeline_Speeds": {
                "Physics_Engine": f"{times.get('deep_physics_sec', 0)}s",
                "Z3_Solver": f"{times.get('z3_proof_sec', 0)}s"
            },
            "Rationale_Engine": "Assets with 0% weight were disqualified by the Reliability Gate. 'W' = Optimal Weight, 'Tox' = VPIN Institutional Toxicity, 'E' = Expected Return (60D Alpha)."
        }
        
        return json.dumps(minified, default=self._clean_numpy, indent=2)

    def generate_advisory_report(self, payload: dict, mode: str) -> str:
        if not self.api_key or not self.llm:
            return "⚠️ **AI Offline:** Valid Gemini API Key required for automated advisory generation."

        top_tickers = payload.get("sorted_tickers", [])[:5]
        weights = payload.get("koopman_alpha", {}).get("weights", {})
        vpin = payload.get("vpin_toxicity", {})
        
        details = []
        for t in top_tickers:
            w = weights.get(t, 0) * 100
            v = vpin.get(t, 0) * 100
            details.append(f"- {t}: {w:.1f}% Weight (Toxicity: {v:.1f}%)")
        weight_str = "\n".join(details)
        full_count = len(payload.get("sorted_tickers", []))

        prompt = f"""You are the BGCK Quant Copilot powered by Gemini 2.5 Flash. 
The mathematical ADMM Minimax engine has just finished scanning a universe of {full_count} assets.
Mode Selected: {mode.upper()}

Top 5 Recommended Allocations (from {full_count} total assets):
{weight_str}

Write a definitive technical briefing justifying these top selections via KOOPMAN SPECTRAL DISPLACEMENT and NASH ROBUSTNESS. 
If the user asks about an asset NOT in the top 5, you have access to the full universe data.
CRITICAL RULES:
1. Explain 0% weights as Reliability Gate disqualification (Negative Momentum/High VPIN).
2. For each in the Top 5, append a clear 'ACTIONABLE SIGNAL: INVEST'.
Deliver a mathematical verdict."""
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            err = str(e)
            if "API_KEY_INVALID" in err or "API key not valid" in err:
                return "⚠️ **AI Offline:** Gemini API key is invalid."
            return f"⚠️ **AI Briefing Failed:** {err[:120]}"

    def interactive_chat(self, message: str, payload: dict) -> str:
        if not self.api_key or not self.llm:
            return "⚠️ Agent Offline: GEMINI_API_KEY missing or invalid in .env."
            
        slim_context = self._minify_payload(payload)
        portfolio_context = payload.get("portfolio_context", "None specified.")
        
        prompt = f"""You are BGCK, a hyper-intelligent quant advisory agent. 
You now have telemetry for the FULL SCANNED UNIVERSE of {len(payload.get('sorted_tickers', []))} assets.
If the user asks for a comparison between two assets (e.g. Apollo vs Asian Paints), use the 'W' (Weight), 'Tox' (Toxicity), and 'E' (Expected Return) metrics below to provide a definitive answer.

TECHNICAL TELEMETRY:
{slim_context}

User Query: {message}
Deliver a definitive mathematical verdict with 'DEFINITIVE ACTIONABLE SIGNAL' for any specific assets mentioned."""
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            return f"⚠️ Exception in LLM execution: {str(e)[:120]}"