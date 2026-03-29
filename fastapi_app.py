from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List
from main_loop import BGCKOrchestrator
import pandas as pd
import os
from dotenv import load_dotenv

app = FastAPI(title="BGCK Architecture API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

NIFTY50_TICKERS = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BPCL.NS", "BHARTIARTL.NS",
    "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DIVISLAB.NS", "DRREDDY.NS",
    "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS",
    "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "ITC.NS",
    "INDUSINDBK.NS", "INFY.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LTIM.NS",
    "LT.NS", "M&M.NS", "MARUTI.NS", "NTPC.NS", "NESTLEIND.NS", "ONGC.NS",
    "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS", "SUNPHARMA.NS",
    "TCS.NS", "TATACONSUM.NS", "TRENT.NS", "TATASTEEL.NS", "TECHM.NS",
    "TITAN.NS", "SHRIRAMFIN.NS", "ULTRACEMCO.NS", "WIPRO.NS"
]

LATEST_PAYLOAD = {}

class BGCKKeyManager:
    def __init__(self):
        # Mission-Critical: Rely strictly on environment variables for public repo safety
        self.keys = []
        for i in range(1, 6):
            key = os.getenv(f"GEMINI_API_KEY_{i}")
            if key:
                self.keys.append(key)
                
        self.current_idx = 0
        
        # Legacy support
        if not self.keys:
            singleton = os.getenv("GEMINI_API_KEY")
            if singleton:
                self.keys = [singleton]

        if not self.keys:
            # Fallback for legacy singleton key
            legacy_key = os.getenv("GEMINI_API_KEY")
            if legacy_key:
                self.keys = [legacy_key]

    def get_active_key(self) -> str:
        if not self.keys:
            return ""
        return self.keys[self.current_idx % len(self.keys)]

    def rotate(self) -> bool:
        if len(self.keys) <= 1:
            return False
        self.current_idx += 1
        print(f"[BGCK Backend] Rotating API Cluster to Node {self.current_idx % len(self.keys) + 1}")
        return True

key_manager = BGCKKeyManager()

class ExecutionRequest(BaseModel):
    tickers: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    portfolio_context: Optional[str] = None
    recommendation_mode: Optional[str] = "optimal"
    risk_profile: Optional[str] = "aggressive"
    is_initial_run: Optional[bool] = True

class ChatRequest(BaseModel):
    message: str

@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")

@app.post("/api/execute")
async def execute_pipeline(request: ExecutionRequest):
    global LATEST_PAYLOAD
    
    # Retry loop for seamless failover
    max_retries = len(key_manager.keys)
    last_error = ""

    for attempt in range(max_retries):
        try:
            api_key = key_manager.get_active_key()
            if not api_key:
                raise ValueError("GEMINI_API_KEY Cluster is missing from your .env file.")

            tickers_list = NIFTY50_TICKERS if request.tickers == "NIFTY50_UNIVERSE" else [t.strip() for t in request.tickers.split(",") if t.strip()]

            orchestrator = BGCKOrchestrator(
                tickers=tickers_list, 
                api_key=api_key,
                portfolio_context=request.portfolio_context,
                recommendation_mode=request.recommendation_mode,
                risk_profile=request.risk_profile,
                is_initial_run=request.is_initial_run
            )
            
            result = await orchestrator.execute_pipeline()
            
            if request.tickers == "NIFTY50_UNIVERSE" and "sorted_tickers" in result and len(result["sorted_tickers"]) < 20 and LATEST_PAYLOAD:
                return LATEST_PAYLOAD
                
            LATEST_PAYLOAD = result
            return result

        except Exception as e:
            last_error = str(e)
            if "quota" in last_error.lower() or "limit" in last_error.lower() or "api key" in last_error.lower():
                if key_manager.rotate():
                    continue  # Try next key
            raise HTTPException(status_code=500, detail=last_error)

    raise HTTPException(status_code=500, detail=f"API Cluster Exhausted: {last_error}")

@app.get("/api/chart/{ticker}")
async def get_specific_chart(ticker: str):
    try:
        from nse_fetcher import NSEDataFetcher
        formatted_ticker = f"{ticker}.NS" if not ticker.endswith(".NS") else ticker
        fetcher = NSEDataFetcher(tickers=[formatted_ticker], interval="1h")
        raw_data = fetcher.fetch_data()
        df = raw_data[formatted_ticker].ffill().bfill()
        
        chart_data = []
        seen_timestamps = set()
        for idx, row in df.iterrows():
            if pd.notna(row.get('Close')):
                unix_time = int(pd.Timestamp(idx).timestamp())
                if unix_time not in seen_timestamps:
                    seen_timestamps.add(unix_time)
                    chart_data.append({
                        "time": unix_time,
                        "open": float(row.get('Open', row.get('Close'))),
                        "high": float(row.get('High', row.get('Close'))),
                        "low": float(row.get('Low', row.get('Close'))),
                        "close": float(row.get('Close')),
                        "value": float(row.get('Volume', 0))
                    })
        chart_data.sort(key=lambda x: x['time'])
        return {"ticker": ticker, "data": chart_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chart Data Error: {str(e)}")

@app.post("/api/chat")
async def interactive_chat(request: ChatRequest):
    max_retries = len(key_manager.keys)
    last_error = ""

    for attempt in range(max_retries):
        try:
            api_key = key_manager.get_active_key()
            if not api_key:
                return {"response": "⚠️ **SYSTEM WARNING:** The GEMINI_API_KEY cluster was not detected!"}

            from langchain_agent import LangchainAgent
            agent = LangchainAgent(api_key=api_key)
            response = agent.interactive_chat(request.message, LATEST_PAYLOAD)
            return {"response": response}

        except Exception as e:
            last_error = str(e)
            if "quota" in last_error.lower() or "limit" in last_error.lower() or "api key" in last_error.lower():
                if key_manager.rotate():
                    continue
            raise HTTPException(status_code=500, detail=last_error)

    raise HTTPException(status_code=500, detail=f"API Cluster Exhausted: {last_error}")