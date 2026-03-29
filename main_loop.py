"""
Microservice: Asynchronous BGCK Orchestrator
Architecture Component: Core Execution Loop
PERFORMANCE UPDATE: Added granular sequential telemetry.
BUG FIX: Properly pipes optimal weights and VPIN toxicity into the LLM context payload 
so the Copilot can provide elaborate, data-driven reasoning. Also added is_initial_run gating.
"""
import asyncio
import pandas as pd
import numpy as np
import datetime
import time
from concurrent.futures import ThreadPoolExecutor

from nse_fetcher import NSEDataFetcher
from vpin_firewall import VPINFirewall
from topology_radar import TopologyRadar
from koopman_fluid import KoopmanFluidDynamics
from causal_dag import CausalDAG
from nash_equilibrium import NashRobustOptimizer
from z3_sebi_solver import Z3SEBISolver
from langchain_agent import LangchainAgent

class BGCKOrchestrator:
    def __init__(self, tickers: list, api_key: str, portfolio_context: str = None, recommendation_mode: str = "optimal", risk_profile: str = "aggressive", is_initial_run: bool = True):
        self.raw_tickers = tickers
        self.portfolio_context = portfolio_context if portfolio_context else "Standard Risk. No specific holdings."
        self.recommendation_mode = recommendation_mode
        self.risk_profile = risk_profile
        self.is_initial_run = is_initial_run
        self.api_key = api_key
        
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=59)
        self.start_date = start.strftime('%Y-%m-%d')
        self.end_date = end.strftime('%Y-%m-%d')
        self.interval = "1h"
        
        self.vpin = VPINFirewall()
        self.topology = TopologyRadar()
        self.koopman = KoopmanFluidDynamics()
        self.causal = CausalDAG()
        self.nash = NashRobustOptimizer()
        self.langchain = LangchainAgent(api_key=api_key)

    async def execute_pipeline(self) -> dict:
        loop = asyncio.get_running_loop()
        execution_times = {}
        t_global_start = time.perf_counter()
        
        print(f"\n{'='*60}\n[BGCK Telemetry] INITIATING PHYSICS PIPELINE...\n{'='*60}")
        
        # Step 1: Data Fetching
        t_start_ingestion = time.perf_counter()
        fetcher = NSEDataFetcher(tickers=self.raw_tickers, start_date=self.start_date, end_date=self.end_date, interval=self.interval)
        raw_universe_data = await asyncio.to_thread(fetcher.fetch_data)
        
        close_universe = fetcher.extract_close_prices(raw_universe_data).ffill().bfill()
        volume_universe = fetcher.extract_volume(raw_universe_data)
        valid_tickers = [t for t in close_universe.columns if t in volume_universe.columns]
        
        log_returns = np.log(close_universe[valid_tickers] / close_universe[valid_tickers].shift(1)).fillna(0)
        execution_times["data_ingestion_sec"] = round(time.perf_counter() - t_start_ingestion, 3)
        print(f"[BGCK Telemetry] [{time.perf_counter() - t_global_start:05.2f}s] Data Ingestion Complete.")

        # Step 2: Parallel Physics
        t_start_physics = time.perf_counter()
        with ThreadPoolExecutor(max_workers=52) as executor_pool:
            vpin_futures = [loop.run_in_executor(executor_pool, self.vpin.execute_filter, close_universe[[t]], volume_universe[[t]]) for t in valid_tickers]
            topo_task = loop.run_in_executor(executor_pool, self.topology.generate_crash_signal, log_returns)
            koop_task = loop.run_in_executor(executor_pool, self.koopman.execute_fluid_mapping, log_returns)
            
            all_vpin_results = await asyncio.gather(*vpin_futures)
            topology_results = await topo_task
            fluid_results = await koop_task
            
            vpin_results = {}
            for res in all_vpin_results: vpin_results.update(res)
                
        execution_times["deep_physics_sec"] = round(time.perf_counter() - t_start_physics, 3)
        print(f"[BGCK Telemetry] [{time.perf_counter() - t_global_start:05.2f}s] VPIN & Koopman Resolved.")
            
        # Step 3: Sequential Causal & Nash
        t_start_matrices = time.perf_counter()
        print(f"[BGCK Telemetry] [{time.perf_counter() - t_global_start:05.2f}s] Running Fast-Path Causal Rectification...")
        
        causal_cov = await asyncio.to_thread(self.causal.execute_causal_rectification, log_returns, fluid_results["covariance_matrix"])
        
        vpin_toxicity = {k: float(v['toxicity_zscore']) for k, v in vpin_results.items()}
        
        portfolio_results = await asyncio.to_thread(
            self.nash.execute_robust_allocation, 
            fluid_results["expected_returns"], 
            causal_cov, 
            valid_tickers, 
            vpin_toxicity,
            self.risk_profile
        )
        execution_times["causal_and_nash_sec"] = round(time.perf_counter() - t_start_matrices, 3)
        print(f"[BGCK Telemetry] [{time.perf_counter() - t_global_start:05.2f}s] Nash Equilibrium Optimization Finished.")
        
        # Step 4: Z3 Proof
        t_start_z3 = time.perf_counter()
        z3_solver = Z3SEBISolver(tickers=valid_tickers)
        z3_verification = await asyncio.to_thread(z3_solver.verify_execution_payload, portfolio_results["weights"], 2.0, topology_results["betti_1_l2_derivative"], 1.5)
        execution_times["z3_proof_sec"] = round(time.perf_counter() - t_start_z3, 3)
        
        # Step 5: LLM (Only executes an API call on the very first execution to save tokens)
        t_start_llm = time.perf_counter()
        if self.is_initial_run:
            # FIX: Send actual payload so the LLM can see the weights and toxicity!
            temp_payload = {
                "vpin_toxicity": vpin_toxicity,
                "koopman_alpha": {"weights": portfolio_results["weights"]},
                "sorted_tickers": sorted(valid_tickers, key=lambda t: -portfolio_results["weights"].get(t, 0))
            }
            advisory_report = self.langchain.generate_advisory_report(temp_payload, self.recommendation_mode)
        else:
            advisory_report = "Telemetry successfully updated. Agent on Standby."
            
        execution_times["llm_agent_sec"] = round(time.perf_counter() - t_start_llm, 3)
        
        final_payload = {
            "expected_returns": {k: float(v) for k, v in zip(valid_tickers, fluid_results["expected_returns"])},
            "vpin_toxicity": vpin_toxicity,
            "topology_risk": topology_results,
            "koopman_alpha": {
                "weights": portfolio_results["weights"], 
                "fallback_triggered": portfolio_results.get("fallback_triggered", False),
                "failsafe_reason": portfolio_results.get("failsafe_reason", "")
            },
            "z3_compliance": z3_verification,
            "top_ticker": sorted(valid_tickers, key=lambda t: -portfolio_results["weights"].get(t, 0))[0].replace('.NS', ''),
            "sorted_tickers": sorted(valid_tickers, key=lambda t: -portfolio_results["weights"].get(t, 0)),
            "execution_times": execution_times,
            "advisory_report": advisory_report,
            "recommendation_mode": self.recommendation_mode
        }
        
        print(f"[BGCK Telemetry] [{time.perf_counter() - t_global_start:05.2f}s] PIPELINE COMPLETE. Syncing UI.\n{'='*60}\n")
        return final_payload