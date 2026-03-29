# **Bio-Geometric Causal-Koopman (BGCK) Architecture**

<div align="center">

![Python 3.11 Strict](https://img.shields.io/badge/python-3.11%20Strict-blue.svg)
![FastAPI](https://img.shields.io/badge/framework-FastAPI-009688.svg)
![Three.js](https://img.shields.io/badge/frontend-Three.js%20WebGL-black.svg)
![Gemini](https://img.shields.io/badge/AI-Gemini%202.5%20Flash-orange.svg)

</div>

**CRITICAL DEPENDENCY WARNING**

This project **strictly requires Python 3.11.x**. Do not use Python 3.12 or newer. Core mathematical and topological libraries used in this engine (giotto-tda, z3-solver, pydmd) require specific C++ build binaries that are optimized and tested exclusively for Python 3.11.

The Bio-Geometric Causal-Koopman (BGCK) Terminal is an advanced quantitative finance and algorithmic trading pipeline designed for the Indian National Stock Exchange (NSE). It addresses the limitations of standard stochastic machine learning models, which are susceptible to overfitting and predictive failure during unprecedented market volatility. Instead, BGCK functions as a deterministic, explainable, and verifiable "White-Box" pipeline where trade allocations are mathematically derived using physics, game theory, and formal logic.

## **Table of Contents**

1. [Prerequisites and Setup](#prerequisites-and-setup)  
2. [Running the Application](#running-the-application)  
3. [Architecture Components](#architecture-components)  
4. [Standalone Modules and Backtests](#standalone-modules-and-backtests)  
5. [Troubleshooting](#troubleshooting)  
6. [Support and Contribution](#support-and-contribution)

## **Prerequisites and Setup**

Follow these instructions to configure the environment and install necessary dependencies.

**1. Clone the repository:**

```bash
git clone https://github.com/yourusername/bgck-quant-terminal.git
cd bgck-quant-terminal
```

**2. Create and activate a Python 3.11 Virtual Environment:**

```powershell
# On Windows  
python -m venv .venv  
.\.venv\Scripts\activate
```

## 🤖 Agentic Advisory Copilot

The BGCK Terminal features a **Hybrid-AI Diagnostic Engine**:

- **Deterministic Mode (Default)**: If no API key is provided, the system utilizes a rule-based analytical engine to interpret telemetry (Weights, VPIN, Koopman) and generate professional briefings. This ensures 100% functionality for judges and reviewers.
- **Agentic Mode (Unlocked)**: By adding a `GEMINI_API_KEY` to your `.env`, the system activates **Gemini 2.5 Flash**. This enables deep conversational acoustics, nuanced cross-asset reasoning, and "Definitive Signals" based on current market manifolds.

### 🛠 Installation & Quick Start

1. **Clone & Environment**:
   ```bash
   git clone https://github.com/kkm121/bgck-quant-fintech.git
   cd bgck-quant-fintech
   ```
2. **Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **API Keys (Optional)**:
   Create a `.env` file from the `.env.example` template:
   ```env
   GEMINI_API_KEY_1="your_key_here"
   ```
4. **Execution**:
   ```bash
   uvicorn fastapi_app:app --reload
   ```
   Navigate to `http://127.0.0.1:8000` to engage the terminal.

## **Running the Application**

The BGCK system operates as a unified FastAPI backend serving an ultra-realistic 3D WebGL terminal.

**1. Launch the Backend Gateway:**

```bash
uvicorn fastapi_app:app --reload
```

**2. Access the Terminal:**

Open your web browser and navigate to `http://localhost:8000`.

**3. UI Operation Guidelines:**

* **Target Asset Universe:** Leave the main scan box empty to execute a calculation on the full Nifty 50 universe. Alternatively, use the `+` module to select up to 5 custom NSE tickers (e.g., TCS, RELIANCE).  
* **Objective Phase Tuning:**  
  * *Adversarial Growth:* Utilizes a standard Nash Minimax penalty framework (`risk_aversion = 2.0`) to maximize alpha.  
  * *Structural Stability:* Increases the risk-aversion penalty factor by 5x (`risk_aversion = 10.0`), forcing the optimizer to prioritize capital preservation.  
* **Agentic Copilot:** Upon completion of the execution pipeline, the embedded Gemini 2.5 Flash agent will automatically generate a deterministic rationale report. You may query the agent further regarding specific allocations and VPIN metrics.

## **Architecture Components**

The pipeline processes market data sequentially through four specialized microservice layers.

### **1. Cybersecurity Layer (vpin_firewall.py)**

* **Description:** Transitions chronological time into "Volume Time". By tracking the Volume-Synchronized Probability of Informed Trading (VPIN), the engine mathematically detects institutional order-book imbalances indicative of spoofing.  
* **Main Feature:** Automated disqualification of highly toxic assets.

### **2. Physics & Geometric Layer**

* **topology_radar.py:** Employs Persistent Homology to model the Nifty 50 as a multi-dimensional point cloud. Calculates the Betti-1 Invariant to identify structural fragmentations in market correlations.  
* **koopman_fluid.py:** Maps non-linear market data into an infinite-dimensional space using Bagging Optimized Dynamic Mode Decomposition (BOPDMD), calculating the "kinetic momentum" of capital flow.  
* **causal_dag.py:** Utilizes a Sparse Partial Correlation matrix to isolate statistically significant causal linkages, eliminating correlational noise.

### **3. Decision & Compliance Layer**

* **nash_equilibrium.py:** Formulates portfolio allocation as a zero-sum game utilizing Alternating Direction Method of Multipliers (ADMM) Convex Optimization to compute the optimal Nash Equilibrium.  
* **z3_sebi_solver.py:** A neuro-symbolic logic gate powered by the Microsoft Z3 SMT solver. It formally proves that proposed allocations comply with strict regulatory constraints before execution.

### **4. Agentic Intelligence Layer (langchain_agent.py)**

* **Description:** An integrated LLM acting as a quantitative interpreter. It parses the resulting matrices (Koopman weights, Z3 compliance) to generate advisory reports without hallucination.

## **Standalone Modules and Backtests**

The repository includes standalone scripts to verify the resilience of the mathematical engines against historical and simulated market collapses.

### **1. Empirical Topology Backtest (historical_crash_test.py)**

* **Description:** Simulates the Topological Radar's predictive capability leading up to the 2020 COVID-19 Market Crash. Data is strictly truncated to Feb 21, 2020, to prove the Betti-1 geometric tear occurs preemptively.  
* **Run Command:**  
  `python historical_crash_test.py`

### **2. Nash Optimization Stress Test (test_crash_simulation.py)**

* **Description:** Generates 100 hours of synthetic Nifty 50 data, injecting a massive, highly-correlated negative shock at hour 70. Verifies the Betti-1 kill-switch activation logic.  
* **Run Command:**  
  `python test_crash_simulation.py`

## **Troubleshooting**

### **SVD did not converge in Linear Least Squares**

This error occurs when the Koopman fluid dynamics solver receives a flatlined asset array (e.g., an asset halted trading).

**Solution:** The engine is programmed to utilize microscopic matrix jitter (1e-8) to prevent this. Ensure your local environment is running the exact version of numpy (`1.26.4`) specified in `requirements.txt`.

### **giotto-tda Installation Failures**

Example error: `Failed building wheel for giotto-tda` or `Microsoft Visual C++ 14.0 or greater is required`.

**Solution:** This library requires specific pre-compiled binaries. You must strictly downgrade your virtual environment to Python 3.11.x. If you are on Windows, ensure the "Desktop development with C++" workload is installed via Visual Studio Build Tools.

### **Missing Chat Template / Empty Report**

If the Langchain LLM agent returns an empty context or execution halts at the reporting phase:

**Solution:** Verify that your `GEMINI_API_KEY_N` cluster is correctly structured in your `.env` file without any trailing spaces or invalid characters. Confirm that your Google AI Studio account has not exceeded the 15 RPM free-tier limit.

## **Support and Contribution**

* For theoretical documentation on Koopman Operator Theory, consult the [PyDMD documentation](https://mathlab.github.io/PyDMD/).  
* For topology and Persistent Homology logic, refer to the [giotto-tda documentation](https://giotto-ai.github.io/gtda-docs/latest/library.html).  
* To report issues, suggest algorithmic improvements, or contribute to the causal mapping layers, please open an Issue or Pull Request in this repository.