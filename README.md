# Hiver Support Agent - AmazonHelp AI Support System

A production-grade AI Customer Support Agent tailored for **AmazonHelp** on Twitter, developed for the Hiver SDE Intern Take-Home Assignment.

The system triages incoming customer tweets across **8 support intents**, retrieves relevant historical resolutions using **Sentence-Transformers & FAISS**, generates Amazon-style responses with **Claude API**, and enforces a **two-tier safety escalation gate** (`AUTO_HANDLE` vs `ESCALATE_TO_HUMAN`).

---

## Project Structure

```
hiver-support-agent/
│
├── data/
│   ├── amazon_help_processed.csv    # Preprocessed AmazonHelp dataset
│   ├── intent_model.joblib          # Persisted TF-IDF + Logistic Regression model
│   ├── faiss_index.bin              # Cached FAISS dense vector index
│   └── corpus_reference.parquet     # Cached reference text metadata
│
├── src/
│   ├── __init__.py                  # Package init
│   ├── preprocess.py                # STEP 1: Kaggle filter, clean, pair extraction
│   ├── intent_classifier.py         # STEP 2 & 5: TF-IDF, Logistic Regression & Baselines
│   ├── retriever.py                 # STEP 3: Sentence-Transformers + FAISS top-5 RAG
│   ├── generator.py                 # STEP 3 & 7: Claude API generator & LLM Judge
│   ├── escalation.py                # STEP 4: Rule + confidence escalation engine
│   ├── golden_dataset_gen.py        # STEP 6: 200-sample balanced review set
│   ├── evaluator.py                 # STEP 7: Intent & LLM judge evaluation harness
│   ├── failure_analysis.py          # STEP 8: Automated failure category detection
│   └── report_generator.py          # STEP 9: Dynamic report.md generator
│
├── reports/
│   ├── report.md                    # STEP 9: Full technical engineering report
│   ├── decision_log.md              # STEP 10: 15 documented engineering decisions
│   ├── evaluation_summary.json      # Structured benchmark metrics
│   └── failure_analysis.json        # STEP 8: Detected failure records with diagnostics
│
├── golden_dataset/
│   └── golden_dataset.csv           # STEP 6: 200 review samples (tweet, intent, human_reply, escalation)
│
├── tests/
│   └── test_agent.py                # Unit test suite verifying core components
│
├── main.py                          # Unified CLI entry point
├── requirements.txt                 # Project dependencies
├── .env.example                     # Environment template
├── final_report.md                  # Comprehensive technical engineering report
├── decision_log.md                  # Root copy of 15 engineering decisions
└── README.md                        # Documentation
```

---

## 1. Installation & Environment Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Git

### Clone and Setup Virtual Environment
```bash
# Clone the repository
git clone <repo-url>
cd hiver-support-agent

# Create and activate virtual environment
python -m venv venv

# On Linux/macOS:
source venv/bin/activate
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 2. Configuration & Claude API Setup

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Edit `.env` to configure your Anthropic API Key:
```ini
# Anthropic API Key for Claude reply generation and LLM-as-a-judge evaluation
ANTHROPIC_API_KEY=sk-ant-api03-...

# Model selection (e.g. claude-3-haiku-20240307, claude-3-5-sonnet-20241022)
ANTHROPIC_MODEL=claude-3-haiku-20240307

# Escalation confidence threshold
CONFIDENCE_THRESHOLD=0.55
```

> **Offline / Demo Mode**: If `ANTHROPIC_API_KEY` is not provided or left blank, the agent automatically activates an authentic **Amazon-style deterministic simulation engine** and automated heuristic judge. The entire pipeline, test suite, and evaluation harness can run 100% offline out of the box without crashing.

---

## 3. Dataset Configuration (Kaggle twcs.csv)

The system is configured to ingest the Kaggle **Customer Support on Twitter** dataset:
1. Download `twcs.csv` from [Kaggle Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter).
2. Place `twcs.csv` into the `data/` folder:
   ```
   data/twcs.csv
   ```
3. **Automatic Fallback Bootstrap**: If `twcs.csv` is not present, `src/preprocess.py` automatically initializes an authentic bootstrap corpus covering all 8 intents and AmazonHelp conversational structures (`amzn.to/help`, `^AB` signatures) so you can test all 12 steps immediately.

---

## 4. Running the System (CLI Modes)

The system provides a unified CLI via `main.py`:

### Full Pipeline (End-to-End)
Runs preprocessing, trains models, evaluates baselines, generates the golden dataset, runs the LLM judge, and produces `reports/report.md`:
```bash
python main.py --mode full
```

### Step-by-Step CLI Modes

#### Step 1: Preprocess Dataset
```bash
python main.py --mode preprocess
```
Filters for `AmazonHelp`, links inbound customer tweets to agent replies, strips handles, and saves `data/amazon_help_processed.csv`.

#### Step 2 & 5: Train Intent Classifier & Compare Baselines
```bash
python main.py --mode train
```
Trains TF-IDF + Logistic Regression on the 8 intents and compares against:
- **Baseline 1:** Majority Intent Classifier
- **Baseline 2:** Keyword Rule-Based Classifier
Outputs Accuracy, Precision, Recall, and F1.

#### Step 6: Generate Golden Dataset
```bash
python main.py --mode golden
```
Exports 200 stratified review samples to `golden_dataset/golden_dataset.csv` with columns `tweet`, `intent`, `human_reply`, and `escalation`.

#### Step 7 & 8: Run Evaluation Harness & Failure Analysis
```bash
python main.py --mode evaluate
```
Runs the golden dataset through the pipeline, scores replies using Claude-as-a-Judge (1-5 for Correctness, Relevance, Tone, Helpfulness), and saves failure cases to `reports/failure_analysis.json`.

#### Interactive Single Query Inference
Test any customer tweet in real-time:
```bash
python main.py --mode query --query "My package was marked as delivered yesterday but I never got it. Please refund my money!"
```

**Output Example:**
```
============================================================
           AMAZONHELP AI SUPPORT AGENT - INFERENCE
============================================================
CUSTOMER QUERY:
  My package was marked as delivered yesterday but I never got it. Please refund my money!

CLASSIFIED INTENT:  Delivery Problem  (Confidence: 89.24%)

ESCALATION DECISION:
{
  "decision": "AUTO_HANDLE",
  "reason": "Standard support inquiry within operational bounds for intent 'Delivery Problem' (confidence: 0.89)."
}

TOP-3 RETRIEVED HISTORICAL EXAMPLES:
  [1] (Sim: 0.86, Intent: Delivery Problem)
      Cust: App says delivered to front porch at 2 PM, but I was home and nothing was deliver...
      Agent: Sometimes carriers mark items delivered prematurely. Please check around your por...

GENERATED AMAZONHELP RESPONSE:
  "We apologize for the delivery experience! Sometimes carriers mark packages delivered prematurely. Please check around your property or with neighbors. If it still hasn't arrived, please send us a DM with your order ID at amzn.to/help so we can look into this for you. ^AB"
============================================================
```

Test an Escalation Trigger (e.g. Legal Threat):
```bash
python main.py --mode query --query "My lawyer is filing a lawsuit in consumer court because of this unauthorized $400 charge!"
```

**Output:**
```
CLASSIFIED INTENT:  Payment Issue  (Confidence: 91.15%)

ESCALATION DECISION:
{
  "decision": "ESCALATE_TO_HUMAN",
  "reason": "Legal risk detected: query matches legal complaint criteria ('lawyer')."
}
```

---

## 5. Running Automated Tests

Run the test suite with `pytest`:
```bash
pytest tests/ -v
```

---

## 6. Engineering Highlights

- **8 Support Intents**: `Order Delay`, `Refund Request`, `Account Access`, `Payment Issue`, `Product Complaint`, `Delivery Problem`, `Return Request`, `Other`.
- **RAG Architecture**: Fast dense semantic search using `all-MiniLM-L6-v2` + FAISS (`IndexFlatIP` on unit normalized vectors).
- **Escalation Rules**: Zero-tolerance regex detectors for Legal threats, Fraud, Abuse/Profanity, Account Compromise, Repeated Contacts, and statistical confidence gating (`< 0.55`).
- **Claude as Judge**: Multi-criteria Likert evaluation (1-5) across Correctness, Relevance, Tone, and Helpfulness.
- **Decision Log**: See [decision_log.md](decision_log.md) for 15 detailed engineering decisions and trade-offs.
