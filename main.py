"""
Main CLI Entry Point for Hiver Support Agent.
Supports modular execution modes:
  python main.py --mode preprocess
  python main.py --mode train
  python main.py --mode golden
  python main.py --mode evaluate
  python main.py --mode query "Where is my package?"
  python main.py --mode full
"""

import sys
import json
import argparse
import logging
from typing import Optional

from src.preprocess import load_and_preprocess_data
from src.intent_classifier import train_and_evaluate, IntentClassifier
from src.retriever import HistoricalRetriever
from src.generator import ResponseGenerator
from src.escalation import EscalationDetector
from src.golden_dataset_gen import build_golden_dataset
from src.evaluator import run_evaluation
from src.report_generator import generate_markdown_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("main")


def run_single_query(query: str, confidence_threshold: float = 0.55):
    """Processes a single incoming customer query through the entire agent pipeline."""
    logger.info(f"Processing Incoming Query: '{query}'")

    # 1. Load Intent Classifier
    try:
        classifier = IntentClassifier.load("data/intent_model.joblib")
    except Exception:
        logger.warning("Trained model not found. Training model now...")
        df = load_and_preprocess_data()
        train_and_evaluate(df)
        classifier = IntentClassifier.load("data/intent_model.joblib")

    predicted_intent, confidence = classifier.predict_single(query)

    # 2. Escalation Detection
    escalation_detector = EscalationDetector(confidence_threshold=confidence_threshold)
    escalation_result = escalation_detector.evaluate(
        query=query,
        predicted_intent=predicted_intent,
        confidence=confidence
    )

    # 3. Retrieve Historical Conversations
    processed_df = load_and_preprocess_data()
    retriever = HistoricalRetriever()
    retriever.build_index(processed_df)
    retrieved_examples = retriever.retrieve_similar(query, top_k=5)
    formatted_examples = retriever.format_examples_for_prompt(retrieved_examples)

    # 4. Generate Support Reply
    generator = ResponseGenerator()
    generated_reply = generator.generate_reply(
        query=query,
        retrieved_examples_text=formatted_examples,
        intent=predicted_intent
    )

    # Print structured output
    print("\n" + "=" * 60)
    print("           AMAZONHELP AI SUPPORT AGENT - INFERENCE")
    print("=" * 60)
    print(f"CUSTOMER QUERY:\n  {query}\n")
    print(f"CLASSIFIED INTENT:  {predicted_intent}  (Confidence: {confidence:.2%})")
    print("\nESCALATION DECISION:")
    escalation_output = {
        "decision": escalation_result["decision"],
        "reason": escalation_result["reason"]
    }
    print(json.dumps(escalation_output, indent=2))

    print("\nTOP-3 RETRIEVED HISTORICAL EXAMPLES:")
    for i, ex in enumerate(retrieved_examples[:3], 1):
        print(f"  [{i}] (Sim: {ex['similarity_score']:.2f}, Intent: {ex['intent']})")
        print(f"      Cust: {ex['customer_message'][:80]}...")
        print(f"      Agent: {ex['agent_reply'][:80]}...")

    print("\nGENERATED AMAZONHELP RESPONSE:")
    print(f"  \"{generated_reply}\"")
    print("=" * 60 + "\n")

    return {
        "query": query,
        "intent": predicted_intent,
        "confidence": confidence,
        "escalation": escalation_output,
        "generated_reply": generated_reply
    }


def main():
    parser = argparse.ArgumentParser(description="Hiver Support Agent - AmazonHelp Twitter AI Pipeline")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["preprocess", "train", "golden", "evaluate", "query", "full"],
        default="full",
        help="Pipeline execution mode."
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Where is my order? The tracking has been stuck for 4 days.",
        help="Customer query for '--mode query'."
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.55,
        help="Confidence threshold below which intent is flagged as low confidence."
    )

    args = parser.parse_args()

    if args.mode == "preprocess":
        logger.info("Executing STEP 1: Data Preprocessing...")
        df = load_and_preprocess_data()
        print(f"\nPreprocessing Complete. Processed {len(df)} records.")
        print("Intent distribution:\n", df["intent"].value_counts())

    elif args.mode == "train":
        logger.info("Executing STEP 2 & STEP 5: Intent Classification & Baselines...")
        df = load_and_preprocess_data()
        results = train_and_evaluate(df)
        print("\n--- BASELINE & MODEL EVALUATION ---")
        print(f"Baseline 1 (Majority Class) F1:   {results['baseline_majority']['f1_weighted']}")
        print(f"Baseline 2 (Keyword Rule) F1:      {results['baseline_keyword']['f1_weighted']}")
        print(f"Trained Logistic Regression F1:    {results['logistic_regression']['f1_weighted']}")
        print("\nDetailed Classification Report:\n", results["detailed_classification_report"])

    elif args.mode == "golden":
        logger.info("Executing STEP 6: Golden Dataset Generation...")
        df = load_and_preprocess_data()
        golden_df = build_golden_dataset(df)
        print(f"\nGolden Dataset created at golden_dataset/golden_dataset.csv with {len(golden_df)} samples.")
        print("Intent breakdown:\n", golden_df["intent"].value_counts())
        print("Escalation breakdown:\n", golden_df["escalation"].value_counts())

    elif args.mode == "evaluate":
        logger.info("Executing STEP 7 & STEP 8: Evaluation Harness & Failure Analysis...")
        summary = run_evaluation(confidence_threshold=args.confidence_threshold)
        print("\n--- EVALUATION HARNESS SUMMARY ---")
        print(json.dumps(summary, indent=2))

    elif args.mode == "query":
        run_single_query(args.query, confidence_threshold=args.confidence_threshold)

    elif args.mode == "full":
        logger.info("Executing FULL PIPELINE (STEPS 1 through 10)...")
        # Step 1: Preprocess
        df = load_and_preprocess_data()
        # Step 2 & 5: Train & Baselines
        training_results = train_and_evaluate(df)
        # Step 6: Golden Dataset
        build_golden_dataset(df)
        # Step 7 & 8: Evaluation & Failure Analysis
        eval_summary = run_evaluation(confidence_threshold=args.confidence_threshold)
        # Step 9: Generate Report
        generate_markdown_report(training_results, eval_summary)

        print("\n" + "=" * 60)
        print("           FULL PIPELINE EXECUTION COMPLETED")
        print("=" * 60)
        print(f"Trained Intent Model F1:       {training_results['logistic_regression']['f1_weighted']}")
        print(f"Golden Set Intent Accuracy:    {eval_summary['intent_metrics']['accuracy']}")
        print(f"Escalation Decision Accuracy:  {eval_summary['escalation_metrics']['accuracy']}")
        print(f"Claude Judge Quality (1-5):    {eval_summary['judge_scores']['overall_average']} / 5.0")
        print(f"Failures Identified:           {eval_summary['failure_count']} (Logged in reports/failure_analysis.json)")
        print("Reports Generated:")
        print("  - reports/report.md")
        print("  - reports/decision_log.md")
        print("  - golden_dataset/golden_dataset.csv")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
