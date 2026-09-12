"""
Evaluation Harness Module.
Orchestrates end-to-end evaluation on the Golden Dataset:
- Intent classification performance (Accuracy, Precision, Recall, F1)
- Reply quality via Claude-as-a-Judge (Correctness, Relevance, Tone, Helpfulness 1-5)
- Automated failure identification and aggregation
"""

import os
import json
import logging
import statistics
import pandas as pd
from typing import Dict, Any, List
from tqdm import tqdm  # type: ignore

from src.intent_classifier import IntentClassifier, compute_metrics
from src.retriever import HistoricalRetriever
from src.generator import ResponseGenerator
from src.escalation import EscalationDetector
from src.failure_analysis import FailureAnalyzer

logger = logging.getLogger(__name__)


def run_evaluation(
    golden_csv_path: str = "golden_dataset/golden_dataset.csv",
    model_path: str = "data/intent_model.joblib",
    sample_limit: int = 200,
    confidence_threshold: float = 0.55
) -> Dict[str, Any]:
    """
    Executes full evaluation harness against the golden dataset.
    Returns complete evaluation summary dictionary.
    """
    from src.preprocess import load_and_preprocess_data
    processed_df = load_and_preprocess_data()

    if not os.path.exists(model_path):
        logger.info(f"Model file {model_path} not found. Training model now...")
        from src.intent_classifier import train_and_evaluate
        train_and_evaluate(processed_df, model_save_path=model_path)

    if not os.path.exists(golden_csv_path):
        logger.info(f"Golden dataset {golden_csv_path} not found. Generating now...")
        from src.golden_dataset_gen import build_golden_dataset
        golden_df = build_golden_dataset(processed_df, output_path=golden_csv_path)
    else:
        golden_df = pd.read_csv(golden_csv_path)

    if sample_limit and sample_limit < len(golden_df):
        golden_df = golden_df.iloc[:sample_limit]

    logger.info(f"Running evaluation harness on {len(golden_df)} golden dataset samples...")

    # Load modules
    classifier = IntentClassifier.load(model_path)
    retriever = HistoricalRetriever()
    retriever.build_index(processed_df)

    generator = ResponseGenerator()
    escalation_detector = EscalationDetector(confidence_threshold=confidence_threshold)
    failure_analyzer = FailureAnalyzer()

    true_intents = []
    pred_intents = []
    true_escalations = []
    pred_escalations = []

    judge_scores_list = {
        "correctness": [],
        "relevance": [],
        "tone": [],
        "helpfulness": []
    }

    all_failures = []
    evaluated_records = []

    for idx, row in tqdm(golden_df.iterrows(), total=len(golden_df), desc="Evaluating Golden Samples"):
        tweet = str(row["tweet"])
        true_intent = str(row["intent"])
        true_reply = str(row["human_reply"])
        true_esc = str(row["escalation"])

        # 1. Intent Classification
        pred_intent, confidence = classifier.predict_single(tweet)
        probs_raw = classifier.predict_proba([tweet])[0]
        intent_probs = {cls: float(p) for cls, p in zip(classifier.classes_, probs_raw)}

        true_intents.append(true_intent)
        pred_intents.append(pred_intent)

        # 2. Escalation Decision
        esc_decision = escalation_detector.evaluate(
            query=tweet,
            predicted_intent=pred_intent,
            confidence=confidence
        )
        pred_esc = esc_decision["decision"]
        true_escalations.append(true_esc)
        pred_escalations.append(pred_esc)

        # 3. Retrieval & Generation
        top_examples = retriever.retrieve_similar(tweet, top_k=5)
        formatted_examples = retriever.format_examples_for_prompt(top_examples)
        generated_reply = generator.generate_reply(
            query=tweet,
            retrieved_examples_text=formatted_examples,
            intent=pred_intent
        )

        # 4. Claude as Judge
        judge_res = generator.judge_reply(
            query=tweet,
            intent=pred_intent,
            generated_reply=generated_reply,
            human_reply=true_reply
        )

        for metric_name in judge_scores_list.keys():
            judge_scores_list[metric_name].append(judge_res.get(metric_name, 4))

        # 5. Failure Analysis
        failures = failure_analyzer.analyze_sample(
            tweet=tweet,
            true_intent=true_intent,
            pred_intent=pred_intent,
            intent_probs=intent_probs,
            top_retrievals=top_examples,
            generated_reply=generated_reply,
            true_escalation=true_esc,
            pred_escalation=pred_esc,
            escalation_reason=esc_decision["reason"],
            judge_scores=judge_res
        )

        if failures:
            all_failures.append({
                "sample_id": idx,
                "tweet": tweet,
                "true_intent": true_intent,
                "predicted_intent": pred_intent,
                "confidence": round(confidence, 4),
                "true_escalation": true_esc,
                "predicted_escalation": pred_esc,
                "generated_reply": generated_reply,
                "failures": failures
            })

        evaluated_records.append({
            "tweet": tweet,
            "true_intent": true_intent,
            "pred_intent": pred_intent,
            "confidence": round(confidence, 4),
            "true_escalation": true_esc,
            "pred_escalation": pred_esc,
            "correctness": judge_res.get("correctness", 4),
            "relevance": judge_res.get("relevance", 4),
            "tone": judge_res.get("tone", 4),
            "helpfulness": judge_res.get("helpfulness", 4),
            "generated_reply": generated_reply
        })

    # Compute aggregate metrics
    intent_metrics = compute_metrics(true_intents, pred_intents)
    escalation_metrics = compute_metrics(true_escalations, pred_escalations)

    judge_averages = {
        k: round(float(statistics.mean(v)), 2) for k, v in judge_scores_list.items() if v
    }
    overall_quality_score = round(float(statistics.mean(list(judge_averages.values()))), 2)
    judge_averages["overall_average"] = overall_quality_score

    # Save failure analysis
    failure_analyzer.save_failure_report(all_failures, "reports/failure_analysis.json")

    eval_summary = {
        "total_samples": len(golden_df),
        "intent_metrics": intent_metrics,
        "escalation_metrics": escalation_metrics,
        "judge_scores": judge_averages,
        "failure_count": len(all_failures),
        "failure_rate": round(len(all_failures) / len(golden_df), 4)
    }

    # Save evaluation summary to reports/
    os.makedirs("reports", exist_ok=True)
    with open("reports/evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    logger.info("Evaluation Complete. Summary:\n" + json.dumps(eval_summary, indent=2))
    return eval_summary


if __name__ == "__main__":
    summary = run_evaluation()
