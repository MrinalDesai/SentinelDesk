import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    confusion_matrix
)
from loguru import logger
from agents.agent2_classifier import classify_ticket

def evaluate_classifier(sample_size: int = 20) -> dict:
    logger.info(f"Evaluating classifier on {sample_size} tickets...")
    
    # Load tickets
    df = pd.read_csv("data/synthetic_tickets.csv").fillna("")
    df = df.sample(min(sample_size, len(df)), random_state=42)
    
    true_labels = []
    pred_labels = []
    confidences = []
    
    for idx, row in df.iterrows():
        true_label = row['category']
        
        result = classify_ticket(
            title=str(row['title']),
            description=str(row['description'])
        )
        
        pred_label = result['category']
        confidence = result['confidence']
        
        true_labels.append(true_label)
        pred_labels.append(pred_label)
        confidences.append(confidence)
        
        status = "✅" if true_label == pred_label else "❌"
        logger.info(
            f"{status} True: {true_label:15} "
            f"Pred: {pred_label:15} "
            f"Conf: {confidence:.2f}"
        )
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    f1_macro = f1_score(
        true_labels, pred_labels,
        average='macro',
        zero_division=0
    )
    avg_confidence = sum(confidences) / len(confidences)
    
    report = classification_report(
        true_labels, pred_labels,
        zero_division=0
    )
    
    results = {
        "accuracy":       round(accuracy, 4),
        "f1_macro":       round(f1_macro, 4),
        "avg_confidence": round(avg_confidence, 4),
        "sample_size":    sample_size,
        "report":         report
    }
    
    logger.success(f"Accuracy:       {accuracy:.2%}")
    logger.success(f"F1 Macro:       {f1_macro:.2%}")
    logger.success(f"Avg Confidence: {avg_confidence:.2%}")
    
    return results

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  SENTINELDESK — Classifier Evaluation")
    print("="*60)
    
    results = evaluate_classifier(sample_size=20)
    
    print(f"\nAccuracy:        {results['accuracy']:.2%}")
    print(f"F1 Score:        {results['f1_macro']:.2%}")
    print(f"Avg Confidence:  {results['avg_confidence']:.2%}")
    print(f"\nClassification Report:")
    print(results['report'])