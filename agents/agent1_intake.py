import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from loguru import logger
from utils.database import get_connection
import json

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def redact_pii(text: str) -> dict:
    results = analyzer.analyze(
        text=text,
        language='en'
    )
    
    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results
    )
    
    pii_found = [r.entity_type for r in results]
    
    return {
        "clean_text": anonymized.text,
        "pii_detected": pii_found,
        "pii_count": len(pii_found)
    }

def process_ticket(ticket: dict) -> dict:
    logger.info(f"Agent 1: Processing ticket — {ticket.get('title', '')[:50]}")
    
    # Redact PII from title and description
    title_result = redact_pii(ticket.get('title', ''))
    desc_result = redact_pii(ticket.get('description', ''))
    
    clean_ticket = {
        "title": title_result["clean_text"],
        "description": desc_result["clean_text"],
        "priority": ticket.get("priority", "Medium"),
        "pii_detected": title_result["pii_detected"] + desc_result["pii_detected"],
        "original_title": ticket.get("title", "")
    }
    
    # Log to audit
    log_audit(
        ticket_id=ticket.get("id", 0),
        action="PII_REDACTION",
        details=f"Detected PII: {clean_ticket['pii_detected']}",
        confidence=1.0
    )
    
    logger.success(f"Agent 1: PII redacted — {len(clean_ticket['pii_detected'])} entities found")
    return clean_ticket

def log_audit(ticket_id: int, action: str, details: str, confidence: float):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO audit_log 
            (ticket_id, agent_name, action, details, confidence_score)
            VALUES (%s, %s, %s, %s, %s)
        """, (ticket_id, "Agent1_Intake", action, details, confidence))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"Audit log error: {e}")

if __name__ == "__main__":
    # Test Agent 1
    test_ticket = {
        "id": 1,
        "title": "John Smith cannot access VPN",
        "description": "User john.smith@company.com with IP 192.168.1.100 reports VPN not connecting after password reset yesterday.",
        "priority": "High"
    }
    
    result = process_ticket(test_ticket)
    print("\n--- Agent 1 Result ---")
    print(f"Clean title: {result['title']}")
    print(f"Clean description: {result['description']}")
    print(f"PII detected: {result['pii_detected']}")