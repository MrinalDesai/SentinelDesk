import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from utils.database import get_connection
from config.settings import settings

def get_routing_rule(category: str, priority: str) -> dict:
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT department, escalation_contact
            FROM routing_rules
            WHERE category = %s AND priority = %s
        """, (category, priority))
        
        row = cur.fetchone()
        cur.close()
        conn.close()
        
        if row:
            return {
                "department": row[0],
                "escalation_contact": row[1]
            }
        return {
            "department": "IT General Support",
            "escalation_contact": "it-support@company.com"
        }
    except Exception as e:
        logger.error(f"Routing rule error: {e}")
        return {
            "department": "IT General Support",
            "escalation_contact": "it-support@company.com"
        }

def route_ticket(
    category: str,
    priority: str,
    confidence: float,
    ticket_id: int = 0
) -> dict:
    logger.info(
        f"Agent 3: Routing {category} / {priority} "
        f"(confidence: {confidence})"
    )
    
    # Get RBAC routing rule
    rule = get_routing_rule(category, priority)
    
    # Determine if escalation needed
    needs_escalation = confidence < settings.CONFIDENCE_THRESHOLD
    
    result = {
        "department": rule["department"],
        "escalation_contact": rule["escalation_contact"],
        "needs_escalation": needs_escalation,
        "routing_reason": f"{category} ticket with {priority} priority → {rule['department']}"
    }
    
    # Log audit
    log_audit(
        ticket_id=ticket_id,
        action="ROUTING",
        details=f"Routed to {rule['department']}. "
               f"Escalation needed: {needs_escalation}",
        confidence=confidence
    )
    
    logger.success(
        f"Agent 3: Routed to {rule['department']} "
        f"— Escalation: {needs_escalation}"
    )
    
    return result

def log_audit(
    ticket_id: int,
    action: str,
    details: str,
    confidence: float
):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO audit_log
            (ticket_id, agent_name, action, details, confidence_score)
            VALUES (%s, %s, %s, %s, %s)
        """, (ticket_id, "Agent3_Router", action, details, confidence))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"Audit log error: {e}")

if __name__ == "__main__":
    test_cases = [
        ("Network",    "High",     0.91),
        ("Security",   "Critical", 0.87),
        ("Database",   "Medium",   0.65),
        ("Application","Low",      0.45),
    ]
    
    print("\n--- Agent 3 Routing Tests ---")
    for category, priority, confidence in test_cases:
        result = route_ticket(category, priority, confidence)
        print(f"\nCategory: {category} | Priority: {priority} | Confidence: {confidence}")
        print(f"Department: {result['department']}")
        print(f"Contact: {result['escalation_contact']}")
        print(f"Needs escalation: {result['needs_escalation']}")