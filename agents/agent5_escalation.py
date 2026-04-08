import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ollama
import json
import re
from loguru import logger
from config.settings import settings
from utils.database import get_connection

def escalate_ticket(
    title: str,
    description: str,
    category: str,
    priority: str,
    confidence: float,
    resolution: str,
    department: str,
    escalation_contact: str,
    ticket_id: int = 0
) -> dict:
    logger.info(f"Agent 5: Evaluating escalation — confidence: {confidence}")
    
    # Rule 1: Low confidence → escalate
    if confidence < settings.CONFIDENCE_THRESHOLD:
        result = {
            "escalate": True,
            "reason": f"Low confidence score ({confidence:.2f} < {settings.CONFIDENCE_THRESHOLD})",
            "escalation_level": "L2",
            "contact": escalation_contact,
            "action": "ESCALATE"
        }
        logger.warning(f"Agent 5: ESCALATING — low confidence {confidence}")
        log_audit(ticket_id, "ESCALATION",
                 f"Escalated to L2: {result['reason']}", confidence)
        return result
    
    # Rule 2: Critical priority → always escalate
    if priority == "Critical":
        result = {
            "escalate": True,
            "reason": "Critical priority ticket requires immediate L2 attention",
            "escalation_level": "L2",
            "contact": escalation_contact,
            "action": "ESCALATE"
        }
        logger.warning(f"Agent 5: ESCALATING — critical priority")
        log_audit(ticket_id, "ESCALATION",
                 f"Escalated: Critical priority", confidence)
        return result
    
    # Rule 3: LLM judge resolution quality
    judge_prompt = f"""You are a quality evaluator for IT ticket resolutions.

Ticket: {title}
Category: {category}
Resolution provided: {resolution}

Score this resolution from 1-5:
1 = completely wrong or unhelpful
2 = partially correct
3 = acceptable
4 = good resolution
5 = excellent resolution

Return ONLY this JSON:
{{
    "score": 4,
    "quality": "good",
    "escalate": false
}}"""

    try:
        response = ollama.chat(
            model=settings.LLM_MODEL,
            messages=[{"role": "user", "content": judge_prompt}]
        )
        
        content = response['message']['content'].strip()
        
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            content = match.group()
        
        judge = json.loads(content)
        score = judge.get('score', 3)
        
        if score < 3:
            result = {
                "escalate": True,
                "reason": f"Low resolution quality score ({score}/5)",
                "escalation_level": "L2",
                "contact": escalation_contact,
                "action": "ESCALATE",
                "quality_score": score
            }
            logger.warning(f"Agent 5: ESCALATING — low quality score {score}/5")
        else:
            result = {
                "escalate": False,
                "reason": f"Resolution quality acceptable ({score}/5)",
                "escalation_level": "L1",
                "contact": department,
                "action": "RESOLVE",
                "quality_score": score
            }
            logger.success(f"Agent 5: RESOLVED — quality score {score}/5")
        
        log_audit(
            ticket_id,
            result['action'],
            f"Quality score: {score}/5. {result['reason']}",
            confidence
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Agent 5 error: {e}")
        return {
            "escalate": False,
            "reason": "Judge evaluation failed — defaulting to resolve",
            "escalation_level": "L1",
            "contact": department,
            "action": "RESOLVE",
            "quality_score": 3
        }

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
        """, (ticket_id, "Agent5_Escalation", action, details, confidence))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"Audit log error: {e}")

if __name__ == "__main__":
    print("\n--- Agent 5 Escalation Tests ---")
    
    # Test 1: Low confidence → should escalate
    result = escalate_ticket(
        title="VPN not connecting",
        description="User cannot connect to VPN",
        category="Network",
        priority="High",
        confidence=0.45,
        resolution="Check VPN settings",
        department="NOC Team",
        escalation_contact="noc@company.com"
    )
    print(f"\nTest 1 (low confidence):")
    print(f"Escalate: {result['escalate']}")
    print(f"Reason: {result['reason']}")
    print(f"Action: {result['action']}")
    
    # Test 2: High confidence → should resolve
    result = escalate_ticket(
        title="Database connection timeout",
        description="MySQL database throwing timeout errors",
        category="Database",
        priority="Medium",
        confidence=0.91,
        resolution="Restart database service and check connection pool settings",
        department="DBA Team",
        escalation_contact="dba@company.com"
    )
    print(f"\nTest 2 (high confidence):")
    print(f"Escalate: {result['escalate']}")
    print(f"Reason: {result['reason']}")
    print(f"Action: {result['action']}")