import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ollama
import json
import re
from loguru import logger
from config.settings import settings
from utils.database import get_connection

CATEGORIES = [
    "Infrastructure",
    "Application",
    "Security", 
    "Database",
    "Storage",
    "Network"
]

DOMAIN_SIGNATURES = {
    "Infrastructure": [
        "server down", "cpu spike", "memory leak", "hardware failure",
        "os crash", "reboot", "kernel panic", "physical server",
        "data center", "rack", "power supply", "motherboard"
    ],
    "Application": [
        "app crash", "application error", "software bug", "login failed",
        "ui issue", "api error", "timeout", "500 error", "deployment",
        "code issue", "feature broken", "app not loading"
    ],
    "Security": [
        "unauthorized access", "breach", "malware", "virus", "phishing",
        "certificate expired", "ssl", "firewall", "intrusion", "hack",
        "suspicious", "privilege escalation", "ransomware"
    ],
    "Database": [
        "database down", "sql error", "query slow", "connection timeout",
        "data corruption", "backup failed", "mysql", "postgresql",
        "oracle", "mongodb", "replication", "deadlock"
    ],
    "Storage": [
        "disk full", "storage full", "file missing", "nas", "san",
        "raid failure", "mount point", "backup storage", "s3",
        "blob storage", "file server", "quota exceeded"
    ],
    "Network": [
        "vpn", "internet down", "dns", "packet loss", "latency",
        "firewall rule", "wifi", "ethernet", "routing", "bandwidth",
        "network timeout", "connectivity", "ping failed"
    ]
}

# def build_classification_prompt(title: str, description: str) -> str:
#     signatures = ""
#     for cat, keywords in DOMAIN_SIGNATURES.items():
#         signatures += f"\n{cat}: {', '.join(keywords[:6])}"
    
#     return f"""You are an IT ticket classifier for an enterprise IT system.

# Domain signatures:{signatures}

# Classify this ticket into EXACTLY ONE category.

# Ticket Title: {title}
# Ticket Description: {description}

# Rules:
# - Choose only from: Infrastructure, Application, Security, Database, Storage, Network
# - confidence must be a decimal between 0.0 and 1.0
# - Return ONLY valid JSON, no explanation

# Return this exact JSON format:
# {{
#     "category": "Network",
#     "confidence": 0.92,
#     "reasoning": "one line reason"
# }}"""

def build_classification_prompt(title: str, description: str) -> str:
    # Load n-gram vocabulary
    try:
        from utils.ngram_extractor import load_vocabulary, build_enriched_prompt
        vocabulary = load_vocabulary()
        return build_enriched_prompt(title, description, vocabulary)
    except Exception as e:
        logger.warning(f"N-gram vocab failed, using defaults: {e}")
        # Fallback to basic prompt
        signatures = ""
        for cat, keywords in DOMAIN_SIGNATURES.items():
            signatures += f"\n{cat}: {', '.join(keywords[:6])}"
        
        return f"""You are an IT ticket classifier.

Domain signatures:{signatures}

Classify this ticket into EXACTLY ONE category.

Ticket Title: {title}
Ticket Description: {description}

Return ONLY this JSON:
{{
    "category": "Network",
    "confidence": 0.92,
    "reasoning": "one line reason"
}}"""

def classify_ticket(title: str, description: str) -> dict:
    logger.info(f"Agent 2: Classifying — {title[:50]}")
    
    prompt = build_classification_prompt(title, description)
    
    try:
        # response = ollama.chat(
        #     model=settings.LLM_MODEL,
        #     messages=[{"role": "user", "content": prompt}]
        # )
        response = ollama.chat(
            model=settings.LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert IT ticket classifier. Always choose the most specific category. Never default to one category. Analyze the ticket carefully."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        content = response['message']['content'].strip()
        
        # Clean JSON
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        
        # Extract JSON
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            content = match.group()
        
        result = json.loads(content)
        
        # Validate category
        if result.get('category') not in CATEGORIES:
            result['category'] = CATEGORIES[0]
            result['confidence'] = 0.5
        
        logger.success(
            f"Agent 2: {result['category']} "
            f"(confidence: {result['confidence']})"
        )
        
        # Log audit
        log_audit(
            action="CLASSIFICATION",
            details=f"Category: {result['category']}, "
                   f"Confidence: {result['confidence']}",
            confidence=result['confidence']
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Agent 2 error: {e}")
        return {
            "category": "Infrastructure",
            "confidence": 0.5,
            "reasoning": "Classification failed — defaulting"
        }

def log_audit(action: str, details: str, confidence: float):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO audit_log
            (ticket_id, agent_name, action, details, confidence_score)
            VALUES (%s, %s, %s, %s, %s)
        """, (0, "Agent2_Classifier", action, details, confidence))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"Audit log error: {e}")

if __name__ == "__main__":
    # Test Agent 2
    test_cases = [
        {
            "title": "VPN not connecting after password reset",
            "description": "User cannot connect to corporate VPN since credentials were reset. Authentication timeout error."
        },
        {
            "title": "Database connection timeout",
            "description": "Production MySQL database throwing connection timeout errors. Multiple users affected."
        },
        {
            "title": "Suspicious login attempt detected",
            "description": "Multiple failed login attempts from unknown IP address. Possible brute force attack."
        }
    ]
    
    print("\n--- Agent 2 Classification Tests ---")
    for test in test_cases:
        result = classify_ticket(test['title'], test['description'])
        print(f"\nTicket: {test['title']}")
        print(f"Category: {result['category']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Reasoning: {result.get('reasoning', '')}")