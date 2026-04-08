import ollama
import pandas as pd
import json
import random
from loguru import logger

CATEGORIES = [
    "Infrastructure",
    "Application", 
    "Security",
    "Database",
    "Storage",
    "Network"
]

PRIORITIES = ["Low", "Medium", "High", "Critical"]

PROMPTS = {
    "Infrastructure": [
        "server crash", "CPU overload", "memory leak",
        "hardware failure", "OS update issue", "boot failure"
    ],
    "Application": [
        "app crash", "login error", "slow performance",
        "feature not working", "UI bug", "API timeout"
    ],
    "Security": [
        "unauthorized access", "password breach", "malware detected",
        "certificate expired", "firewall blocked", "suspicious login"
    ],
    "Database": [
        "database down", "slow query", "connection timeout",
        "data corruption", "backup failed", "replication error"
    ],
    "Storage": [
        "disk full", "file missing", "NAS not accessible",
        "backup storage error", "RAID failure", "mount point error"
    ],
    "Network": [
        "VPN not connecting", "internet down", "DNS failure",
        "packet loss", "firewall issue", "WiFi dropping"
    ]
}

def generate_ticket(category: str, issue: str) -> dict:
    prompt = f"""Generate a realistic IT support ticket for this issue: {issue}
Category: {category}

Return ONLY valid JSON in this exact format:
{{
    "title": "short ticket title here",
    "description": "detailed description of the issue here",
    "category": "{category}",
    "resolution": "step by step resolution here",
    "priority": "High"
}}

Make it realistic like a real enterprise IT ticket.
No markdown, no explanation, just the JSON."""

    try:
        response = ollama.chat(
            model="mistral:7b-instruct-q8_0",
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = response['message']['content'].strip()
        
        # Clean response
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        
        ticket = json.loads(content)
        ticket['priority'] = random.choice(PRIORITIES)
        return ticket
        
    except Exception as e:
        logger.error(f"Error generating ticket: {e}")
        return None

def generate_dataset(total_tickets: int = 100):
    tickets = []
    per_category = total_tickets // len(CATEGORIES)
    
    logger.info(f"Generating {total_tickets} synthetic tickets...")
    
    for category in CATEGORIES:
        issues = PROMPTS[category]
        count = 0
        
        while count < per_category:
            issue = random.choice(issues)
            logger.info(f"Generating {category} ticket {count+1}/{per_category}: {issue}")
            
            ticket = generate_ticket(category, issue)
            
            if ticket:
                tickets.append(ticket)
                count += 1
    
    df = pd.DataFrame(tickets)
    df.to_csv("data/synthetic_tickets.csv", index=False)
    logger.success(f"Generated {len(tickets)} tickets saved to data/synthetic_tickets.csv")
    return df

if __name__ == "__main__":
    df = generate_dataset(1000)
    print(df.head())
    print(f"\nTotal tickets: {len(df)}")
    print(f"\nCategory distribution:")
    print(df['category'].value_counts())