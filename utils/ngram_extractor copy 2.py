import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import nltk
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter
from loguru import logger

CATEGORIES = [
    "Infrastructure",
    "Application",
    "Security",
    "Database",
    "Storage",
    "Network"
]

STOP_WORDS = set(stopwords.words('english'))

def extract_ngrams(text: str, n: int) -> list:
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens
              if t.isalpha() and t not in STOP_WORDS]
    return [' '.join(gram) for gram in ngrams(tokens, n)]

def build_vocabulary(df: pd.DataFrame) -> dict:
    logger.info("Building N-gram vocabulary from tickets...")
    
    vocabulary = {}
    
    for category in CATEGORIES:
        cat_df = df[df['category'] == category]
        
        if len(cat_df) == 0:
            logger.warning(f"No tickets found for {category}")
            vocabulary[category] = []
            continue
        
        all_ngrams = []
        
        for _, row in cat_df.iterrows():
            text = f"{row.get('title','')} {row.get('description','')}"
            
            # Extract 1, 2, 3 grams
            all_ngrams.extend(extract_ngrams(text, 1))
            all_ngrams.extend(extract_ngrams(text, 2))
            all_ngrams.extend(extract_ngrams(text, 3))
        
        # Count and get top 20
        counter = Counter(all_ngrams)
        top_ngrams = [gram for gram, count 
                      in counter.most_common(20)]
        
        vocabulary[category] = top_ngrams
        logger.success(
            f"{category}: {len(top_ngrams)} n-grams extracted"
        )
    
    return vocabulary

def save_vocabulary(vocabulary: dict, path: str = "data/vocabulary.json"):
    with open(path, 'w') as f:
        json.dump(vocabulary, f, indent=2)
    logger.success(f"Vocabulary saved to {path}")

def load_vocabulary(path: str = "data/vocabulary.json") -> dict:
    if not os.path.exists(path):
        logger.warning("Vocabulary not found — using defaults")
        return get_default_vocabulary()
    
    with open(path, 'r') as f:
        return json.load(f)
def get_default_vocabulary() -> dict:
    # MUTUALLY EXCLUSIVE — no word appears in two categories
    return {
        "Infrastructure": [
            "bare metal", "rack mount", "psu failure",
            "dimm slot", "bios update", "motherboard fault",
            "chassis temperature", "cooling fan", "ups battery",
            "pdu tripped", "data center rack", "hardware fault"
        ],
        "Application": [
            "null pointer", "http 500", "rest api",
            "apk crash", "heap memory", "cron job",
            "saml token", "build pipeline", "http 403",
            "microservice", "frontend rendering", "batch job"
        ],
        "Security": [
            "ransomware", "phishing email", "brute force",
            "ssl certificate expired", "privilege escalation",
            "dlp alert", "zero day exploit", "dark web",
            "port scanning", "malware trojan", "data exfiltration",
            "credential leak"
        ],
        "Database": [
            "innodb deadlock", "tablespace full", "replication lag",
            "replica set election", "execution plan", "binary log",
            "stored procedure overflow", "connection pool exhausted",
            "b-tree index corrupt", "rman backup failed",
            "tempdb growth", "schema migration"
        ],
        "Storage": [
            "nas raid degraded", "san lun", "iscsi disconnecting",
            "tape library robot", "nfs quota exceeded",
            "deduplication ratio", "snapshot consuming",
            "fsck corruption", "vmotion failing",
            "s3 bucket acl", "thin provisioned", "cifs smb"
        ],
        "Network": [
            "ipsec tunnel dropping", "bgp routing table",
            "dns nxdomain", "mpls packet loss", "802.11ac ssid",
            "vlan trunk misconfiguration", "qos dscp marking",
            "firewall acl blocking", "proxy pac file",
            "spanning tree loop", "dhcp scope exhausted",
            "nat translation full"
        ]
    }
# def get_default_vocabulary() -> dict:
#     return {
#         "Infrastructure": [
#             "server down", "cpu spike", "memory leak",
#             "hardware failure", "os crash", "reboot",
#             "kernel panic", "physical server", "data center"
#         ],
#         "Application": [
#             "app crash", "application error", "software bug",
#             "login failed", "ui issue", "api error",
#             "timeout", "500 error", "deployment failed"
#         ],
#         # "Security": [
#         #     "unauthorized access", "breach", "malware",
#         #     "certificate expired", "firewall", "intrusion",
#         #     "suspicious login", "ransomware", "phishing"
#         # ],

#         # "Security": [
#         #     "unauthorized access", "breach", "malware",
#         #     "certificate expired", "suspicious login",
#         #     "ransomware", "phishing", "intrusion detected",
#         #     "privilege escalation", "security alert",
#         #     "firewall blocked", "vulnerability", "exploit"
#         # ],
#         "Security": [
#               "unauthorized access", "breach", "malware",
#               "suspicious login", "ransomware", "phishing",
#               "intrusion detected", "certificate expired",
#               "privilege escalation", "brute force"
#                     ],
#         "Database": [
#             "database down", "sql error", "query slow",
#             "connection timeout", "data corruption",
#             "backup failed", "replication error", "deadlock"
#         ],
#         "Storage": [
#             "disk full", "file missing", "nas not accessible",
#             "raid failure", "mount point", "backup storage",
#             "quota exceeded", "storage full"
#         ],
#         "Network": [
#             "vpn not connecting", "internet down", "dns failure",
#             "packet loss", "latency", "firewall rule",
#             "wifi dropping", "bandwidth", "connectivity"
#         ]
#     }

# def build_enriched_prompt(
#     title: str,
#     description: str,
#     vocabulary: dict
# ) -> str:
#     prompt = """You are an IT ticket classifier for an enterprise system.

# Domain vocabulary signatures (extracted from real tickets):
# """
#     for category, ngrams_list in vocabulary.items():
#         top_terms = ', '.join(ngrams_list[:8])
#         prompt += f"\n{category}: {top_terms}"
    
#     prompt += f"""

# Classify this ticket into EXACTLY ONE category.

# Ticket Title: {title}
# Ticket Description: {description}

# Rules:
# - Choose only from: Infrastructure, Application, Security, Database, Storage, Network
# - confidence must be decimal 0.0 to 1.0
# - Return ONLY valid JSON

# {{
#     "category": "Network",
#     "confidence": 0.92,
#     "reasoning": "one line reason"
# }}"""
    
#     return prompt
def build_enriched_prompt(
    title: str,
    description: str,
    vocabulary: dict
) -> str:
    signatures = ""
    for category, ngrams_list in vocabulary.items():
        top_terms = ', '.join(ngrams_list[:6])
        signatures += f"\n{category}: {top_terms}"

    return f"""You are an expert IT ticket classifier.

Category signatures:{signatures}

Ticket: {title}. {description[:300]}

CRITICAL RULES - each category is MUTUALLY EXCLUSIVE:
- Infrastructure = PHYSICAL hardware ONLY: server rack, PSU, motherboard, BIOS, UPS, cooling fan, bare metal, chassis
- Application = SOFTWARE ONLY: API errors, app crashes, HTTP codes, null pointer, deployment, heap, cron job, microservice
- Security = THREATS ONLY: ransomware, phishing, malware, breach, unauthorized access, privilege escalation, dark web
- Database = DATA LAYER ONLY: MySQL, Oracle, PostgreSQL, deadlock, replication, tablespace, stored procedure, binary log
- Storage = DISK/FILE ONLY: NAS, SAN, RAID, iSCSI, disk full, NFS, CIFS, snapshot, tape library
- Network = CONNECTIVITY ONLY: VPN, DNS, BGP, firewall ACL, packet loss, WiFi, VLAN, DHCP, spanning tree
- KEY: "server" alone is NOT Infrastructure — must see physical terms like rack/PSU/BIOS/motherboard
- KEY: "error" alone is NOT Application — must see software terms like HTTP/API/null pointer/deployment

Return ONLY this JSON:
{{
    "category": "Infrastructure",
    "confidence": 0.92,
    "reasoning": "specific keyword that matched"
}}"""
# def build_enriched_prompt(
#     title: str,
#     description: str,
#     vocabulary: dict
# ) -> str:
#     signatures = ""
#     for category, ngrams_list in vocabulary.items():
#         top_terms = ', '.join(ngrams_list[:5])
#         signatures += f"\n{category}: {top_terms}"

#     return f"""Classify this IT ticket. Choose ONE category only.

# Category signatures:{signatures}

# Ticket: {title}. {description[:200]}

# Important rules:
# - Security tickets mention: hacking, malware, unauthorized, breach, suspicious
# - Infrastructure tickets mention: server, hardware, cpu, memory, reboot
# - Never default to one category
# - Analyze ticket content carefully

# Return ONLY this JSON:
# {{
#     "category": "Infrastructure",
#     "confidence": 0.85,
#     "reasoning": "brief reason"
# }}"""

# Category signatures:{signatures}

# Ticket: {title}. {description[:200]}

# Important: Choose the category that BEST matches the ticket content.
# Do NOT default to Network.

# Return ONLY this JSON with no explanation:
# {{
#     "category": "Infrastructure",
#     "confidence": 0.85,
#     "reasoning": "brief reason"
# }}"""
# Category signatures:{signatures}

# Ticket: {title}. {description[:200]}

# Return ONLY this JSON:
# {{
#     "category": "Network",
#     "confidence": 0.92,
#     "reasoning": "brief reason"
# }}"""

if __name__ == "__main__":
    # Try to load existing tickets
    csv_path = "data/synthetic_tickets.csv"
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path).fillna("")
        logger.info(f"Loaded {len(df)} tickets")
        
        vocabulary = build_vocabulary(df)
        save_vocabulary(vocabulary)
        
        print("\n--- Extracted Vocabulary ---")
        for cat, terms in vocabulary.items():
            print(f"\n{cat}:")
            print(f"  {', '.join(terms[:5])}")
    else:
        logger.warning("No CSV found — using defaults")
        vocabulary = get_default_vocabulary()
        save_vocabulary(vocabulary)