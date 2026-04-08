import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ollama
import pandas as pd
import json
import random
import re
from loguru import logger

PRIORITIES = ["Low", "Medium", "High", "Critical"]

# MUTUALLY EXCLUSIVE issue prompts
# Each category uses UNIQUE terminology not shared with others
PROMPTS = {
    "Infrastructure": [
        "physical server hardware failure in data center rack",
        "CPU overheating due to cooling fan malfunction",
        "RAM memory DIMM slot failure on bare metal server",
        "BIOS firmware update bricked the motherboard",
        "UPS battery backup unit not holding charge",
        "server power supply unit PSU failed",
        "RAID controller card hardware fault",
        "kernel panic on bare metal OS boot",
        "data center PDU power distribution unit tripped",
        "server chassis temperature sensor critical alert",
        "NIC network interface card hardware failure",
        "server rack mount rails broken",
    ],
    "Application": [
        "ERP software throwing null pointer exception on login screen",
        "mobile app APK crashing on Android after update",
        "REST API endpoint returning HTTP 500 error code",
        "frontend UI rendering bug in Chrome browser",
        "batch processing job failing with timeout exception",
        "microservice throwing connection refused on startup",
        "SSO SAML authentication token validation failing",
        "scheduled cron job not executing at midnight",
        "web portal throwing HTTP 403 forbidden error",
        "software deployment pipeline failing at build stage",
        "application throwing OutOfMemoryError on heap",
        "API rate limiting causing throttle errors",
    ],
    "Security": [
        "ransomware encrypting files on workstation",
        "phishing email with malicious attachment detected",
        "brute force attack on SSH port 22",
        "SSL TLS certificate expired on web server",
        "unauthorized privilege escalation on domain controller",
        "DLP data loss prevention alert triggered",
        "zero day exploit detected by antivirus",
        "dark web credential leak for employee accounts",
        "port scanning reconnaissance activity in firewall logs",
        "malware trojan found in email attachment",
        "suspicious process injection on endpoint",
        "insider threat data exfiltration attempt",
    ],
    "Database": [
        "MySQL InnoDB deadlock causing transaction rollback",
        "Oracle tablespace full cannot insert records",
        "PostgreSQL replication lag exceeding threshold",
        "MongoDB replica set primary election timeout",
        "SQL query execution plan using full table scan",
        "database binary transaction log consuming disk",
        "stored procedure arithmetic overflow exception",
        "database connection pool exhausted all threads",
        "corrupt B-tree index causing query failure",
        "database backup RMAN job failed checksum",
        "SQL Server tempdb growing uncontrollably",
        "database schema migration script rollback failed",
    ],
    "Storage": [
        "NAS device RAID array degraded missing disk",
        "SAN LUN volume not mounting on host",
        "iSCSI target disconnecting from initiator randomly",
        "tape library robot arm mechanical failure",
        "backup target NFS share quota exceeded",
        "deduplication ratio dropped causing capacity full",
        "snapshot clone consuming entire storage pool",
        "file system fsck corruption check failed",
        "storage vMotion failing during VM migration",
        "cloud S3 bucket ACL permissions misconfigured",
        "thin provisioned volume overcommitted",
        "CIFS SMB share not accessible to clients",
    ],
    "Network": [
        "VPN IPSec tunnel dropping every 30 minutes",
        "BGP routing table not propagating to peers",
        "DNS NXDOMAIN resolution failing for internal zones",
        "WAN MPLS link packet loss exceeding 30 percent",
        "WiFi 802.11ac access point not broadcasting SSID",
        "VLAN trunk port misconfiguration causing isolation",
        "QoS DSCP marking not prioritizing voice traffic",
        "firewall ACL blocking legitimate HTTPS traffic",
        "proxy PAC file misconfiguration blocking cloud apps",
        "spanning tree loop causing broadcast storm",
        "DHCP scope exhausted no more IP addresses",
        "NAT translation table full dropping connections",
    ]
}

# MUTUALLY EXCLUSIVE keyword sets — no overlap between categories
EXCLUSIVE_KEYWORDS = {
    "Infrastructure": [
        "bare metal", "rack mount", "psu", "dimm", "bios",
        "motherboard", "chassis", "cooling fan", "ups battery",
        "pdu", "data center rack", "hardware fault"
    ],
    "Application": [
        "null pointer", "http 500", "rest api", "apk crash",
        "heap memory", "cron job", "saml token", "build pipeline",
        "http 403", "microservice", "frontend ui", "batch job"
    ],
    "Security": [
        "ransomware", "phishing", "brute force", "ssl certificate",
        "privilege escalation", "dlp alert", "zero day", "dark web",
        "port scanning", "malware trojan", "data exfiltration",
        "credential leak"
    ],
    "Database": [
        "innodb deadlock", "tablespace", "replication lag",
        "replica set", "execution plan", "binary log",
        "stored procedure", "connection pool", "b-tree index",
        "rman backup", "tempdb", "schema migration"
    ],
    "Storage": [
        "nas raid", "san lun", "iscsi target", "tape library",
        "nfs quota", "deduplication", "snapshot clone",
        "fsck corruption", "vmotio", "s3 bucket acl",
        "thin provisioned", "cifs smb share"
    ],
    "Network": [
        "ipsec tunnel", "bgp routing", "dns nxdomain",
        "mpls packet loss", "802.11ac", "vlan trunk",
        "qos dscp", "firewall acl", "proxy pac",
        "spanning tree", "dhcp scope", "nat translation"
    ]
}

def generate_ticket(category: str, issue: str) -> dict:
    # Get exclusive keywords for this category
    keywords = ', '.join(EXCLUSIVE_KEYWORDS[category][:5])
    
    prompt = f"""Generate a realistic IT support ticket.

Issue type: {issue}
Category: {category}
Must contain these technical terms: {keywords}

Return ONLY valid JSON:
{{"title": "specific technical title using {category} terminology", "description": "2 sentences with specific {category} error details and technical terms", "category": "{category}", "resolution": "3 specific technical steps", "priority": "High"}}

No markdown. No explanation. JSON only."""

    try:
        response = ollama.chat(
            model="mistral:7b-instruct-q8_0",
            messages=[
                {
                    "role": "system",
                    "content": f"You are an IT support engineer specializing in {category}. Generate realistic ticket. Return only JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        content = response['message']['content'].strip()
        
        # Clean markdown
        if "```" in content:
            parts = content.split("```")
            for part in parts:
                if '{' in part:
                    content = part.strip()
                    break
        if content.startswith("json"):
            content = content[4:].strip()
        
        # Extract JSON
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            content = match.group()
        
        ticket = json.loads(content)
        ticket['priority'] = random.choice(PRIORITIES)
        ticket['category'] = category  # Always force correct category
        return ticket
        
    except Exception as e:
        logger.error(f"Error generating {category} ticket: {e}")
        return None

def generate_dataset(total_tickets: int = 1000):
    tickets = []
    per_category = total_tickets // len(EXCLUSIVE_KEYWORDS)
    
    logger.info(
        f"Generating {total_tickets} mutually exclusive tickets "
        f"({per_category} per category)..."
    )
    
    for category in EXCLUSIVE_KEYWORDS.keys():
        issues = PROMPTS[category]
        count = 0
        attempts = 0
        max_attempts = per_category * 3
        
        while count < per_category and attempts < max_attempts:
            issue = random.choice(issues)
            attempts += 1
            
            logger.info(
                f"{category} {count+1}/{per_category}: "
                f"{issue[:40]}"
            )
            
            ticket = generate_ticket(category, issue)
            
            if ticket:
                tickets.append(ticket)
                count += 1
        
        logger.success(f"{category}: {count} tickets done")
    
    df = pd.DataFrame(tickets)
    df = df.fillna("")
    df.to_csv("data/synthetic_tickets.csv", index=False)
    logger.success(f"Saved {len(df)} tickets")
    return df

if __name__ == "__main__":
    df = generate_dataset(1000)
    print(f"\nTotal: {len(df)}")
    print(df['category'].value_counts())