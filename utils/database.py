import psycopg2
from psycopg2.extras import RealDictCursor
from loguru import logger
from config.settings import settings

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_connection():
    return psycopg2.connect(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        dbname=settings.POSTGRES_DB,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD
    )

def setup_database():
    logger.info("Setting up PostgreSQL database...")
    
    conn = get_connection()
    cur = conn.cursor()
    
    # Tickets table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            id SERIAL PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            category VARCHAR(50),
            priority VARCHAR(20),
            status VARCHAR(20) DEFAULT 'open',
            confidence_score FLOAT,
            assigned_dept VARCHAR(100),
            resolution TEXT,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        )
    """)
    
    # RBAC routing rules table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS routing_rules (
            id SERIAL PRIMARY KEY,
            category VARCHAR(50) NOT NULL,
            priority VARCHAR(20) NOT NULL,
            department VARCHAR(100) NOT NULL,
            escalation_contact VARCHAR(100)
        )
    """)
    
    # Audit log table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id SERIAL PRIMARY KEY,
            ticket_id INTEGER,
            agent_name VARCHAR(50),
            action TEXT,
            details TEXT,
            confidence_score FLOAT,
            timestamp TIMESTAMP DEFAULT NOW()
        )
    """)
    
    # Insert RBAC routing rules
    cur.execute("DELETE FROM routing_rules")
    
    rules = [
        ("Infrastructure", "Critical", "Infrastructure Team", "infra-lead@company.com"),
        ("Infrastructure", "High",     "Infrastructure Team", "infra@company.com"),
        ("Infrastructure", "Medium",   "Infrastructure Team", "infra@company.com"),
        ("Infrastructure", "Low",      "Infrastructure Team", "infra@company.com"),
        ("Application",    "Critical", "App Dev Team",        "appdev-lead@company.com"),
        ("Application",    "High",     "App Dev Team",        "appdev@company.com"),
        ("Application",    "Medium",   "App Dev Team",        "appdev@company.com"),
        ("Application",    "Low",      "App Dev Team",        "appdev@company.com"),
        ("Security",       "Critical", "CISO Team",           "ciso@company.com"),
        ("Security",       "High",     "CISO Team",           "ciso@company.com"),
        ("Security",       "Medium",   "Security Team",       "security@company.com"),
        ("Security",       "Low",      "Security Team",       "security@company.com"),
        ("Database",       "Critical", "DBA Team",            "dba-lead@company.com"),
        ("Database",       "High",     "DBA Team",            "dba@company.com"),
        ("Database",       "Medium",   "DBA Team",            "dba@company.com"),
        ("Database",       "Low",      "DBA Team",            "dba@company.com"),
        ("Storage",        "Critical", "Storage Team",        "storage-lead@company.com"),
        ("Storage",        "High",     "Storage Team",        "storage@company.com"),
        ("Storage",        "Medium",   "Storage Team",        "storage@company.com"),
        ("Storage",        "Low",      "Storage Team",        "storage@company.com"),
        ("Network",        "Critical", "NOC Team",            "noc-lead@company.com"),
        ("Network",        "High",     "NOC Team",            "noc@company.com"),
        ("Network",        "Medium",   "NOC Team",            "noc@company.com"),
        ("Network",        "Low",      "NOC Team",            "noc@company.com"),
    ]
    
    cur.executemany("""
        INSERT INTO routing_rules 
        (category, priority, department, escalation_contact)
        VALUES (%s, %s, %s, %s)
    """, rules)
    
    conn.commit()
    cur.close()
    conn.close()
    
    logger.success("Database setup complete!")

if __name__ == "__main__":
    setup_database()