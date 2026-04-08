import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from loguru import logger
from utils.embeddings import setup_collection, store_ticket
from utils.database import get_connection

def clean_value(val, default=""):
    if val is None:
        return default
    if isinstance(val, float):
        import math
        if math.isnan(val):
            return default
    return str(val).strip()

def ingest_tickets():
    logger.info("Starting ticket ingestion...")
    
    # Setup Qdrant collection
    setup_collection()
    
    # Load CSV
    df = pd.read_csv("data/synthetic_tickets.csv")
    
    # Clean NaN values
    df = df.fillna("")
    logger.info(f"Loaded {len(df)} tickets")
    
    # Store each ticket
    success = 0
    for idx, row in df.iterrows():
        try:
            title       = clean_value(row['title'])
            description = clean_value(row['description'])
            category    = clean_value(row['category'], "Infrastructure")
            resolution  = clean_value(row['resolution'])
            priority    = clean_value(row['priority'], "Medium")
            
            # Skip empty tickets
            if not title or not description:
                logger.warning(f"Skipping empty ticket {idx}")
                continue
            
            # Combined text for embedding
            text = f"{title} {description}"
            
            metadata = {
                "title":       title,
                "description": description,
                "category":    category,
                "resolution":  resolution,
                "priority":    priority
            }
            
            store_ticket(
                ticket_id=idx + 1,
                text=text,
                metadata=metadata
            )
            success += 1
            logger.info(f"Ingested {success}/{len(df)} — {title[:40]}")
            
        except Exception as e:
            logger.error(f"Error ingesting ticket {idx}: {e}")
    
    logger.success(f"Ingested {success}/{len(df)} tickets into Qdrant")

if __name__ == "__main__":
    ingest_tickets()