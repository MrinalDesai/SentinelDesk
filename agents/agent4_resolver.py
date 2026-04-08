import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ollama
import json
import re
from loguru import logger
from config.settings import settings
from utils.embeddings import search_similar
from utils.database import get_connection

def get_resolution(
    title: str,
    description: str,
    category: str,
    ticket_id: int = 0
) -> dict:
    logger.info(f"Agent 4: Finding resolution for — {title[:50]}")
    
    # Search similar tickets
    query = f"{title} {description}"
    similar = search_similar(query, top_k=settings.TOP_K_RESULTS)
    
    # Build context from similar tickets
    context = ""
    sources = []
    
    if similar:
        context = "Similar past tickets and resolutions:\n"
        for i, hit in enumerate(similar):
            payload = hit.payload
            context += f"\n{i+1}. Title: {payload.get('title', '')}"
            context += f"\n   Resolution: {payload.get('resolution', '')}"
            context += f"\n   Score: {hit.score:.2f}\n"
            sources.append({
                "title": payload.get('title', ''),
                "score": hit.score,
                "resolution": payload.get('resolution', '')
            })
    else:
        context = "No similar tickets found in database."
    
    # Generate resolution
    prompt = f"""You are an IT resolution specialist.

{context}

Current ticket:
Title: {title}
Description: {description}
Category: {category}

Based on the similar tickets above, provide a resolution.

Return ONLY this JSON:
{{
    "resolution": "step by step resolution here",
    "confidence": 0.85,
    "sources_used": {len(sources)}
}}"""

    try:
        response = ollama.chat(
            model=settings.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = response['message']['content'].strip()
        
        # Clean JSON
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            content = match.group()
        
        result = json.loads(content)
        result['sources'] = sources
        
        # Log audit
        log_audit(
            ticket_id=ticket_id,
            action="RESOLUTION",
            details=f"Resolution generated using {len(sources)} similar tickets",
            confidence=result.get('confidence', 0.8)
        )
        
        logger.success(
            f"Agent 4: Resolution generated "
            f"(sources: {len(sources)})"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Agent 4 error: {e}")
        return {
            "resolution": "Please escalate to L2 support for manual resolution.",
            "confidence": 0.3,
            "sources": [],
            "sources_used": 0
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
        """, (ticket_id, "Agent4_Resolver", action, details, confidence))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"Audit log error: {e}")

if __name__ == "__main__":
    print("\n--- Agent 4 RAG Resolver Test ---")
    
    result = get_resolution(
        title="VPN not connecting after password reset",
        description="User cannot connect to corporate VPN since credentials were reset. Authentication timeout error.",
        category="Network"
    )
    
    print(f"\nResolution: {result['resolution']}")
    print(f"Confidence: {result.get('confidence', 'N/A')}")
    print(f"Sources used: {result.get('sources_used', 0)}")
    
    if result['sources']:
        print("\nSimilar tickets found:")
        for s in result['sources']:
            print(f"  - {s['title']} (score: {s['score']:.2f})")