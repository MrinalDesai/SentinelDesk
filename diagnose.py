import sys
import os
sys.path.append('.')

from agents.agent2_classifier import classify_ticket
import pandas as pd

df = pd.read_csv('data/synthetic_tickets.csv').fillna('')
samples = df.groupby('category').head(2)

for _, row in samples.iterrows():
    result = classify_ticket(row['title'], row['description'])
    true_cat = row['category']
    pred_cat = result['category']
    match = 'OK' if pred_cat == true_cat else 'WRONG'
    print(f"{match} | True: {true_cat:15} | Pred: {pred_cat:15} | {row['title'][:40]}")