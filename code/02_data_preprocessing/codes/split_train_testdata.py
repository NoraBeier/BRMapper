import csv
import numpy as np
import sys
from sklearn.model_selection import train_test_split


filename = sys.argv[1]
test_size = 0.2  
random_state = 42 

with open(filename, mode='r', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    rows = list(reader)

header = rows[0]
rows = rows[1:]

cluster_ids = [row[4] for row in rows]


train_rows, test_rows = train_test_split(
    rows,
    test_size=test_size,
    random_state=random_state,
    stratify=cluster_ids
)

print(f"Gesamt: {len(rows)} Zeilen")
print(f"Training: {len(train_rows)} Zeilen")
print(f"Test: {len(test_rows)} Zeilen")


train_filename = sys.argv[2]
test_filename = sys.argv[3]

with open(train_filename, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter=';')
    writer.writerow(header)
    writer.writerows(train_rows)

with open(test_filename, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter=';')
    writer.writerow(header)
    writer.writerows(test_rows)

print(f"Trainingsdaten gespeichert unter: {train_filename}")
print(f"Testdaten gespeichert unter: {test_filename}")

