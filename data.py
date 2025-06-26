import json
import random

# Core financial terms & explanations
terms = {
    "P/E ratio": "Measures a company's share price relative to EPS. Formula: Market Value per Share / Earnings per Share.",
    "Diversification": "Spreading investments across assets to reduce risk exposure.",
    "Compound interest": "Interest earned on both principal and accumulated interest. Formula: A = P(1 + r/n)^(nt).",
    # 150+ more entries
}

# Generate 10,000 entries
dataset = []
variations = ["Define {}", "Explain {}", "What is {}?", "Describe {}", "How does {} work?"]

for _ in range(10000):
    term = random.choice(list(terms.keys()))
    instruction = random.choice(variations).format(term)
    dataset.append({
        "instruction": instruction,
        "input": "",
        "output": terms[term]
    })

# Save to JSON
with open("financial_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)