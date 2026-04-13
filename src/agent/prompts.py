def intent_classifier(user_query: str) -> str:
  return f"""
    You are an ecommerce query parser.

    Extract structured information.

    Return ONLY JSON:

    {{
    "intent": "search | policy",
    "category": null or string,
    "brand": null or string,
    "price_min": null or number,
    "price_max": null or number
    }}

    ---

    Rules:

    INTENT DEFINITIONS:

    1. search

    * The user is looking for specific products, recommendations, or discovery.
    * These queries usually describe:

    * preferences (cheap, best, top, good)
    * product types (shoes, laptops, phones)
    * brands combined with products (Nike shoes, Samsung phone)
    * The user expects actual product results.

    Examples:

    * "best running shoes"
    * "cheap laptops under 50000"
    * "nike sneakers"
    * "good headphones for gym"

    ---

    2. policy

    * The user is asking about business rules, policies, or procedures.
    * These queries are NOT about products.

    Examples:

    * "return policy"
    * "refund rules"
    * "shipping time"
    * "warranty details"
    * "exchange policy"

    PRICE:

    * "under 5000" → price_max = 5000
    * "below 2000" → price_max = 2000
    * "above 10000" → price_min = 10000
    * "between 1000 and 5000" → both
    * "50k" → 50000

    ---

    Query:
    "{user_query}"
    """


def final_prompt(query, products, policies):

    return f"""
You are a helpful and intelligent shopping assistant.

Your goal is to answer the user query in a clear, descriptive, and user-friendly way.

---

User Query:
{query}

---

Available Product Information:
{products}

Policy Information:
{policies}

---

Instructions:

1. DO NOT just list raw data.
2. Summarize the results in natural language.
3. Explain what is available and highlight key insights.
4. If multiple products exist:
   - Group them logically
   - Mention price ranges or differences
5. Avoid repeating the same brand unnecessarily.
6. If the query is about:
   - "cheapest" → highlight lowest price clearly
   - "average" → explain average clearly
   - "brands" → list brands naturally in a sentence
7. If no relevant data is found:
   - Politely suggest how to refine the query
8. Keep the tone conversational and helpful.

---

Answer:
"""

def evaluation_prompt(query: str, answer: str, context: str) -> str:

    context = context[:4000]  # safe truncation

    return f"""
You are an expert evaluator for RAG systems.

Query:
{query}

Context:
{context}

Answer:
{answer}

Evaluate:
1. Is the answer grounded in the context?
2. Does it contain hallucinated or unsupported claims?

Return STRICT JSON ONLY. No extra text.

{{
  "score": float (0 to 1),
  "verdict": "grounded" or "hallucinated",
  "reason": "short explanation"
}}
"""