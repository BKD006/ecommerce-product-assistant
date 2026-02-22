def reasoning_prompt(user_query: str) -> str:
    return f"""
You are an intelligent ecommerce research agent.

User Query:
{user_query}

Decide the next action.

Respond ONLY with valid JSON. No markdown. No explanation.
{{
  "thought": "your reasoning",
  "action": "product_tool | policy_tool | both | final",
  "tool_query": "query to pass to tool (if needed)"
}}
"""


def reflection_prompt(user_query: str, product_results, policy_results) -> str:
    return f"""
You are reviewing gathered information.

User Query:
{user_query}

Product Results:
{product_results}

Policy Results:
{policy_results}

Should we:
- Continue gathering more info
- Or produce final answer

Respond strictly in JSON:

{{
  "decision": "continue | final"
}}
"""


def final_prompt(user_query: str, product_results, policy_results) -> str:
    return f"""
You are an ecommerce assistant.

User Query:
{user_query}

Product Results:
{product_results}

Policy Results:
{policy_results}

Provide a helpful, accurate, and concise answer.
"""