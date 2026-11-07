"""
Single source of truth for the assistant's identity, scope and canned replies.

Constants only (no imports) so guardrails, agent nodes and evals can all use it
without pulling in heavy dependencies. Keep the wording here; do not duplicate
it elsewhere.
"""

ASSISTANT_NAME = "JAFZA Rules Assistant"
CORPUS_NAME = "the Jebel Ali Free Zone (JAFZA) Rules, 9th Edition"

SCOPE_SUMMARY = (
    "licences and leases, employees and labour (sponsorship, passes, overtime, termination, "
    "gratuity, WPS), inspections, violations and fines, insurance, health, safety and "
    "environment, staff accommodation, traffic and security, and the JAFZA tariff and fees"
)

# Planner sentinels (stored in AgentState.current_query).
CONVERSATIONAL = "CONVERSATIONAL"
OUT_OF_SCOPE = "OUT_OF_SCOPE"

# Canned replies. Guardrail replies are matched by the RAIL_INDICATORS substrings in
# app/guardrails/colang_rules.py, so change them together.
OFF_TOPIC_REPLY = (
    "I'm the JAFZA Rules Assistant. I can only answer questions about the Jebel Ali Free Zone "
    "Rules, so I can't help with that. Ask me about licences, leases, employees, inspections, "
    "fines, insurance, safety or fees."
)
JAILBREAK_REPLY = (
    "I keep to my guidelines however I'm prompted, and I can't set them aside. "
    "I'm here to answer questions about the Jebel Ali Free Zone Rules. What would you like to know?"
)
GREETING_REPLY = (
    "Hello! I'm the JAFZA Rules Assistant. I answer questions about the Jebel Ali Free Zone Rules: "
    "licences and leases, employees, inspections and fines, insurance, safety and fees. "
    "What can I help you with today?"
)
CAPABILITIES_REPLY = (
    "I'm the JAFZA Rules Assistant. I answer questions about the Jebel Ali Free Zone Rules, "
    "9th Edition (March 2023), covering " + SCOPE_SUMMARY + ". Ask me anything in those areas."
)
FAREWELL_REPLY = (
    "Goodbye! Come back any time you have more questions about the JAFZA Rules."
)

# Returned without calling the LLM when retrieval finds nothing to ground an answer on.
NO_CONTEXT_REPLY = (
    "I couldn't find anything in the JAFZA Rules that answers this, so I won't guess. "
    "Try rephrasing, or ask about a specific rule, fee or penalty."
)
