from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.agents.state import AgentState
from app.agents.nodes.planner import planner_node
from app.agents.nodes.retriever import retrieve_node
from app.agents.nodes.responder import generate_node
from app.messages import CONVERSATIONAL, OUT_OF_SCOPE, OFF_TOPIC_REPLY


# 1. Initialize the State Graph
workflow = StateGraph(AgentState)


def refuse_node(state: AgentState):
    """
    Fixed refusal for out-of-scope or injected requests. Deliberately makes NO LLM call:
    a model that never sees the request cannot be talked into answering it.
    The refusal is still written to the conversation memory so later turns stay consistent.
    """
    return {
        "final_answer": OFF_TOPIC_REPLY,
        "documents": [],
        "status": "Refused: outside the scope of the JAFZA Rules.",
        "plan": state["plan"],
        "messages": [{"role": "assistant", "content": OFF_TOPIC_REPLY}],
    }


# 2. Define the Nodes
workflow.add_node("planner", planner_node)
workflow.add_node("retriever", retrieve_node)
workflow.add_node("responder", generate_node)
workflow.add_node("refuse", refuse_node)

# 3. Define the Edges & Routing Logic
def route_planner(state: AgentState):
    """
    Routes the workflow based on the planner's decision.
    """
    if state["current_query"] == OUT_OF_SCOPE:
        return "refuse"
    if state["current_query"] == CONVERSATIONAL:
        return "responder"
    return "retriever"

workflow.set_entry_point("planner")


# Conditional Edge: Planner -> Router -> (Retriever OR Responder OR Refuse)
workflow.add_conditional_edges(
    "planner",
    route_planner,
    {
        "retriever": "retriever",
        "responder": "responder",
        "refuse": "refuse"
    }
)


workflow.add_edge("retriever", "responder")
workflow.add_edge("responder", END)
workflow.add_edge("refuse", END)


# --- MEMORY UPGRADE ---
# MemorySaver allows the agent to remember conversations based on 'thread_id'
checkpointer = MemorySaver()


# 4. Compile the Graph with Memory
rag_agent = workflow.compile(checkpointer=checkpointer)
