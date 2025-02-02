from langgraph.graph import END,START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition
from RAG import query_or_respond,tools,generate

graph_builder = StateGraph(MessagesState)

graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

graph_builder.set_entry_point("query_or_respond")
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
graph_builder.add_edge("tools","generate")
graph_builder.add_edge("generate", END)

graph = graph_builder.compile()

input_message = "tell me what are transfromers"

for step in graph.stream(
    {"messages": [{"role": "user", "content": input_message}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

