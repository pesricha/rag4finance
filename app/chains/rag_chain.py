from typing import TypedDict, Annotated, Sequence
import operator
import logging

from langgraph.graph import StateGraph
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SmartChatAgent:
    """
    A smart chat agent that breaks down user questions into multiple steps internally,
    reasons through them, and returns a final answer — while maintaining natural chat history.
    Intermediate reasoning steps are NOT saved in the chat history.
    """

    def __init__(
        self,
        model_name="qwen:1.8b",
        temperature=0.3,
        max_chat_history=4,
        max_plan_steps=4,
    ):
        self.max_chat_history = max_chat_history
        self.max_plan_steps = max_plan_steps
        self.llm = ChatOllama(model=model_name, temperature=temperature)
        self._history = []  # Stores only user + final AI messages
        self._build_graph()

    def _build_graph(self):
        """Build the internal reasoning graph."""

        class AgentState(TypedDict):
            task: str
            context: str
            plan: str
            steps: Annotated[Sequence[str], operator.add]
            results: Annotated[Sequence[str], operator.add]
            final_answer: str

        def plan_task(state: AgentState):
            prompt = ChatPromptTemplate.from_template(
                """
        You are a helpful assistant. Use the conversation history to understand context.

        Previous Conversation:
        {context}

        Current Task: {task}

        Break this task into clear, logical steps. Just list the steps, do not answer yet.
        Maximum Steps Allowed: {max_steps}
                """
            )
            chain = prompt | self.llm | StrOutputParser()
            plan = chain.invoke(
                {
                    "task": state["task"],
                    "context": state["context"],
                    "max_steps": self.max_plan_steps,  # Pass max steps as input
                }
            )
            steps = [line.strip() for line in plan.splitlines() if line.strip()]

            # Limit steps to max_plan_steps
            steps = steps[: self.max_plan_steps]

            logger.info("🧠 Plan Generated:")
            for i, step in enumerate(steps, 1):
                logger.info(f"  {i}. {step}")
            return {"plan": plan, "steps": steps}

        def execute_step(state: AgentState):
            current_step_index = len(state["results"])
            current_step = state["steps"][current_step_index]
            logger.info(f"🛠️ Executing Step {current_step_index + 1}: {current_step}")
            prompt = ChatPromptTemplate.from_template(
                "Perform this step: {step}\nProvide a concise and accurate result."
            )
            chain = prompt | self.llm | StrOutputParser()
            result = chain.invoke({"step": current_step})
            logger.info(f"✅ Result: {result}")
            return {"results": [result]}

        def should_continue(state: AgentState):
            return "continue" if len(state["results"]) < len(state["steps"]) else "end"

        def finalize_answer(state: AgentState):
            logger.info("📝 Finalizing Answer...")
            prompt = ChatPromptTemplate.from_template(
                """
Original Task: {task}

Steps Taken:
{steps}

Results:
{results}

Write a clear, natural, and helpful final response to the user.
                """
            )
            chain = prompt | self.llm | StrOutputParser()
            final_answer = chain.invoke(
                {
                    "task": state["task"],
                    "steps": "\n".join(state["steps"]),
                    "results": "\n".join(state["results"]),
                }
            )
            logger.info("✅ Final Answer Generated.")
            return {"final_answer": final_answer}

        # Build graph
        workflow = StateGraph(AgentState)
        workflow.add_node("plan_task", plan_task)
        workflow.add_node("execute_step", execute_step)
        workflow.add_node("finalize_answer", finalize_answer)

        workflow.set_entry_point("plan_task")
        workflow.add_edge("plan_task", "execute_step")
        workflow.add_conditional_edges(
            "execute_step",
            should_continue,
            {"continue": "execute_step", "end": "finalize_answer"},
        )
        workflow.add_edge("finalize_answer", "__end__")

        self.graph = workflow.compile()

    def _get_context(self) -> str:
        """Format chat history as plain text context."""
        return "\n".join(
            [
                f"User: {msg.content}"
                if isinstance(msg, HumanMessage)
                else f"Assistant: {msg.content}"
                for msg in self._history[-self.max_chat_history :]
            ]
        )

    def answer_user_query(self, question: str) -> str:
        """
        Answer the user's question. Returns the final answer after multi-step reasoning.
        Only the user question and final answer are saved in chat history.
        """
        # Add user message to internal history
        self._history.append(HumanMessage(content=question))

        # Run reasoning with context
        result = self.graph.invoke(
            {
                "task": question,
                "context": self._get_context(),
                "steps": [],
                "results": [],
                "plan": "",
                "final_answer": "",
            }
        )

        final_answer = result["final_answer"]

        # Save only the final answer (not intermediate steps)
        self._history.append(AIMessage(content=final_answer))

        return final_answer


# ========================
# Example Usage
# ========================
if __name__ == "__main__":
    logger.info("🧠 Initializing Smart Chat Agent...")
    agent = SmartChatAgent(
        model_name="qwen:1.8b", max_chat_history=2, max_plan_steps=2
    )  # Change to "llama3" or others if needed

    # First question
    logger.info("User: What is the capital of France?")
    response1 = agent.answer_user_query("What is the capital of France?")
    logger.info(f"Agent: {response1}")

    # Second (follow-up)
    logger.info("User: What was the previous question?")
    response2 = agent.answer_user_query("What was the previous question?")
    logger.info(f"Agent: {response2}")

    # Third (comparison)
    logger.info("User: Which country has more influence on art and design?")
    response3 = agent.answer_user_query("Which country has more influence on art and design?")
    logger.info(f"Agent: {response3}")

    # Optional: print full chat history
    logger.info("--- Full Chat History (Only user + final answers) ---")
    for msg in agent._history:
        role = "👤 User" if isinstance(msg, HumanMessage) else "🤖 Agent"
        logger.info(f"{role}: {msg.content}")
