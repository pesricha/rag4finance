from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

#TODO: Integrate chat history
def get_ollama_response(user_query: str, chat_history: list):
    """
    Generates a response from the Ollama model.
    """
    # For now, we will use a simple prompt template.
    # We can enhance this later to include conversation history.
    template = """
    You are a helpful assistant. Answer the following question.

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    # Initialize the Ollama model
    # Make sure you have a running Ollama instance.
    # You can pull models using `ollama pull <model_name>`
    llm = ChatOllama(model="qwen:1.8b")

    # Create the chain
    chain = prompt | llm | StrOutputParser()

    # Invoke the chain with the user query
    response = chain.invoke({"question": user_query})

    return response


if __name__ == "__main__":
    # Example usage
    user_query = "What is the capital of France?"
    chat_history = []  # This can be expanded to include previous messages
    response = get_ollama_response(user_query, chat_history)
    print(f"Response from Ollama: {response}")
