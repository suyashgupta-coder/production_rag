
from dotenv import load_dotenv
import os
import asyncio
import boto3

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.llms import ChatMessage


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
KNOWLEDGE_BASE_ID = os.getenv("BEDROCK_KNOWLEDGE_BASE_ID")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-3.1-flash-lite")


# ============================================================
# CONFIGURATION
# ============================================================

print("========== APPLICATION CONFIGURATION ==========")
print(f"AWS Region       : {AWS_REGION}")
print(f"Knowledge Base   : {KNOWLEDGE_BASE_ID}")
print(f"Google Model     : {GOOGLE_MODEL}")
print(f"Google API Key   : {'Loaded' if GOOGLE_API_KEY else 'NOT FOUND'}")
print("================================================")


if not KNOWLEDGE_BASE_ID:
    raise ValueError(
        "BEDROCK_KNOWLEDGE_BASE_ID environment variable is missing."
    )

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY environment variable is missing."
    )


# ============================================================
# AWS BEDROCK CLIENT
# ============================================================

bedrock_client = boto3.client(
    "bedrock-agent-runtime",
    region_name=AWS_REGION
)


# ============================================================
# KNOWLEDGE BASE SEARCH
# ============================================================

def search_knowledge_base(query: str) -> str:

    print("\n========================================")
    print("KNOWLEDGE BASE SEARCH")
    print("========================================")
    print(f"Query: {query}")

    try:
        response = bedrock_client.retrieve(
            knowledgeBaseId=KNOWLEDGE_BASE_ID,
            retrievalQuery={
                "text": query
            },
            retrievalConfiguration={
                "managedSearchConfiguration": {
                    "numberOfResults": 5
                }
            }
        )

        print("Bedrock retrieve() completed successfully.")

        results = response.get("retrievalResults", [])

        print(f"Number of results: {len(results)}")

        if not results:
            return "No relevant information found in the knowledge base."

        documents = []

        for index, result in enumerate(results, start=1):

            content = result.get("content", {})
            text = content.get("text", "")

            if text:
                print(f"Result {index}: {text[:200]}...")
                documents.append(text)

        if not documents:
            return "The knowledge base returned no readable text."

        final_context = "\n\n".join(documents)

        print("Knowledge Base search successful.")

        return final_context

    except Exception as e:

        error_type = type(e).__name__
        error_message = str(e)

        print("\n========================================")
        print("BEDROCK ERROR")
        print("========================================")
        print(f"Error Type    : {error_type}")
        print(f"Error Message : {error_message}")
        print("========================================")

        raise


# ============================================================
# CREATE KNOWLEDGE BASE TOOL
# ============================================================

knowledge_base_tool = FunctionTool.from_defaults(
    fn=search_knowledge_base,
    name="amazon_knowledge_base",
    description="""
    Search the AWS Bedrock Knowledge Base.

    Use this tool whenever the user asks about:
    - Company information
    - Financial information
    - Revenue
    - Sales
    - Business data
    - Historical financial data
    - Information stored in the Knowledge Base

    Do not invent information.
    """
)


# ============================================================
# GOOGLE GEMINI
# ============================================================

llm = GoogleGenAI(
    model=GOOGLE_MODEL,
    api_key=GOOGLE_API_KEY
)


# ============================================================
# REACT AGENT
# ============================================================

agent = ReActAgent(
    tools=[knowledge_base_tool],
    llm=llm,
    system_prompt="""
    You are a business and financial AI assistant.

    Your job is to answer questions using the AWS
    Bedrock Knowledge Base.

    IMPORTANT RULES:

    1. Use the amazon_knowledge_base tool whenever
       the user asks about company, financial,
       revenue, sales, or business information.

    2. Do not invent financial information.

    3. Use only information returned by the
       Knowledge Base for company-specific answers.

    4. If the Knowledge Base does not contain
       relevant information, clearly say so.

    5. If the Knowledge Base tool fails, explain
       that the information could not be retrieved.

    6. Give clear and concise answers.

    7. When presenting multiple years of financial
       information, organize the information clearly.
    """
)


# ============================================================
# CONVERT GRADIO CHAT HISTORY TO LLAMAINDEX
# ============================================================

def convert_chat_history(chat_history):

    messages = []

    for item in chat_history or []:

        # Already a LlamaIndex message
        if isinstance(item, ChatMessage):
            messages.append(item)
            continue

        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content", "")

        if role not in ("user", "assistant"):
            continue

        # Gradio may supply text as a dictionary
        if isinstance(content, dict):
            content = content.get("text", "")

        # Gradio may supply a list of text blocks
        if isinstance(content, list):
            content = "\n".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict)
                and isinstance(block.get("text"), str)
            )

        if not isinstance(content, str):
            continue

        messages.append(
            ChatMessage(
                role=role,
                content=content
            )
        )

    return messages


# ============================================================
# MAIN FUNCTION USED BY GRADIO
# ============================================================

async def get_agent_response(
    message: str,
    chat_history=None
) -> str:

    print("\n========================================")
    print("USER REQUEST")
    print("========================================")
    print(message)

    try:

        # Convert Gradio history into LlamaIndex messages
        history = convert_chat_history(chat_history)

        # Run ReActAgent
        response = await agent.run(
            user_msg=message,
            chat_history=history
        )

        final_response = str(response)

        print("\n========================================")
        print("AGENT RESPONSE")
        print("========================================")
        print(final_response)

        return final_response

    except Exception as e:

        error_type = type(e).__name__
        error_message = str(e)

        print("\n========================================")
        print("AGENT ERROR")
        print("========================================")
        print(f"Error Type    : {error_type}")
        print(f"Error Message : {error_message}")
        print("========================================")

        return (
            "I encountered an error while processing your request. "
            "Please check the application logs for details."
        )


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    test_question = "Tell me about the revenue of past years"

    result = asyncio.run(
        get_agent_response(test_question)
    )

    print("\nFINAL RESPONSE:")
    print(result)
