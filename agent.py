#                  USER
#                    │
#                    ▼
#           get_agent_response()
#                    │
#                    ▼
#              ReActAgent
#                    │
#             ┌──────┴──────┐
#             │             │
#       Need information?   No
#             │             │
#            Yes            │
#             ▼             │
#    Bedrock Knowledge Base │
#             │             │
#        Relevant data      │
#             └──────┬──────┘
#                    ▼
#              Google Gemini
#                    │
#                    ▼
#                 Answer
#                    │
#                    ▼
#                  USER

# Think of it like this:
# Gemini = Brain
# Bedrock Knowledge Base = Library
# ReActAgent = Manager
# LlamaIndex = Bridge connecting everything
# Gradio = UI, if you later connect this to Gradio


#|      Component                 |           Job                   |
# | **ReActAgent**                 |   Decides whether to use the tool |
# | **QueryEngineTool**            | Makes the query engine available as an agent tool |
# | **QueryEngine**                | Provides the querying interface/workflow |
# | **Retriever**                  | Finds relevant information |
# | **Bedrock Knowledge Base**       | Contains/indexes your knowledge | 
# 
# Easy analogy 📚
# Imagine a library:
# You → Librarian → Search System → Books

# Mapping to your code:
# User
#  ↓
# ReActAgent          = Librarian
#  ↓
# QueryEngineTool     = Library service
#  ↓
# QueryEngine         = Search interface
#  ↓
# Retriever           = Search mechanism
#  ↓
# Bedrock Knowledge Base = Library 


# from dotenv import load_dotenv
# load_dotenv()

# import os
# import boto3

# from llama_index.llms.google_genai import GoogleGenAI
# from llama_index.core.tools import FunctionTool
# from llama_index.core.agent.workflow import ReActAgent


# # Configuration
# AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
# KNOWLEDGE_BASE_ID = os.getenv("BEDROCK_KNOWLEDGE_BASE_ID")
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-3.1-flash-lite")


# # AWS Bedrock client
# bedrock_client = boto3.client(
#     "bedrock-agent-runtime",
#     region_name=AWS_REGION
# )


# # Knowledge Base search
# def search_knowledge_base(query: str) -> str:
#     try:
#         response = bedrock_client.retrieve(
#             knowledgeBaseId=KNOWLEDGE_BASE_ID,
#             #we can change managedSearchConfiguration into vectorSearchConfiguration when we using vector engine   
#             retrievalConfiguration={
#                 "managedSearchConfiguration": {
#                     "numberOfResults": 5
#                 }
#             },
#             retrievalQuery={
#                 "text": query
#             }
#         )

#         results = response.get("retrievalResults", [])

#         if not results:
#             return "No relevant information found."

#         return "\n\n".join(
#             result.get("content", {}).get("text", "")
#             for result in results
#         )

#     except Exception as e:
#         return f"Knowledge base search failed: {str(e)}"


# # Create Knowledge Base tool
# knowledge_base_tool = FunctionTool.from_defaults(
#     fn=search_knowledge_base,
#     name="amazon_knowledge_base",
#     description="""
#     Search the AWS Bedrock Knowledge Base for company,
#     financial, revenue, sales, and business information.
#     Use this tool whenever the user asks for information
#     that should come from the knowledge base.
#     """
# )


# # Gemini
# llm = GoogleGenAI(
#     model=GOOGLE_MODEL,
#     api_key=GOOGLE_API_KEY
# )


# # ReAct Agent
# agent = ReActAgent(
#     tools=[knowledge_base_tool],
#     llm=llm,
#     system_prompt="""
#     You are a business and financial AI assistant.

#     Use the amazon_knowledge_base tool whenever the
#     user asks about company or financial information.

#     Do not invent information. Base your answers on
#     the retrieved knowledge base data.

#     Give clear and concise answers.
#     """
# )


# # Main function used by Gradio
# async def get_agent_response(message: str, chat_history=None) -> str:
#     try:
#         response = await agent.run(
#             user_msg=message,
#             chat_history=chat_history or []
#         )

#         return str(response)

#     except Exception as e:
#         return f"Error: {str(e)}"


# # Test
# if __name__ == "__main__":
#     import asyncio

#     response = asyncio.run(
#         get_agent_response(
#             "What is Amazon total revenue?"
#         )
#     )

#     print(response)


from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import os
import boto3

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import ReActAgent


# ============================================================
# CONFIGURATION
# ============================================================

AWS_REGION = os.getenv(
    "AWS_DEFAULT_REGION",
    "us-east-1"
)

KNOWLEDGE_BASE_ID = os.getenv(
    "BEDROCK_KNOWLEDGE_BASE_ID"
)

GOOGLE_API_KEY = os.getenv(
    "GOOGLE_API_KEY"
)

GOOGLE_MODEL = os.getenv(
    "GOOGLE_MODEL",
    "gemini-3.1-flash-lite"
)


# ============================================================
# VALIDATE ENVIRONMENT VARIABLES
# ============================================================

print("========== APPLICATION CONFIGURATION ==========")
print(f"AWS Region       : {AWS_REGION}")
print(f"Knowledge Base   : {KNOWLEDGE_BASE_ID}")
print(f"Google Model     : {GOOGLE_MODEL}")

if GOOGLE_API_KEY:
    print("Google API Key   : Loaded")
else:
    print("Google API Key   : NOT FOUND")

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

        # ----------------------------------------------------
        # Call AWS Bedrock Knowledge Base
        # ----------------------------------------------------

        response = bedrock_client.retrieve(
            knowledgeBaseId=KNOWLEDGE_BASE_ID,

            retrievalConfiguration={
                "managedSearchConfiguration": {
                    "numberOfResults": 5
                }
            },

            retrievalQuery={
                "text": query
            }
        )

        print("Bedrock retrieve() completed successfully.")

        # ----------------------------------------------------
        # Get results
        # ----------------------------------------------------

        results = response.get(
            "retrievalResults",
            []
        )

        print(f"Number of results: {len(results)}")

        # ----------------------------------------------------
        # No results
        # ----------------------------------------------------

        if not results:

            print("No relevant information found.")

            return (
                "No relevant information was found "
                "in the knowledge base."
            )

        # ----------------------------------------------------
        # Extract text
        # ----------------------------------------------------

        documents = []

        for index, result in enumerate(results, start=1):

            content = result.get(
                "content",
                {}
            )

            text = content.get(
                "text",
                ""
            )

            if text:

                print(
                    f"Result {index}: "
                    f"{text[:200]}..."
                )

                documents.append(text)

        # ----------------------------------------------------
        # Final context
        # ----------------------------------------------------

        if not documents:

            return (
                "The knowledge base returned results, "
                "but no readable text was found."
            )

        final_context = "\n\n".join(
            documents
        )

        print("Knowledge Base search successful.")

        return final_context

    except Exception as e:

        # ----------------------------------------------------
        # IMPORTANT:
        # Print the real AWS error to CloudWatch
        # ----------------------------------------------------

        error_type = type(e).__name__
        error_message = str(e)

        print("\n========================================")
        print("BEDROCK ERROR")
        print("========================================")
        print(f"Error Type    : {error_type}")
        print(f"Error Message : {error_message}")
        print("========================================\n")

        # TEMPORARY:
        # Return the real error to the chatbot
        return (
            f"BEDROCK ERROR\n"
            f"Error Type: {error_type}\n"
            f"Error Message: {error_message}"
        )


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
    - Other information stored in the Knowledge Base

    Always use this tool when the answer should come
    from the company's Knowledge Base.

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
    tools=[
        knowledge_base_tool
    ],

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

    3. Use only the information returned by the
       Knowledge Base.

    4. If the Knowledge Base does not contain
       relevant information, clearly say that.

    5. If the Knowledge Base tool returns an error,
       show the error clearly to help diagnose
       the problem.

    6. Give clear and concise answers.

    7. When presenting multiple years of financial
       information, organize the information clearly.
    """
)


# ============================================================
# MAIN FUNCTION
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

        response = await agent.run(
            user_msg=message,
            chat_history=chat_history or []
        )

        print("\n========================================")
        print("AGENT RESPONSE")
        print("========================================")
        print(str(response))

        return str(response)

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
            f"AGENT ERROR\n"
            f"Error Type: {error_type}\n"
            f"Error Message: {error_message}"
        )


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    import asyncio

    test_question = (
        "Tell me about the revenue of past years"
    )

    result = asyncio.run(
        get_agent_response(
            test_question
        )
    )

    print("\nFINAL RESPONSE:")
    print(result)