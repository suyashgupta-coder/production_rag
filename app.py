
import gradio as gr
from agent import get_agent_response

# Custom CSS
css = """
.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}

#header {
    text-align: center;
    padding: 20px 0;
}

#chatbot {
    border-radius: 16px;
}

#message-box {
    border-radius: 12px;
}

#footer {
    text-align: center;
    font-size: 12px;
    opacity: 0.65;
    margin-top: 20px;
}
"""

# Suggested questions
examples = [
    "What is Amazon's total revenue?",
    "Compare revenue for the past 3 years.",
    "What is the company's sales growth?",
    "Summarize the financial performance."
]


# Agent response handler
async def chat(user_message, history):
    if not user_message or not user_message.strip():
        return "", history or []

    history = list(history or [])

    # Add user message to the UI
    history.append({
        "role": "user",
        "content": user_message
    })

    try:
        # Use only previous messages as context
        previous_history = history[:-1]

        # Call existing AI agent
        response = await get_agent_response(
            user_message,
            previous_history
        )

        history.append({
            "role": "assistant",
            "content": str(response)
        })

    except Exception as e:
        history.append({
            "role": "assistant",
            "content": f"⚠️ Error: {str(e)}"
        })

    return "", history


# Clear conversation
def clear_chat():
    return [], ""


# Create application
def create_app():

    with gr.Blocks(
        title="AI Business Intelligence Assistant",
        theme=gr.themes.Soft(
            primary_hue="blue",
            neutral_hue="slate"
        ),
        css=css
    ) as demo:

        # Header
        gr.Markdown(
            """
            <div id="header">
                <h1>✦ AI Business Intelligence</h1>
                <p>Powered by Google Gemini + AWS Bedrock Knowledge Base</p>
                <p>Ask questions. Explore financial data. Get AI-powered insights.</p>
            </div>
            """
        )

        # Main chat area
        chatbot = gr.Chatbot(
            elem_id="chatbot",
            height=520,
            show_label=False,
            placeholder=(
                "👋 Welcome to your AI Business Assistant!\n\n"
                "Ask me about company revenue, sales, "
                "financial performance, or business data."
            )
        )

        # Input section
        with gr.Row():
            message = gr.Textbox(
                placeholder="Ask anything about your business data...",
                show_label=False,
                lines=2,
                max_lines=5,
                elem_id="message-box",
                scale=8
            )

            send = gr.Button(
                "➤ Send",
                variant="primary",
                scale=1,
                min_width=100
            )

        # Action buttons
        with gr.Row():
            clear = gr.Button(
                "🗑 Clear Conversation",
                variant="secondary"
            )

        # Example questions
        gr.Examples(
            examples=examples,
            inputs=message,
            label="💡 Suggested Questions"
        )

        # Footer
        gr.Markdown(
            """
            <div id="footer">
                AI Business Intelligence Assistant |
                Google Gemini • AWS Bedrock • LlamaIndex • Gradio
            </div>
            """
        )

        # Send message
        send.click(
            fn=chat,
            inputs=[message, chatbot],
            outputs=[message, chatbot],
            show_progress="full"
        )

        # Enter to send
        message.submit(
            fn=chat,
            inputs=[message, chatbot],
            outputs=[message, chatbot],
            show_progress="full"
        )

        # Clear chat
        clear.click(
            fn=clear_chat,
            outputs=[chatbot, message]
        )

    return demo


# Run application
if __name__ == "__main__":
    app = create_app()

    app.queue()

    app.launch(
        server_name="0.0.0.0",
        server_port=8080,
        share=False,
        inbrowser=True
    )
