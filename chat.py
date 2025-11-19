import os
from dotenv import load_dotenv
import gradio
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))




def initialize_messages():
    return [{
        "role": "system",
        "content": """You are a highly skilled culinary expert with deep
        knowledge of global and Indian cuisines. Your role is to assist
        people by providing clear, accurate, and easy-to-follow food
        recipes, cooking tips, ingredient substitutions, and professional
        culinary guidance."""
    }]

def customLLMBot(user_input, history):
    messages = initialize_messages()
    
    # Reconstruct messages from history
    for msg in history:
        messages.append({"role": "user", "content": msg[0]})
        messages.append({"role": "assistant", "content": msg[1]})
    
    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        messages=messages,
        model="llama-3.3-70b-versatile",
    )
    
    LLM_reply = response.choices[0].message.content
    return LLM_reply

iface = gradio.ChatInterface(
    customLLMBot,
    chatbot=gradio.Chatbot(height=300),
    textbox=gradio.Textbox(placeholder="Ask me anything about cooking or recipes"),
    title="Food Recipe ChatBot",
    description="Your personal assistant for recipes, cooking tips, and ingredients",
    theme="soft",
    examples=[
        "Hi",
        "Give me an easy chicken curry recipe",
        "How to make soft chapathi?",
        "Best substitute for fresh cream?"
    ]
)

if __name__ == "__main__":
    iface.launch()