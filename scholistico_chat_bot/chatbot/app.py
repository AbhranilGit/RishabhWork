import os
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""


import os
import streamlit as st
from openai import OpenAI
import anthropic
from typing import List, Dict
import json

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


# Check if API keys are available
if not OPENAI_API_KEY:
    st.error("OpenAI API key is missing. Please set the OPENAI_API_KEY environment variable.")
    st.stop()
if not ANTHROPIC_API_KEY:
    st.error("Anthropic API key is missing. Please set the ANTHROPIC_API_KEY environment variable.")
    st.stop()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model" not in st.session_state:
    st.session_state.model = "OpenAI GPT-4"

# Initialize the API clients
try:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
except Exception as e:
    st.error(f"Error initializing API clients: {str(e)}")
    st.stop()

def get_openai_response(messages: List[Dict[str, str]], system_prompt: str) -> str:
    try:
        m1 = messages.copy()
        system_message = {"role": "system", "content": system_prompt}
        m1.insert(0, system_message)
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=m1
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error getting OpenAI response: {str(e)}")
        return "I'm sorry, but I encountered an error while processing your request."

def get_anthropic_response(messages: List[Dict[str, str]], system_prompt: str) -> str:
    try:
        formatted_messages = []
        for message in messages:
            if message['role'] == 'user':
                formatted_messages.append({"role": "user", "content": message['content']})
            else:
                formatted_messages.append({"role": "assistant", "content": message['content']})
        
        response = claude.messages.create(
            model="claude-3-sonnet-20240229",
            system=system_prompt,
            max_tokens=1000,
            messages=formatted_messages
        )
        return response.content[0].text
    except Exception as e:
        st.error(f"Error getting Anthropic response: {str(e)}")
        return "I'm sorry, but I encountered an error while processing your request."
def calculate_relevance(current_message, all_messages):
    relevant_messages = []
    
    for msg in reversed(all_messages[:-1]):  # Exclude the current message
        try:
            prompt = f"""
            Determine the relevance between the following two messages on a scale of 0 to 1, where 0 is completely irrelevant and 1 is highly relevant.
            
            Current message: {current_message}
            Previous message: {msg['content']}
            
            Provide your response as a JSON object with a single key 'relevance' and the value as a float between 0 and 1.
            """
            
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that determines the relevance between two messages."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100
            )
            
            relevance_json = json.loads(response.choices[0].message.content)
            relevance = relevance_json['relevance']
            
            if relevance >= 0.5:  # Adjust this threshold as needed
                relevant_messages.insert(0, msg)
        except Exception as e:
            st.error(f"Error calculating relevance: {str(e)}")
    
    relevant_messages.append(all_messages[-1])  # Add the current message
    return relevant_messages

def filter_relevant_messages(messages):
    if len(messages) <= 2:
        return messages

    current_message = messages[-1]['content']
    return calculate_relevance(current_message, messages)

st.title("Chat System")

# Sidebar for model selection and system prompt
with st.sidebar:
    new_model = st.selectbox("Choose AI Model", ["OpenAI GPT-4", "Anthropic Claude"])
    if new_model != st.session_state.model:
        st.session_state.model = new_model
        st.session_state.messages = []  # Clear chat history when model changes
        st.rerun()  # Rerun the app to reflect the change
    
    system_prompt = st.text_area("System Prompt (optional)", value="", help="Enter a custom system prompt or leave blank for default.")
    if not system_prompt:
        system_prompt = "You are a helpful AI assistant"

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("What is your message?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = "I'm sorry, but I encountered an error while generating a response."

        try:
            # Filter relevant messages before sending to the API
            relevant_messages = filter_relevant_messages(st.session_state.messages)

            print(relevant_messages)
            
            if st.session_state.model == "OpenAI GPT-4":
                full_response = get_openai_response(relevant_messages, system_prompt)
            else:  # Anthropic Claude
                full_response = get_anthropic_response(relevant_messages, system_prompt)
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
        
        message_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})



# Error handling for unexpected exceptions
try:
    st.empty()  # This is just to trigger any potential Streamlit-related errors
except Exception as e:
    st.error(f"An unexpected error occurred: {str(e)}")