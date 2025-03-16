
import os
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""

import os
from typing import List, Dict
import json
from openai import OpenAI
import anthropic

class ChatAPI:
    def __init__(self):
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
        
        if not self.OPENAI_API_KEY:
            raise ValueError("OpenAI API key is missing. Please set the OPENAI_API_KEY environment variable.")
        if not self.ANTHROPIC_API_KEY:
            raise ValueError("Anthropic API key is missing. Please set the ANTHROPIC_API_KEY environment variable.")
        
        try:
            self.openai_client = OpenAI(api_key=self.OPENAI_API_KEY)
            self.claude = anthropic.Anthropic(api_key=self.ANTHROPIC_API_KEY)
        except Exception as e:
            raise Exception(f"Error initializing API clients: {str(e)}")
        
        self.messages = []
        self.model = "OpenAI GPT-4"

    def get_openai_response(self, messages: List[Dict[str, str]], system_prompt: str) -> str:
        try:
            m1 = messages.copy()
            system_message = {"role": "system", "content": system_prompt}
            m1.insert(0, system_message)
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=m1
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Error getting OpenAI response: {str(e)}")

    def get_anthropic_response(self, messages: List[Dict[str, str]], system_prompt: str) -> str:
        try:
            formatted_messages = []
            for message in messages:
                if message['role'] == 'user':
                    formatted_messages.append({"role": "user", "content": message['content']})
                else:
                    formatted_messages.append({"role": "assistant", "content": message['content']})
            
            response = self.claude.messages.create(
                model="claude-3-sonnet-20240229",
                system=system_prompt,
                max_tokens=1000,
                messages=formatted_messages
            )
            return response.content[0].text
        except Exception as e:
            raise Exception(f"Error getting Anthropic response: {str(e)}")

    def calculate_relevance(self, current_message, all_messages):
        relevant_messages = []
        
        for msg in reversed(all_messages[:-1]):  # Exclude the current message
            try:
                prompt = f"""
                Determine the relevance between the following two messages on a scale of 0 to 1, where 0 is completely irrelevant and 1 is highly relevant.
                
                Current message: {current_message}
                Previous message: {msg['content']}
                
                Provide your response as a JSON object with a single key 'relevance' and the value as a float between 0 and 1.
                """
                
                response = self.openai_client.chat.completions.create(
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
                raise Exception(f"Error calculating relevance: {str(e)}")
        
        relevant_messages.append(all_messages[-1])  # Add the current message
        return relevant_messages

    def filter_relevant_messages(self, messages):
        if len(messages) <= 2:
            return messages

        current_message = messages[-1]['content']
        return self.calculate_relevance(current_message, messages)

    def set_model(self, model: str):
        if model not in ["OpenAI GPT-4", "Anthropic Claude"]:
            raise ValueError("Invalid model selection. Choose 'OpenAI GPT-4' or 'Anthropic Claude'.")
        self.model = model
        self.messages = []  # Clear chat history when model changes

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def get_response(self, system_prompt: str = "You are a helpful AI assistant"):
        try:
            relevant_messages = self.filter_relevant_messages(self.messages)

            print(relevant_messages)
            
            if self.model == "OpenAI GPT-4":
                return self.get_openai_response(relevant_messages, system_prompt)
            else:  # Anthropic Claude
                return self.get_anthropic_response(relevant_messages, system_prompt)
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")

# Example usage:
if __name__ == "__main__":
    chat_api = ChatAPI()
    
    # Set the model
    chat_api.set_model("OpenAI GPT-4")
    
    # Add a user message
    chat_api.add_message("user", "What is the capital of France?")
    
    # Get a response
    try:
        response = chat_api.get_response()
        print("AI:", response)
    except Exception as e:
        print(f"Error: {str(e)}")