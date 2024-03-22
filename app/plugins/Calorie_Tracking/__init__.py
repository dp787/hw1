import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.commands import Command

class MacroTrackingAgent(Command):
    def __init__(self):
        super().__init__()
        self.name = "calorie_tracking"
        self.description = "This agent tracks macro usage and logs interactions"
        self.history = []
        self.log_file = "interaction_log.txt"  # Log file to record interactions
        load_dotenv()
        API_KEY = os.getenv('OPENAI_API_KEY')
        # Initialize the OpenAI chat model
        self.llm = ChatOpenAI(openai_api_key=API_KEY)

    def calculate_tokens(self, text):
        # Calculate the number of tokens used based on the length of the text
        return len(text.split())

    def log_interaction(self, user_input, response, tokens_used):
        # Log the interaction into a file
        with open(self.log_file, "a") as f:
            f.write(f"User: {user_input}\n")
            f.write(f"Agent: {response}\n")
            f.write(f"Tokens Used: {tokens_used}\n\n")

    def interact_with_ai(self, user_input):
        # Generate a prompt and interact with the OpenAI chat model
        prompt_text = "You're a macro tracking agent. What can I assist you with today?"
        prompt = ChatPromptTemplate.from_messages(self.history + [("system", prompt_text)])
        
        output_parser = StrOutputParser()
        chain = prompt | self.llm | output_parser

        response = chain.invoke({"input": user_input})

        tokens_used = self.calculate_tokens(prompt_text + user_input + response)
        logging.info(f"OpenAI API call made. Tokens used: {tokens_used}")
        
        return response, tokens_used

    def execute(self, *args, **kwargs):
        print("Welcome to the Macro Tracking Agent. How can I assist you today?")
        
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() == "done":
                print("Thank you for using the Macro Tracking Agent. Goodbye!")
                break

            self.history.append(("user", user_input))
            
            try:
                response, tokens_used = self.interact_with_ai(user_input)
                print(f"Agent: {response}")
                print(f"(This interaction used {tokens_used} tokens.)")
                self.history.append(("system", response))
                self.log_interaction(user_input, response, tokens_used)
            except Exception as e:
                print("Sorry, there was an error processing your request. Please try again.")
                logging.error(f"Error during interaction: {e}")
