from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits.zapier.toolkit import ZapierToolkit
from langchain_community.utilities.zapier import ZapierNLAWrapper
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize components



llm = ChatOpenAI(
    model_name="mistral/mixtral-8x7b-instruct",
    base_url="https://openrouter.ai/api/v1",  # ✅ Required
    api_key="YOUR_API_KEY"
)
zapier = ZapierNLAWrapper()
toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)

# Use the recommended agent type
agent = initialize_agent(
    tools=toolkit.tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# Define the actual summary
summary = (
    "Sharat contacted ABCBangs Credit Card customer care to make a payment of $200. "
    "The customer care executive verified Sharat's account information and processed the payment successfully. "
    "The updated balance is now $350."
)

# Compose the email task
task = (
    "Send an email via Gmail to shivamshar0310@gmail.com with the subject 'Audio Transcription Summary'. "
    f"The body of the email should be:\n\n{summary}"
)

# Run the agent
response = agent.invoke({"input": task})

# Print the output
print(response)