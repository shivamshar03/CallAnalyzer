
import whisper
from git.cmd import handle_process_output
from langchain_groq import ChatGroq
from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits.zapier.toolkit import ZapierToolkit
from langchain_community.utilities.zapier import ZapierNLAWrapper
from dotenv import load_dotenv

# Set API keys (preferably from environment variables)
load_dotenv()


def email_summary(file):
    """
    Transcribe an audio file and email a summary of the transcription.

    Args:
        file (str): Path to the audio file to transcribe
    """
    try:
        # Initialize the large language model
        llm = ChatGroq(temperature=1, model_name="llama3-8b-8192")

        zapier = ZapierNLAWrapper()
        toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)

        # Use the recommended agent type
        agent = initialize_agent(
            tools=toolkit.tools,
            llm=llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors = True
        )

        # Load and transcribe the audio file
        print(f"Loading Whisper model...")
        model = whisper.load_model("base")

        print(f"Transcribing audio file: {file}")
        result = model.transcribe(file)
        transcript = result["text"]
        print(f"Transcript: {transcript}")

        # Format the task for the agent
        task = f"Send an email to nexhubcommunity@gmail.com via Gmail with the subject 'Audio Transcription Summary'. The email should include a concise summary of this transcript: {transcript}"

        # Execute the agent
        response = agent.invoke({"input": task})
        print(f"Agent response: {response}")

        return True

    except Exception as e:
        print(f"Error in email_summary: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
