
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
#from ingest import documents
from langchain.chains import RetrievalQA
from langchain.agents import Tool
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers.language.language_parser import LanguageParser
from langchain.text_splitter import Language
from git import Repo

repo = Repo.clone_from("https://github.com/karpathy/micrograd", to_path="/Users/matildawiech/Desktop/docschat/repo")

# load source docs
loader = GenericLoader.from_filesystem(
    "repo/micrograd",
    glob="**/*",
    suffixes=[".py"],
    parser=LanguageParser(language=Language.PYTHON, parser_threshold=500)
)

documents = loader.load()

#split docs
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size =2000, chunk_overlap = 200
    )

texts = python_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(disallowed_special=())

db = Chroma.from_documents(texts, embeddings)

#create retriever function

retriever1 = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8},
)

llm=ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    temperature=0,
    model_name='gpt-3.5-turbo'
)

retriever = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever1
)

tool_desc = """Use this tool to answer user questions using the Tiny Grad Github. If the user states 'ask Seb' use this tool to get
the answer. This tool can also be used for follow up questions from the user."""
     
tools = [Tool(
    func=retriever.run,
    description=tool_desc,
    name='Tiny Grad Github'
)]

memory = ConversationBufferWindowMemory(
    memory_key="chat_history",  # important to align with agent prompt (below)
    k=5,
    return_messages=True
)

conversational_agent = initialize_agent(
    agent='chat-conversational-react-description', 
    tools=tools, 
    llm=llm,
    verbose=True,
    max_iterations=2,
    early_stopping_method="generate",
    memory=memory,
)

sys_msg = """You are a helpful chatbot that answers the user's questions."""

prompt = conversational_agent.agent.create_prompt(
    system_message=sys_msg,
    tools=tools
)

while True:
    question = input("What do you want to ask: ")
    response = conversational_agent(question)
    print(response)
    another_question = input("Do you have another question? (yes/no): ")
    if another_question.lower() == "no":
        break

