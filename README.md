# TCC2

First, install the required packages:

Python

TypeScript

Copy

Ask AI
pip install -qU langsmith openai langchain_core
Next, make sure you have signed up for a LangSmith account, then create and set your API key. You will also want to sign up for an OpenAI API key to run the code in this tutorial.

Copy

Ask AI
LANGSMITH_API_KEY = '<your_api_key>'
OPENAI_API_KEY = '<your_api_key>'
​
2. Create a prompt
To create a prompt in LangSmith, define the list of messages you want in your prompt and then wrap them using the ChatPromptTemplate function (Python) or TypeScript function. Then all you have to do is call push_prompt (Python) or pushPrompt (TypeScript) to send your prompt to LangSmith!

Python

TypeScript

Copy

Ask AI
from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate

# Connect to the LangSmith client
client = Client()

# Define the prompt
prompt = ChatPromptTemplate([
    ("system", "You are a helpful chatbot."),
    ("user", "{question}"),
])

# Push the prompt
client.push_prompt("my-prompt", object=prompt)
​
3. Test a prompt
To test a prompt, you need to pull the prompt, invoke it with the input values you want to test and then call the model with those input values. your LLM or application expects.

Python

TypeScript

Copy

Ask AI
from langsmith import Client
from openai import OpenAI
from langchain_core.messages import convert_to_openai_messages

# Connect to LangSmith and OpenAI
client = Client()
oai_client = OpenAI()

# Pull the prompt to use
# You can also specify a specific commit by passing the commit hash "my-prompt:<commit-hash>"
prompt = client.pull_prompt("my-prompt")

# Since our prompt only has one variable we could also pass in the value directly
# The code below is equivalent to formatted_prompt = prompt.invoke("What is the color of the sky?")
formatted_prompt = prompt.invoke({"question": "What is the color of the sky?"})

# Test the prompt
response = oai_client.chat.completions.create(
    model="gpt-4o",
    messages=convert_to_openai_messages(formatted_prompt.messages),
)
​
4. Iterate on a prompt
LangSmith makes it easy to iterate on prompts with your entire team. Members of your workspace can select a prompt to iterate on, and once they are happy with their changes, they can simply save it as a new commit.
To improve your prompts:
We recommend referencing the documentation provided by your model provider for best practices in prompt creation, such as Best practices for prompt engineering with the OpenAI API and Gemini’s Introduction to prompt design.
To help with iterating on your prompts in LangSmith, we’ve created Prompt Canvas — an interactive tool to build and optimize your prompts. Learn about how to use Prompt Canvas.
To add a new commit to a prompt, you can use the same push_prompt (Python) or pushPrompt (TypeScript) methods as when you first created the prompt.

Python

TypeScript

Copy

Ask AI
from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate

# Connect to the LangSmith client
client = Client()

# Define the prompt to update
new_prompt = ChatPromptTemplate([
    ("system", "You are a helpful chatbot. Respond in Spanish."),
    ("user", "{question}"),
])

# Push the updated prompt making sure to use the correct prompt name
# Tags can help you remember specific versions in your commit history
client.push_prompt("my-prompt", object=new_prompt, tags=["Spanish"])