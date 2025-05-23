from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

# init model
model = ChatOllama(
    model="llama3.2:3b",  # has to be in the {name}:{tag} format
    temperature=0.2,  # between 0 and 1
)

# init messages
messages = [
    SystemMessage("Translate the following from English to Vietnamese"),
    HumanMessage("It's a good day to go camping today."),
]

# print output to stdout
output = model.invoke(messages)
print(output.content)

# print output with diagnostics to stdout
print(f"""
Content: {output.content}
Input tokens: {output.usage_metadata["input_tokens"]}
Output tokens: {output.usage_metadata["output_tokens"]}
Tokens/s: {(output.response_metadata["eval_count"] * 10**9 / output.response_metadata["eval_duration"]):.2f} tokens/s
""")

# stream output to stdout
for token in model.stream(messages):
    print(token.content, end="|")  # '|' separates each token
print()

# use prompt template
system_template = "Translate the following from English to {language}"

# possible roles are: system, user, assistant, tool
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_template),  # tuple of (role/class, template)
        ("user", "{text}"),
    ]
)
# pass in a dictionary of template keys & values
prompt = prompt_template.invoke(
    {
        "language": "Vietnamese",
        "text": "Let's go out for lunch!",
    }
)

response = model.invoke(prompt)
print(response.content)
