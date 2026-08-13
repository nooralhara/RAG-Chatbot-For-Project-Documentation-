from langchain_core.prompts import PromptTemplate

template = """

You Are an AI Assistant.

Answer the user's questions using only the provided context.

If the answer is not in context, say:
"I dont have that information."

The answer should be 20 words or fewer.

## Context:
{Context}

## Question:
{question}

"""

prompt = PromptTemplate(
    input_variables= ["Context", "question"],
    template= template
)