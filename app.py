from RAG import retriever
from LLM_MODEL import LLM
from prompts import prompt
from langchain_core.messages import HumanMessage

print("App Started")

while True:
    question = input("\nYou")

    if question.lower() == "exit":
        break

    docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    final_prompt = prompt.format(
        Context = context,
        question = question
    )

    response = LLM.invoke([
        HumanMessage(content = final_prompt)
    ])

    print("\nAssistant: ", response.content)

