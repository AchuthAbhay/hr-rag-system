from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re

expand_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Generate 3 alternative search queries for HR policy retrieval. "
     "Keep them short keyword-style."
     "Do not write sentences."
     "No numbering. One per line."),
    ("human", "{question}")
])

parser = StrOutputParser()



def expand_queries(llm, question: str):
    chain = expand_prompt | llm | parser

    text = chain.invoke({"question": question})

    queries = []

    for q in text.split("\n"):
        q = q.strip()

        # remove numbering like "1. " or "2) "
        q = re.sub(r"^\d+[\.\)]\s*", "", q)

        if len(q) > 3:
            queries.append(q.lower())

    return list(dict.fromkeys([question] + queries))

