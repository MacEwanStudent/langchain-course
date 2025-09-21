import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from template import template

load_dotenv()


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press F9 to toggle the breakpoint.

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #print_hi('PyCharm')
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI()

    query = "what is Pinecone in machine learning?"
    """
    chain = PromptTemplate.from_template(template=query) | llm
    result = chain.invoke(input={})

    print(result.content)"""

    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
    )
    """
    #propmt
    retrival_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain= create_stuff_documents_chain(llm, retrival_qa_chat_prompt)
    retrival_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain= combine_docs_chain
    )

    result = retrival_chain.invoke(input={"input":query})

    print(result)"""

    custom_rag_prompt= PromptTemplate.from_template(template)

    rag_chain= (
        {"context":vectorstore.as_retriever() | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
    )

    res = rag_chain.invoke(query)
    print(res)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
