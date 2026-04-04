from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

load_dotenv()


loader = TextLoader("notes.txt")
documents = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

vectorstore.add_documents(docs)

retriever = vectorstore.as_retriever()


llm = ChatMistralAI(model="mistral-small-latest")


prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Answer ONLY from given context. If not found say: I could not find the answer."),
    ("human",
     "Context:\n{context}\n\nQuestion:\n{question}")
])

print("✅ RAG Ready 🚀 (type 0 to exit)")

while True:
    query = input("You: ")

    if query == "0":
        print("👋 Exiting...")
        break

    docs = retriever.invoke(query)

    if not docs:
        print("AI: I could not find the answer.")
        continue

    context = "\n\n".join([doc.page_content for doc in docs])

    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    response = llm.invoke(final_prompt)

    print("AI:", response.content)               12345678
