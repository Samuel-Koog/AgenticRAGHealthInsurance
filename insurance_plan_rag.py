import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

def create_rag_pipeline(pdf_path: str, model_name: str = "llama3"):
    """
    Creates a complete RAG pipeline from a PDF file using a local Ollama model.
    """
    # --- 1. LOAD AND CHUNK THE DOCUMENT ---
    print("Loading and splitting the PDF...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    if not chunks:
        raise ValueError("PDF could not be chunked. Check the document content.")

    print(f"PDF split into {len(chunks)} chunks.")

    # --- 2. EMBED AND STORE IN VECTOR DATABASE (FAISS) ---
    print("Creating vector embeddings with HuggingFace...")
    # Use a popular, lightweight sentence-transformer model that runs locally
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # Create a FAISS vector store from the document chunks
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("Vector store created successfully.")

    # --- 3. CREATE THE RAG CHAIN ---
    # Define a retriever to fetch relevant documents
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Set up the Ollama LLM
    print(f"Setting up local LLM with Ollama model: {model_name}...")
    # Ensure Ollama application is running and the model is pulled
    llm = ChatOllama(model=model_name, temperature=0)

    # Define the prompt template
    template = """
    You are an assistant for answering questions about a health insurance plan.
    Use only the following retrieved context to answer the question.
    If you don't know the answer, just say that you don't know. Be concise.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Build the RAG chain using LangChain Expression Language (LCEL)
    rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return rag_chain


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    pdf_file_path = "sampleinsurance.pdf"

    if not os.path.exists(pdf_file_path):
        print(f"Error: The file '{pdf_file_path}' was not found.")
        print("Please make sure the PDF is in the same directory as the script.")
    else:
        try:
            # Create the pipeline
            insurance_qa_chain = create_rag_pipeline(pdf_file_path)
            print("\n✅ RAG pipeline is ready. Ask questions about your insurance plan.")
            print("   Make sure the Ollama application is running.")
            print("   Type 'exit' to quit.")

            # Start an interactive question-answering loop
            while True:
                user_question = input("\n> Your Question: ")
                if user_question.lower().strip() == 'exit':
                    break

                # Invoke the chain to get an answer
                answer = insurance_qa_chain.invoke(user_question)
                print(f"\n💡 Answer: {answer}")

        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Please ensure the Ollama application is running and the model 'llama3' is installed ('ollama pull llama3').")
