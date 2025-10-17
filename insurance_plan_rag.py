import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from operator import itemgetter

# --- Mock Provider Data ---
PROVIDER_DATA = [
    {"name": "BRIDGET ABAJIAN PA-C", "specialty": "Physician Assistant, Medical", "phone": "213-284-3200", "address": "400 W 30TH ST, LOS ANGELES, CA 90007"},
    {"name": "ALI ABAIAN M.D.", "specialty": "General Practice", "phone": "323-581-7400", "address": "8460 S CENTRAL AVE, LOS ANGELES, CA 90001"},
    {"name": "SANAZ ASAL ABADI D.D.S.", "specialty": "Dentist, General Practice", "phone": "310-923-1700", "address": "2143 CAMDEN AVE, LOS ANGELES, CA 90025"},
    {"name": "OMER ABA-OMER MD", "specialty": "Family Medicine", "phone": "310-828-4530", "address": "2424 WILSHIRE BLVD, SANTA MONICA, CA 90403"}
]

def find_relevant_providers(specialty_query, providers):
    if not specialty_query or specialty_query.lower() == 'general inquiry':
        return []
    query = specialty_query.lower()
    return [p for p in providers if query in p["specialty"].lower()]

def create_injury_suggestion_chain(llm: ChatOllama):
    template = """
    Based on the user's question, what is a single, likely injury or medical condition they might have?
    Focus on the core symptom (e.g., "Food Poisoning", "Sprained Ankle", "Skin Allergy").
    Do not mention injuries that have already been suggested.

    USER'S QUESTION: {question}
    PREVIOUSLY SUGGESTED INJURIES: {attempted_injuries}
    NEXT LIKELY INJURY/CONDITION:
    """
    prompt = ChatPromptTemplate.from_template(template)
    return prompt | llm | StrOutputParser()

def create_doctor_specialty_extractor_chain(llm: ChatOllama):
    template = """
    Based on the confirmed injury or condition, what is the most appropriate medical specialty?
    Your answer should be a concise medical specialty (e.g., "Dermatology", "Orthopedics", "Gastroenterology").

    CONFIRMED INJURY/CONDITION: {injury}
    MEDICAL SPECIALTY:
    """
    prompt = ChatPromptTemplate.from_template(template)
    return prompt | llm | StrOutputParser()

def create_final_rag_chain(vector_store: FAISS, llm: ChatOllama):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    template = """
    You are a professional and friendly health insurance assistant.
    Based on the user's question and the information provided, structure your final answer as follows:

    1.  **Situation Analysis:** Assess potential urgency. If the query sounds like an emergency, first provide this exact advice: "If you believe this is a medical emergency, please call 911 or go to the nearest emergency room immediately."

    2.  **Coverage Details:** Using the IDENTIFIED MEDICAL SPECIALTY, answer the user's question about insurance costs using ONLY the provided CONTEXT.

    3.  **Nearby Providers That May Be Relevant:** If providers are listed, present them clearly.

    4.  **Medical Advice:** Provide general, non-prescriptive advice. Always end with "This is not a substitute for professional medical advice. Please consult with a healthcare provider."

    INFORMATION:
    ---
    CONTEXT FROM INSURANCE PLAN: {context}
    IDENTIFIED MEDICAL SPECIALTY: {specialty}
    NEARBY PROVIDERS LIST: {providers}
    USER'S QUESTION: {question}
    ---
    YOUR FINAL ANSWER:
    """
    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
            "providers": itemgetter("providers"),
            "specialty": itemgetter("specialty")
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    pdf_file_path = "sampleinsurance.pdf"
    index_path = "faiss_index"

    if not os.path.exists(pdf_file_path):
        print(f"Error: The file '{pdf_file_path}' was not found.")
    else:
        try:
            print("Initializing the insurance assistant... Please wait.")
            embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            if os.path.exists(index_path):
                print("Loading existing knowledge base...")
                vector_store = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
            else:
                print("Creating new knowledge base from PDF (this will take a moment)...")
                loader = PyPDFLoader(pdf_file_path)
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
                chunks = text_splitter.split_documents(documents)
                vector_store = FAISS.from_documents(chunks, embedding_model)
                vector_store.save_local(index_path)
                print("Knowledge base created and saved for future use.")

            llm = ChatOllama(model="llama3", temperature=0)
            
            injury_chain = create_injury_suggestion_chain(llm)
            specialty_chain = create_doctor_specialty_extractor_chain(llm)
            final_answer_chain = create_final_rag_chain(vector_store, llm)
            
            print("\nWelcome! I'm here to help you understand your insurance plan.")
            print("Please describe your medical situation.")
            print("Type 'exit' to quit.")

            while True:
                user_question = input("\n> Your Question: ")
                if user_question.lower().strip() == 'exit':
                    break

                print("\nAnalyzing your request...")
                
                confirmed_injury = None
                attempted_injuries = []
                for _ in range(3): # Loop a max of 3 times to suggest injuries
                    suggestion = injury_chain.invoke({"question": user_question, "attempted_injuries": ", ".join(attempted_injuries)})
                    if suggestion in attempted_injuries: continue
                    
                    user_response = input(f"   -> Based on your description, I think the issue might be a '{suggestion}'. Is this correct? (yes/no/exit): ").lower()
                    
                    if user_response == 'yes':
                        confirmed_injury = suggestion
                        break
                    elif user_response == 'exit':
                        break
                    else:
                        attempted_injuries.append(suggestion)
                
                if not confirmed_injury:
                    print("   -> I couldn't confirm the specific issue. Let's try another question.")
                    continue

                print(f"\nUnderstood. Looking up information for '{confirmed_injury}'.")
                
                identified_specialty = specialty_chain.invoke({"injury": confirmed_injury})
                print(f"   -> Identified Medical Specialty: {identified_specialty}")
                
                relevant_providers = find_relevant_providers(identified_specialty, PROVIDER_DATA)
                provider_info_str = "\n".join([f"- {p['name']} ({p['specialty']}) at {p['address']}, Phone: {p['phone']}" for p in relevant_providers]) or "No specific providers found in our sample data."

                print("   -> Now, looking up your coverage and compiling your answer...")
                final_input = {
                    "question": user_question,
                    "providers": provider_info_str,
                    "specialty": identified_specialty
                }
                answer = final_answer_chain.invoke(final_input)
                print(f"\n💡 Here is the information I found:\n---\n{answer}")

        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Please ensure the Ollama application is running and the model 'llama3' is installed ('ollama pull llama3').")

