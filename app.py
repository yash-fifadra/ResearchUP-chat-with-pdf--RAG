import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain_community.llms import HuggingFaceHub

def get_pdf_text(pdf_docs):
    text = ""
    if not pdf_docs:
        return text
    
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            st.error(f"Error reading PDF {pdf.name}: {str(e)}")
            continue
    
    if not text.strip():
        st.warning("No text could be extracted from the uploaded PDFs.")
        return ""
    
    return text


def get_text_chunks(text):
    if not text.strip():
        return []
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    if not text_chunks:
        st.error("No text chunks available to create vector store.")
        return None
    
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None


def get_conversation_chain(vectorstore, model_choice="compound-beta"):
    if not vectorstore:
        st.error("Vector store is not available.")
        return None
    
    try:
        # Check if GROQ API keys are available
        groq_api_key_1 = os.getenv("GROQ_API_KEY_1")
        groq_api_key_2 = os.getenv("GROQ_API_KEY_2")
        
        if not groq_api_key_1 and not groq_api_key_2:
            st.error("No GROQ API keys found! Please add GROQ_API_KEY_1 and/or GROQ_API_KEY_2 to your .env file.")
            return None
        
    
        llm = ChatGroq(
            api_key=groq_api_key_1,
            model=model_choice,
            temperature=0.4,
            max_tokens=250
        )

        memory = ConversationBufferMemory(
            memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        return conversation_chain
    except Exception as e:
        st.error(f"Error creating conversation chain: {str(e)}")
        return None


def generate_document_summary(vectorstore, api_key_2):
    """Generate an in-depth summary of the uploaded documents using API 2 (gemma2-9b-it)"""
    try:
        if not api_key_2:
            return "Summary not available (API key 2 not configured)"
        
        if not vectorstore:
            return "Vector store not available for summary generation"
        
        # Use API 2 for summary generation
        summary_llm = ChatGroq(
            api_key=api_key_2,
            model="llama-3.1-8b-instant",
            temperature=0.2,
            max_tokens=2000
        )
        
        # Retrieve comprehensive content from vector store for summary
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})  # Get top 10 most relevant chunks
        relevant_chunks = retriever.get_relevant_documents("summarize this entire document comprehensively")
        
        # Combine all relevant chunks for comprehensive summary
        combined_content = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
        
        # Create a comprehensive summary prompt
        summary_prompt = f"""
        Please provide a comprehensive, in-depth summary of the following document content. 
        The summary should cover all major topics, key points, and important details from the entire document.
        Make it detailed enough to give readers a complete understanding of the document's content.
        
        Document Content:
        {combined_content}
        
        Please provide a comprehensive summary that covers:
        1. Main topics and themes
        2. Key findings or conclusions
        3. Important details and data
        4. Overall purpose and scope of the document
        
        Comprehensive Summary:
        """
        
        # Generate summary
        summary_response = summary_llm.invoke(summary_prompt)
        return summary_response.content
        
    except Exception as e:
        return f"Error generating summary: {str(e)}"


def handle_userinput(user_question):
    if not st.session_state.conversation:
        st.error("Please upload and process PDFs first.")
        return
    
    try:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error processing your question: {str(e)}")


def main():
    load_dotenv()
    st.set_page_config(page_title="ResearchUp",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "document_summary" not in st.session_state:
        st.session_state.document_summary = None

    st.header("ResearchUp : Improve your Research :books:")
    
    # Check if required environment variables are set
    groq_api_key_1 = os.getenv("GROQ_API_KEY_1")
    groq_api_key_2 = os.getenv("GROQ_API_KEY_2")
    
    if not groq_api_key_1 and not groq_api_key_2:
        st.error("⚠️ No GROQ API keys found! Please create a .env file with your API keys.")
        st.info("Create a .env file in your project root with:")
        st.info("GROQ_API_KEY_1=your_first_api_key_here")
        st.info("GROQ_API_KEY_2=your_second_api_key_here")
        return
    
    # Display API configuration info
    st.info("🔧 **API Configuration**: Using dual-API system for optimal performance")
    
    # Display document summary in main page if available
    if st.session_state.document_summary:
        st.divider()
        
        # Summary with bot image
        col1, col2 = st.columns([1, 4])
        with col1:
            st.image("https://img.freepik.com/premium-vector/cute-robot-mortarboard-vector-character-education-academic-tech-learning-themes_1020043-961.jpg", 
                    width=80, caption="AI Assistant")
        with col2:
            st.subheader("📋 Comprehensive Research Paper Summary")
        
        st.info(st.session_state.document_summary)
        st.divider()
        
        # Show Q&A interface after summary
        st.subheader("💬 Ask Questions About Your Research Paper")
        user_question = st.text_input("Ask a question about your Research Paper:")
        if user_question:
            handle_userinput(user_question)
    else:
        # Show Q&A input field (disabled until processing)
        st.subheader("💬 Ask Questions About Your Research Paper")
        st.text_input("Ask a question about your Research Paper:", disabled=True, 
                     help="Upload and process your PDFs first to enable Q&A")

    with st.sidebar:
        st.subheader("Your Research Paper")
    
        # Show instructions in sidebar
        st.info("📚 **Getting Started:**")
        st.info("1. Upload PDF files below")
        st.info("2. Click 'Process' to generate summary")
        st.info("3. Summary will appear in main page")
        st.info("4. Ask questions in main page")
        st.divider()
        
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True, type=['pdf'])
        
        if st.button("Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
                return
            
            # Clear previous summary and conversation when processing new documents
            st.session_state.document_summary = None
            st.session_state.conversation = None
            st.session_state.chat_history = []
                
            with st.spinner("Processing PDFs..."):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                
                if not raw_text.strip():
                    st.error("No text could be extracted from the uploaded PDFs.")
                    return

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                
                if not text_chunks:
                    st.error("Failed to create text chunks from the PDF content.")
                    return

                # create vector store
                vectorstore = get_vectorstore(text_chunks)
                
                if not vectorstore:
                    st.error("Failed to create vector store.")
                    return

                # Generate document summary using API 2
                groq_api_key_2 = os.getenv("GROQ_API_KEY_2")
                if groq_api_key_2:
                    with st.spinner("Generating comprehensive Research Paper summary..."):
                        summary = generate_document_summary(vectorstore, groq_api_key_2)
                        st.session_state.document_summary = summary
                
                # Create conversation chain using API 1 (compound-beta) for Q&A
                st.session_state.conversation = get_conversation_chain(vectorstore, "compound-beta")
                
                if st.session_state.conversation:
                    st.success("PDFs processed successfully! You can now ask questions.")
                    st.info("💡 **Q&A Mode**: Using compound-beta model (API 1) for detailed answers")
                    # Force page refresh to show the summary immediately
                    st.rerun()
                else:
                    st.error("Failed to create conversation chain.")


if __name__ == '__main__':
    main()