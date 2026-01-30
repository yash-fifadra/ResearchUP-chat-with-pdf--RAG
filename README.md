# ResearchUP-chat-with-pdf--RAG
# PDF Chat Application

A Streamlit-based application that allows you to chat with multiple PDF documents using AI. The application uses LangChain, Groq API, and FAISS vector database for document processing and question answering.

## Features

- Upload multiple PDF documents
- Extract and process text from PDFs
- Create vector embeddings for semantic search
- **Comprehensive document summary** using API 2 (llama-3.1-8b-instant) - covers entire PDF content
- **Q&A conversations** using API 1 (compound-beta) for detailed answers
- Conversation memory for context-aware responses

## Prerequisites

- Python 3.8 or higher
- Groq API key(s) (get them from [https://console.groq.com/](https://console.groq.com/))
  - **GROQ_API_KEY_1**: For compound-beta model (Q&A conversations - detailed reasoning)
  - **GROQ_API_KEY_2**: For llama-3.1-8b-instant (comprehensive document summaries)

## Installation

1. **Clone or download this project**

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the project root with:
   ```
   # For compound-beta model (Q&A conversations - detailed reasoning)
   GROQ_API_KEY_1=your_first_groq_api_key_here
   
   # For gemma2-9b-it model (comprehensive document summaries)
   GROQ_API_KEY_2=your_second_groq_api_key_here
   
   # You can use just one key if you prefer
   ```

## Usage

1. **Run the application:**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. **Upload PDF documents** using the sidebar

4. **Click "Process"** to analyze your documents

5. **Ask questions** about your documents in the chat interface

## Troubleshooting

### Common Issues:

1. **"No GROQ API keys found" error:**
   - Make sure you have a `.env` file in the project root
   - Add at least one API key: `GROQ_API_KEY_1=your_key_here`
   - You can add a second key: `GROQ_API_KEY_2=your_second_key_here`

2. **Import errors:**
   - Make sure all dependencies are installed: `pip install -r requirements.txt`
   - Try upgrading pip: `python -m pip install --upgrade pip`

3. **PDF processing errors:**
   - Ensure PDFs are not password-protected
   - Check if PDFs contain extractable text (not just images)
   - Try with different PDF files

4. **Memory issues:**
   - Large PDFs may cause memory issues
   - Try processing smaller documents first
   - Close other applications to free up memory

5. **Streamlit not found:**
   - Install streamlit: `pip install streamlit`
   - Make sure your virtual environment is activated

### Performance Tips:

- Use smaller PDF files for faster processing
- Process documents in batches if you have many files
- Close the browser tab when not in use to save resources

## Dependencies

- **Streamlit**: Web application framework
- **LangChain**: AI/LLM framework
- **Groq**: Fast LLM API provider
- **FAISS**: Vector similarity search
- **PyPDF2**: PDF text extraction
- **HuggingFace**: Embeddings and models



## Support

If you encounter any issues:
1. Check the troubleshooting section above
2. Ensure all dependencies are properly installed
3. Verify your API key is correct
4. Check the console for detailed error messages

## Quick Start

### 1. Clone and enter the project
```bash
git clone https://github.com/yash-fifadra/ResearchUP-chat-with-pdf--RAG.git
cd your-repo-name
```

### 2. Create and activate virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set GROQ API keys
Create a `.env` file in the project root:
```env
GROQ_API_KEY_1=your_first_groq_api_key
GROQ_API_KEY_2=your_second_groq_api_key   # optional but recommended
```

### 5. Run the app
```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`), upload your research PDFs, click **Process**, read the summary, and start asking questions.
