import streamlit as st
import pandas as pd
from llama_cpp import Llama
import gc
import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer
import torch
import os

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 


class RAGProcessor:
    def __init__(self, model_name="nomic-ai/nomic-embed-text-v1.5", db_path="rag_db", chunk_size=200, overlap=40):
        """Initialize embedding model, vector DB, and chunking parameters."""
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(name="rag_collection")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def embed_text(self, text, task="search_document"):
        """Convert text into vector embeddings with the required prefix."""
        formatted_text = f"{task}: {text}"
        return self.model.encode(formatted_text).tolist()

    def chunk_text(self, text):
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk = " ".join(words[i : i + self.chunk_size])
            chunks.append(chunk)
        return chunks

    def add_to_db(self, texts):
        """Store chunked text embeddings in vector database."""
        for idx, text in enumerate(texts):
            chunks = self.chunk_text(text)
            for j, chunk in enumerate(chunks):
                embedding = self.embed_text(chunk, task="search_document")
                self.collection.add(ids=[f"{idx}_{j}"], embeddings=[embedding], metadatas=[{"text": chunk}])

    def extract_text_from_pdf(self, pdf_file):
        """Extract text from a PDF file from a BytesIO object."""
        with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
            text = "\n".join([page.get_text() for page in doc])
        return text

    def add_pdf_to_db(self, pdf_files):
        """Extract and store text chunks from PDFs."""
        for pdf in pdf_files:
            text = self.extract_text_from_pdf(pdf)
            self.add_to_db([text])

    def retrieve_relevant(self, query, top_k=3):
        """Retrieve the most relevant text chunks for a given query."""
        query_embedding = self.embed_text(query, task="search_query")
        results = self.collection.query(query_embedding, n_results=top_k)
        return [r["text"] for r in results["metadatas"][0]]

    def augment_query(self, query):
        """Retrieve relevant chunks and format an enhanced query."""
        relevant_chunks = self.retrieve_relevant(query)
        augmented_prompt = f"Extract and utilize relevant knowledge from the provided context within `<context></context>` XML tags. If tags are blank, ignore and proceed to User Given Context. \n\n <context> \n\n {''.join(relevant_chunks)} </context> \n\n {query}"
        return augmented_prompt



class DataProcessor:

    def __init__(self, df):
        self.df = df
        self.total_rows, self.total_columns = df.shape

        # Numeric statistics
        self.mean_values = df.mean(numeric_only=True).round(2).to_dict()
        self.std_values = df.std(numeric_only=True).round(2).to_dict()
        self.min_max_values = df.agg(["min", "max"]).to_dict()
        self.quartiles = df.quantile([0.25, 0.5, 0.75]).to_dict()
        self.correlation_insights = self.compute_correlation()

        # Categorical analysis
        self.categorical_summary = self.compute_categorical_summary()

    def format_as_table(self, data_dict):
        """Formats dictionary data into a readable table string."""
        return "\n".join([f"- **{key}**: {value}" for key, value in data_dict.items()])
    
    def compute_correlation(self):
        correlation_insights = []
        num_cols = self.df.select_dtypes(include=['number']).columns
        if len(num_cols) >= 2:
            correlations = self.df.corr().round(2)
            for col in num_cols:
                if col != self.df.columns[-1]:  
                    correlation_insights.append(
                        f"- **{col} & {self.df.columns[-1]}**: {correlations.loc[col, self.df.columns[-1]]}"
                    )
        return correlation_insights if correlation_insights else ["No strong correlations found."]

    def compute_categorical_summary(self):
        categorical_summary = {}
        cat_cols = self.df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            categorical_summary[col] = {
                "unique_values": self.df[col].nunique(),
                "most_frequent": self.df[col].mode()[0] if not self.df[col].mode().empty else None
            }
        return categorical_summary



class PromptFormatter:

    def __init__(self, data_processor, user_context=None):
        self.data_processor = data_processor
        self.user_context = user_context 

    def get_system_prompt(self):
        return f"""You are an expert data analyst. 
            Your task is to analyze the given dataset and generate a structured, insightful report. 
            Provide a professional breakdown of the dataset, highlighting key patterns, statistics, and recommendations. 
            Ensure clarity and completeness in your response."""
    
    def get_user_prompt(self):
        return f"""
If User Given Context is blank, ignore and proceed to Instruction.

User Given Context: {self.user_context}

Instruction:
Analyze the dataset and generate a **clear, well-structured, and human-readable** report. Follow the format below and **bold all feature/column names** for readability.

Understand and explain the dataset's purpose. Infer the dataset purpose in atleast 7 lines based on your common knowledge.

## **1. Overview**  
Provide a brief summary of the dataset, including:
- **Total Records**: {self.data_processor.total_rows}  
- **Total Attributes**: {self.data_processor.total_columns}  
- **Dataset Purpose**: Identify the dataset purpose based on available data.

---

## **2. Summary Statistics**  
Present key statistical metrics using **tables where appropriate** for clarity:
- **Total Records**: {self.data_processor.total_rows}  
- **Mean Values**:  
  {self.data_processor.format_as_table(self.data_processor.mean_values)}  
- **Standard Deviations**:  
  {self.data_processor.format_as_table(self.data_processor.std_values)}  
- **Min / Max Values**:  
  {self.data_processor.format_as_table(self.data_processor.min_max_values)}  
- **Quartiles (25%, 50%, 75%)**:  
  {self.data_processor.format_as_table(self.data_processor.quartiles)}  

### **Key Insights:**  
Summarize important findings in **3-4 concise sentences**:  
- What trends do the statistics reveal?  
- Are there any noticeable outliers or anomalies?  
- What do the quartile values indicate about data distribution?  

---

## **3. Key Insights & Correlation Analysis**  
Analyze relationships between attributes and highlight significant correlations:
{chr(10).join(self.data_processor.correlation_insights) if self.data_processor.correlation_insights else "No strong correlations found in the dataset."}

For any detected correlations, briefly **explain their significance**:
- How do they impact the dataset overall interpretation?  
- Which correlations are strongest, and why might they matter?  
- Are there unexpected relationships between features?  

---

## **4. Recommendations & Next Steps**  
Based on the dataset structure and findings, suggest:  
- **Data Cleaning Steps:** Handling missing values, removing duplicates, addressing outliers.  
- **Feature Engineering:** Creating new features to improve predictive power.  
- **ML Modeling Strategies:** Suitable models (e.g., logistic regression, random forests, gradient boosting).  
- **Visualization Techniques:** Best plots (e.g., heatmaps, scatter plots, violin plots) to explore patterns effectively.  

---

## **5. Conclusion**  
Summarize **key insights and trends**, emphasizing potential improvements.  
- What actionable insights can be drawn?  
- How can the dataset be better utilized for **analysis, decision-making, or modeling**?  
- Provide recommendations for further exploration or refinements. 
- Provide data-driven decisions for business. 
"""

    

class AIReportGenerator:
    """Handles AI-based report generation using a Qwen 2.5 model."""
    
    def __init__(self, model_path):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=12000,  
            n_gpu_layers=30,  
            n_batch = 512
        )

        #Model Specific Format, make sure to change it
    def final_prompt(self, system_prompt, user_prompt):
                    #Currently using Qwen 2.5 format
        return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    
                    # Format the prompt for Mistral 7B Instruct
             # f"""<s>[INST] {system_prompt} \n\n {user_prompt} [/INST]"""

                    # Format the prompt for Llama 3.1 8B Instruct
            # f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            #                         Cutting Knowledge Date: December 2023
            #                         Today Date: 26 Jul 2024
            #                         {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

            #                         {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    
    def generate_report(self, df, user_context , use_rag):
        try:
            data_processor = DataProcessor(df)
            prompt_format = PromptFormatter(data_processor, user_context)
            system_prompt = prompt_format.get_system_prompt()
            user_prompt = prompt_format.get_user_prompt()
            rag_processor = RAGProcessor()
            final_user_prompt =rag_processor.augment_query(user_prompt) if use_rag else user_prompt
            report_prompt = self.final_prompt(system_prompt, final_user_prompt)

            response = self.llm(
            prompt=report_prompt,
            max_tokens=4024,
            temperature=0.1,
            top_p=0.8,
        )
            return response["choices"][0]["text"].strip()
    
        finally:
            del self.llm  # Delete after completion
            gc.collect()
    def display_prompt(self, df, user_context , use_rag):
            data_processor = DataProcessor(df)
            prompt_format = PromptFormatter(data_processor, user_context)
            system_prompt = prompt_format.get_system_prompt()
            user_prompt = prompt_format.get_user_prompt()
            rag_processor = RAGProcessor()
            final_user_prompt =rag_processor.augment_query(user_prompt) if use_rag else user_prompt
            report_prompt = self.final_prompt(system_prompt, final_user_prompt)
            return report_prompt

st.title("📊 AI Report Generator")



# Toggle for enabling RAG
toggle_rag = st.toggle("Enable RAG (Retrieval-Augmented Generation)")

if toggle_rag:
    st.info("Upload documents (TXT, CSV, PDF) to enhance AI responses.")

    uploaded_files = st.file_uploader(
        "Upload files", 
        type=["txt", "csv", "pdf"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        # Initialize RAG system
        rag_processor = RAGProcessor()
        for file in uploaded_files:
            if file.type == "application/pdf":
                rag_processor.add_pdf_to_db([file])  # Handle PDFs
            else:
                text = file.read().decode("utf-8")  # Handle TXT/CSV
                rag_processor.add_to_db([text])

        st.success(f"{len(uploaded_files)} documents added to RAG!")


# File uploader for CSV
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Display a sample of the dataset
    st.write("📄 Data Preview:", df.head())

    # Let user select columns to analyze
    selected_columns = st.multiselect("Select columns to analyze:", df.columns, default=df.columns.tolist())

    if "context" not in st.session_state:
        st.session_state.context = " "
    # Let user specify a context for the CSV file
    user_context = st.text_area("Provide Context:", st.session_state.context)
    
    # Sampling large datasets
    if len(df) > 5000:
        st.warning("Dataset is too large! Sampling 5000 rows for analysis.")
        df = df.sample(n=5000, random_state=42)

    # Proceed if columns are selected
    if selected_columns:
        df = df[selected_columns]
        ai_generator = AIReportGenerator("GGUF Models\Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf")
        if st.button("Preview Prompt"):
            prompt_preview = ai_generator.display_prompt(df, user_context,  use_rag=toggle_rag)
            # Display final report
            st.subheader("📄 Generated Report:")
            st.text_area(prompt_preview, height=400, key="prompt_preview")

        if st.button("Generate Report"):
            st.session_state.context = user_context
            ai_generated_report = ai_generator.generate_report(df, user_context,  use_rag=toggle_rag)
            # Display final report
            st.subheader("📄 Generated Report:")
            st.text_area("📄 AI-Generated Report:", ai_generated_report, height=400, key="final_report")
            
            
            
            
