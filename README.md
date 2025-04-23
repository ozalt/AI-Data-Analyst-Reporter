
## 🔍 Project Introduction: AI-Powered Data Analysis with RAG-Augmented Reporting

This project introduces an advanced AI-driven tool designed to **automate data exploration and report generation**, bridging structured datasets with unstructured documents like PDFs through **Retrieval-Augmented Generation (RAG)**. At its core, the system leverages:

- **Streamlit** for an interactive UI,
- **SentenceTransformers + ChromaDB** for semantic search and document retrieval,
- **Llama.cpp + Qwen 2.5** for on-device LLM inference,
- And **custom-built pipelines** for statistical profiling, correlation detection, and context-aware prompt formatting.

### Key Innovations:
- 🔗 **Hybrid RAG Pipeline**: Embeds and indexes PDF chunks for intelligent context retrieval, dynamically augmenting LLM prompts with the most relevant content.
- 📊 **Auto-Statistics Engine**: Extracts mean, std, quartiles, and categorical insights from uploaded datasets—no preprocessing required.
- 🧠 **Dynamic Prompt Construction**: Prompts are structured, readable, and customized based on user context, enabling high-quality, human-like analytics reports.
- ⚡ **Fast Local Inference**: Deployed entirely offline using `llama.cpp`, optimized with GPU acceleration for low-latency response.

### Why It Matters:
Most LLM-powered analytics tools are either *cloud-locked* or *stateless*. This solution tackles both flaws by embedding long-term memory (via ChromaDB) and running everything locally for **privacy-preserving, scalable data intelligence**.

Ideal for researchers, analysts, or product teams who need fast insights without sending sensitive data to external APIs.

---

# 🛠️ Environment Setup

Follow the steps below to set up the project environment correctly.

---

## 1. Clone the Repository

```bash
git clone https://github.com/yourusername/your-project.git
cd your-project
```

---

## 2. Create the Conda Environment

Install all required dependencies using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
```

This will create a new environment named  **LLMENV** with all necessary libraries.

---

## 3. Activate the Environment

```bash
conda activate LLMENV
```

> You can replace replace `LLMENV` with any name defined at the top of `environment.yml`.

---

# ❓ What If...

### ➡️ If the environment already exists?

You can update it instead of recreating:

```bash
conda env update --file environment.yml --prune
```

The `--prune` flag removes packages not listed in the file to keep things clean.

---

### ➡️ If Conda isn't installed?

You must install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) first.

---

### ➡️ If you face any errors?

- Make sure you're using an updated version of `conda`.
- Delete the partially created environment and retry:
  ```bash
  conda remove --name your-env-name --all
  ```
  Then rerun the steps.

---

# 🚀 You're Ready!

After activating the environment, you can run the project using:

```bash
streamlit run AI_rep_gen.py
```
