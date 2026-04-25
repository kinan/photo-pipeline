To index documents for **Retrieval-Augmented Generation (RAG)**, you must transform raw data (like PDFs, markdown files, or text) into a searchable numerical format called **embeddings** and store them in a **Vector Database**.

Here is the standard 5-step pipeline used in modern AI applications:

### 1. Document Loading

The first step is to extract text from your source files. Depending on the format, you use specialized loaders:

- **Text/Markdown:** Simple string extraction.
    
- **PDFs:** Tools like `PyPDF` or `marker` (for high-quality OCR).
    
- **Code/HTML:** Loaders that preserve structure (like headers and functions).
    

### 2. Document Chunking (Splitting)

LLMs have a limited **Context Window** (the "token" limit we discussed earlier). You cannot feed a 100-page book at once. You must split documents into smaller "chunks."

- **Fixed-size:** e.g., chunks of 500 tokens.
    
- **Recursive Character Splitting:** Splits by paragraphs, then sentences, to keep context together.
    
- **Overlapping:** Usually, you overlap chunks by ~10–15% so that context isn't lost at the "seam" where a sentence was cut.
    

### 3. Creating Embeddings

This is the "magic" step. You pass each text chunk through an **Embedding Model** (like `nomic-embed-text` or `OpenAI text-embedding-3`).

- The model converts the text into a **Vector** (a long list of numbers, e.g., 768 or 1536 dimensions).
    
- **Why?** Computers can't compare the "meaning" of words, but they can calculate the mathematical distance between two vectors. "King" and "Queen" will be physically close to each other in vector space.
    

### 4. Vector Storage

You store these vectors in a specialized **Vector Database**.

- **Options:** Pinecone (Cloud), Milvus (Enterprise), or **ChromaDB / Qdrant** (Local/Open Source).
    
- **Metadata:** Along with the vector, you store the original text and metadata (e.g., `{"source": "chapter1.pdf", "page": 12}`).
    

### 5. Indexing for Retrieval

The database creates an index (like HNSW) to make searching fast. When a user asks a question, the system:

1. Embeds the **User Query**.
    
2. Performs a **Similarity Search** against the database.
    
3. Retrieves the top-k (usually 3–5) most relevant chunks to send to the LLM.
    

---

### Comparison of Indexing Strategies

|**Strategy**|**Best For**|**Pros**|**Cons**|
|---|---|---|---|
|**Flat Chunking**|Simple FAQs|Easy to implement|Loses global context|
|**Parent-Document**|Technical Docs|Retrieves full context|Higher token usage|
|**Hierarchical**|Large Books|