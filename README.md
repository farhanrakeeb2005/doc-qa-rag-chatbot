# Doc Q&A RAG Chatbot

A Retrieval-Augmented Generation (RAG) pipeline that answers questions about a PDF document — and proves, side by side, why grounding an LLM's answer in real retrieved text beats letting it answer from memory alone.

Upload a PDF, ask it a question, and the notebook shows you **two answers**: one generated purely from the model's internal knowledge (prone to hallucination), and one generated using only text actually retrieved from the document (grounded, verifiable) — along with similarity scores showing how confident the retrieval was.

## How it works

1. **Document Ingestion** — [PyMuPDF](https://pymupdf.readthedocs.io/) extracts raw text from every page of the uploaded PDF.
2. **Chunking** — the text is split into overlapping fixed-size chunks (500 characters, 100-character overlap) so sentences crossing a chunk boundary aren't lost.
3. **Embedding & Indexing** — each chunk is encoded into a 384-dimensional semantic vector using the `all-MiniLM-L6-v2` Sentence Transformer, and stored in an in-memory [ChromaDB](https://www.trychroma.com/) collection.
4. **Retrieval** — at query time, the question is encoded the same way, and ChromaDB returns the top-k most semantically similar chunks.
5. **Grounded generation** — those chunks are injected into a prompt that instructs [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) to answer using *only* that context.
6. **Hallucination baseline** — the same question is also sent to the model with no retrieved context at all, as a "closed-book" comparison.
7. Both answers, the retrieved passages, per-chunk similarity scores, and a summary comparison table are displayed together.

## Tech stack

| Component | Purpose |
|---|---|
| [PyMuPDF (`fitz`)](https://pymupdf.readthedocs.io/) | PDF text extraction |
| [Sentence Transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) | Turning text into semantic embeddings |
| [ChromaDB](https://www.trychroma.com/) | Vector storage and similarity search |
| [Hugging Face Transformers](https://huggingface.co/docs/transformers) | Loading and running Mistral-7B-Instruct |
| [Pandas](https://pandas.pydata.org/) | The final RAG-vs-hallucination comparison table |

## Running this

This is built and tested to run in **Google Colab** with a GPU runtime (Runtime → Change runtime type → GPU). Mistral-7B-Instruct needs real GPU memory to run at a reasonable speed — it is not practical to run on CPU-only hardware.

1. Open `doc_qa_rag_chatbot.ipynb` in Colab.
2. Upload a PDF of your choice (any PDF with real, extractable text works — this was tested against ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)).
3. Update `PDF_PATH` to point at your uploaded file.
4. Run all cells top to bottom.
5. When prompted, type a question about the document.

No API keys are required — every model runs locally within the notebook's own runtime.

## Example output

**Question:** *"What is the attention mechanism, and what are the query, key, and value matrices?"*

**Without retrieval:** a generic, plausible-sounding answer based only on the model's training data — with no way to verify it's actually describing *this* document.

**With retrieval:** the model is handed the paper's actual attention formula and multi-head attention description before answering, producing a response grounded in the retrieved text — along with the similarity scores behind that retrieval.

## Notes

Built as part of an Edureka AI Bootcamp exercise on Retrieval-Augmented Generation. The bootcamp's own course materials (problem statement, solution walkthrough, slides) are excluded from this repository as they're the provider's copyrighted content — only the implementation here is original work.
