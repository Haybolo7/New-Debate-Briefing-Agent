# 🌐 AI News Debate & Summary Agent Suite
The AI News Debate & Summary Agent Suite is a highly optimized, modular NLP workstation deployed via Hugging Face Spaces. Engineered specifically to deliver low-latency, zero-error execution on constrained computational environments (Basic shared CPU instances), this system provides dual workspaces for adversarial debate reasoning and technical text simplification.

## 🚀 Core Architecture & Tech Stack
1.**Interface Layer**: Built on Gradio 6.0, employing a clean, scannable dashboard hierarchy divided into interactive control panels and active output monitors.
2.**Inference Engine**: Powered by Qwen/Qwen2.5-1.5B-Instruct, an ultra-dense micro-large language model offering advanced reasoning, strict instruction-following capabilities, and high structural resilience.
3.**Retrieval-Augmented Generation (RAG)**: Utilizes a local FAISS Vector Database combined with sentence-transformers/all-MiniLM-L6-v2 embeddings to inject verified background facts into system prompts dynamically.

## 🛠️ Key Workspaces
1.🎙️ **News Debate Agent** : Evaluates a user's stated opinion on a given topic, retrieves factual, real-time contextual context from the semantic data store, generates a concise paragraph containing a logical counter-argument, and provides an objective Argument Logic Score.
2. 📰 **Snippet Generalizer**: Acts as an executive editor by ingesting complex, jargon-heavy technical, financial, or ecological statements and translating them into clear, simple, plain-English bullet points.

## ⚡ Optimization & Guardrail Engineering
To eliminate execution timeouts, mid-sentence truncation, and memory spikes standard with CPU deployments, the project implements four explicit guardrails:-
1. *Few-Shot Prompt Anchoring*: Explicit structural templates give the 1.5B model direct context patterns to imitate, reducing rambling and improving correctness.
2. *Deterministic Parameter Tuning*: Keeps temperature low ($0.25$) and repetition penalties high ($1.2$) to maintain fast, predictable token generation.
3. *Context Constraints*: Tight bounds on maximum tokens ($140$–$160$) and data retrieval ($k=1$) keep the prompt footprint small for rapid execution.
4. *Programmatic Tail Post-Processing*: Automated Python string parsers slice off incomplete sentences if the model gets cut off by hard token limits.
