import gradio as gr
import torch
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ==========================================
# 1. INITIALIZE ENGINE & VECTOR DATA
# ==========================================
print("Loading Lightweight Embedding Engine...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("Initializing High-Performance Qwen Core on CPU...")
chat_pipeline = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    device="cpu",
    dtype=torch.float32
)

# Fact-checking RAG context database
mock_news_db = [
    "AI and Automation: Global studies confirm that while AI will replace 85 million routine operational jobs by late 2026, it simultaneously offsets this by engineering 97 million technical roles.",
    "EV Industrial Lifecycle: Electric Vehicles emit zero greenhouse operational gases, but lithium-ion battery extraction increases heavy localized ecosystem toxicity by 42%.",
    "Universal Basic Income: Comprehensive economic trials indicate UBI successfully lowers base poverty indexes but sparks minor macro-inflationary pressures across basic retail commodities."
]
documents = [Document(page_content=text) for text in mock_news_db]
vector_store = FAISS.from_documents(documents, embedding_model)


# ==========================================
# 2. FEW-SHOT PROMPT COMPILER
# ==========================================
def execution_engine(workspace_type, user_input, max_tokens, temp, top_p):
    if not user_input.strip():
        return "⚠️ **System Input Error:** Workspace field cannot be processed empty."

    # 1. Structural Instructions + Few-Shot Examples (The Quality Anchors)
    if workspace_type == "debate":
        retriever = vector_store.as_retriever(search_kwargs={"k": 1})
        matched_docs = retriever.invoke(user_input)
        context = matched_docs[0].page_content if matched_docs else "No specific context available."
        
        system_instruction = (
            "You are a sharp, polite adversarial debate assistant. Disagree with the user's stance. "
            "Provide exactly one paragraph of logical counter-argument, then provide a score out of 10.\n\n"
            f"VERIFIED CONTEXT TO USE: {context}\n\n"
            "### GOOD EXAMPLE RESPONSE FORMAT:\n"
            "While your point highlights consumer convenience, verified data shows that global emissions from battery manufacturing offset initial green gains by up to 40%. Therefore, an immediate full transition remains structurally flawed.\n\n"
            "**Argument Logic Score: 6/10**"
        )
    else:  # Generalize mode
        system_instruction = (
            "You are an expert news editor. Translate complex, jargon-heavy statements into plain, simple English. "
            "Output your answer as clear, high-level summary bullet points under 50 words total.\n\n"
            "### GOOD EXAMPLE RESPONSE FORMAT:\n"
            "* Global markets fell sharply due to sudden central bank interest rate changes.\n"
            "* Local manufacturing activity slowed down to an all-time low."
        )

    # 2. Construct Prompt using Official Qwen ChatML Template format
    formatted_prompt = (
        f"<|im_start|>system\n{system_instruction}\n"
        "CRITICAL: Keep your response concise. Do not ramble. Complete your final sentence perfectly.<|im_end|>\n"
        f"<|im_start|>user\n{user_input}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    # 3. CPU Execution Pipeline
    try:
        raw_output = chat_pipeline(
            formatted_prompt,
            max_new_tokens=int(max_tokens),
            do_sample=True,
            temperature=float(temp),
            top_p=float(top_p),
            top_k=40,
            repetition_penalty=1.2
        )
        
        processed_text = raw_output[0]["generated_text"].split("<|im_start|>assistant\n")[-1].strip()
        
        # Programmatic Clean Up: Ensure no trailing cut-off sentences are displayed
        if processed_text and processed_text[-1] not in [".", "!", "?", '"', "*"]:
            last_punctuation = max(processed_text.rfind("."), processed_text.rfind("!"), processed_text.rfind("?"))
            if last_punctuation != -1:
                processed_text = processed_text[:last_punctuation + 1]
                
        return processed_text

    except Exception as error:
        return f"❌ **Pipeline Fault:** Execution error on CPU node. Details: {str(error)}"


# ==========================================
# 3. MODULAR UI DASHBOARD (GRADIO 6.0)
# ==========================================
with gr.Blocks() as demo:
    
    # Header Module
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("# 🤖 AI News Debate & Summary Agent Suite")
            gr.Markdown(
                "**About This Agent:** This framework utilizes localized vector embeddings alongside "
                "few-shot prompting to anchor small model reasoning. It delivers deterministic, "
                "concise outputs optimized for zero-error execution on base CPU clusters. "
                "This framework processes user statements and news artifacts to provide dual NLP functions: " 
                "*Adversarial Debate Environment* using real-time vectorized context verification to engage with users "
                "and *Automated Generalization Engine* for complex-news summarization and analysis."
            )
        
        # Status Module Panel
        with gr.Column(scale=1):
            gr.HighlightedText(value=[("ONLINE", "🟢 NLP Match Engine Status")], show_label=False)
            gr.Markdown(
                "🪪 **Agent Profile:** Few-Shot Context Assistant  \n"
                "📦 **Model Matrix:** Qwen2.5-1.5B-Instruct  \n"
                "🧬 **Vectorization Engine:** Text FAISS Engine  \n"
                "💻 **Processing Hardware:** Base CPU Tier"
            )

    gr.HTML("<hr style='border: 0.5px solid #374151; margin: 15px 0;'/>")

    # Lower Workspace Splitting Layout
    with gr.Row():
        
        # Left Panel Workspace
        with gr.Column(scale=1):
            gr.Markdown("### 🛠️ Agent Workspaces")
            
            with gr.Tabs() as active_workspace_tabs:
                
                # Tab (i): Debate Module
                with gr.Tab(label="🎙️ News Debate Agent", id=0):
                    gr.Markdown("*State a topic and your stance. The agent will pull factual context, debate you, and score your logic.*")
                    
                    # Robust Example System: Swapped to an immune dropdown component to bypass the hidden tab rendering bug
                    debate_example_picker = gr.Dropdown(
                        label="💡 Suggestions for Preset Debate Topic",
                        choices=[
                            "LLMs are indispensable for application development.",
                            "Electric vehicles are the only way to save the environment.",
                            "Universal Basic Income will just make people lazy."
                        ]
                    )
                    
                    debate_text_input = gr.Textbox(
                        label="Your Stance", 
                        placeholder="Select or paste your claim here..."
                    )
                    
                    # Dynamically push selected dropdown choice into the text input instantly
                    debate_example_picker.change(fn=lambda x: x, inputs=debate_example_picker, outputs=debate_text_input)
                    
                    run_debate_action = gr.Button("Deploy Counter-Argument & Score", variant="primary")

                # Tab (ii): Text Simplification Module
                with gr.Tab(label="📰 Snippet Generalizer", id=1):
                    gr.Markdown("*Paste complex, jargon-heavy news columns here to output plain-English, executive summaries.*")
                    
                    # Robust Example System: Swapped to an immune dropdown component to bypass the hidden tab rendering bug
                    generalizer_example_picker = gr.Dropdown(
                        label="💡 Examples for Preset Complex News",
                        choices=[
                            "Despite aggressive macroeconomic headwinds and structural quantitative tightening by central banking systems, retail index trends demonstrated non-linear elasticity during fiscal evaluations.",
                            "The ecosystem validation vectors indicate substantial degradation in sub-surface lithospheric layers owing to chemical manufacturing runoffs, sparking compliance friction."
                        ],
                        value="Despite aggressive macroeconomic headwinds and structural quantitative tightening by central banking systems, retail index trends demonstrated non-linear elasticity during fiscal evaluations."
                    )
                    
                    simplify_text_input = gr.Textbox(
                        label="Complex News Snippet", 
                        lines=4,
                        value="Despite aggressive macroeconomic headwinds and structural quantitative tightening by central banking systems, retail index trends demonstrated non-linear elasticity during fiscal evaluations.",
                        placeholder="Paste text block or official reports here..."
                    )
                    
                    # Dynamically push selected dropdown choice into the text input instantly
                    generalizer_example_picker.change(fn=lambda x: x, inputs=generalizer_example_picker, outputs=simplify_text_input)
                    
                    run_simplifier_action = gr.Button("Deploy for generalization system", variant="primary")
            
            gr.HTML("<div style='margin-bottom: 20px;'></div>")
                       
            
            # Param Configuration Module (Hidden Tuning to keep CPU fast and reliable)
            with gr.Accordion("⚙️ Runtime Parameter Tuning", open=False):
                max_tokens_val = gr.Slider(minimum=64, maximum=256, value=140, step=16, label="Max tokens")
                temp_val = gr.Slider(minimum=0.1, maximum=1.0, value=0.25, step=0.05, label="Temperature (Low = Deterministic)")
                top_p_val = gr.Slider(minimum=0.50, maximum=1.00, value=0.85, step=0.05, label="Top-p")

        # Right Panel Output Monitor
        with gr.Column(scale=1):
            gr.Markdown("### 🖥️ Active Output Stream")
            active_output_panel = gr.Markdown(
                value="*System ready. Select a workspace on the left panel and click deploy to review parsed generation stream.*"
            )

    # ==========================================
    # 4. EVENT CONTROLLERS
    # ==========================================
    run_debate_action.click(
        fn=lambda user_in, tokens, t, p: execution_engine("debate", user_in, tokens, t, p),
        inputs=[debate_text_input, max_tokens_val, temp_val, top_p_val],
        outputs=active_output_panel
    )

    run_simplifier_action.click(
        fn=lambda user_in, tokens, t, p: execution_engine("generalize", user_in, tokens, t, p),
        inputs=[simplify_text_input, max_tokens_val, temp_val, top_p_val],
        outputs=active_output_panel
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
