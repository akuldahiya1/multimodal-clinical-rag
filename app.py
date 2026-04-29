import sys, os
os.environ["PATH"] = "/home/aakul001/.conda/envs/rag310/bin:" + os.environ.get("PATH","")
sys.path.insert(0, os.path.expanduser("~/multimodal_rag"))
os.environ["JAVA_HOME"] = "/home/aakul001/.conda/envs/rag310/lib/jvm"
os.environ["JVM_PATH"]  = "/home/aakul001/.conda/envs/rag310/lib/jvm/lib/server/libjvm.so"
os.environ["HF_TOKEN"]  = "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"

import gradio as gr
import pandas as pd
import tempfile
from pathlib import Path
from src.utils import setup_java
setup_java()

from src.retrieval import (load_bm25, load_text_model, load_faiss_text,
                            load_faiss_image, load_metadata,
                            search_dense_image_from_query)
from src.generation import load_llm
from src.query_understanding import classify_query
from src.pdf_rag import (build_pdf_index, hybrid_search_with_pdf,
                          extract_pdf_page_images, reset_pdf_index,
                          format_pdf_context)

print("Loading models...")
load_bm25()
load_text_model()
load_faiss_text()
load_faiss_image()

import src.retrieval as _ret
_ret._metadata_df = None
load_metadata()
load_llm()
print("All models ready")

LANGUAGES = {
    "English": None, "Spanish": "es", "French": "fr",
    "German": "de", "Hindi": "hi", "Arabic": "ar",
    "Chinese": "zh", "Japanese": "ja", "Portuguese": "pt", "Italian": "it",
}

def translate_text(text, lang_code):
    if not lang_code:
        return text
    try:
        from deep_translator import GoogleTranslator
        result = GoogleTranslator(source="en", target=lang_code).translate(text)
        return result if result else text
    except Exception:
        return text

def transcribe_audio(audio_path):
    if not audio_path:
        return ""
    try:
        import whisper, subprocess
        wav = str(audio_path) + "_conv.wav"
        subprocess.run(["ffmpeg","-y","-i",str(audio_path),
                        "-ar","16000","-ac","1","-f","wav",wav],
                       capture_output=True)
        use = wav if Path(wav).exists() else str(audio_path)
        m   = whisper.load_model("base")
        r   = m.transcribe(use, fp16=False, language="en")
        return r.get("text","").strip()
    except Exception as e:
        print(f"Audio error: {e}")
        return ""

def clean_answer(answer):
    for a in ["Best regards","Let me know","Please let me know",
              "Radiologist &","[Your Name]","Note:","Also, I can",
              "I can provide","If you need","---"]:
        if a in answer:
            answer = answer[:answer.find(a)].strip()
    if "." in answer:
        answer = answer[:answer.rfind(".")+1]
    return answer.strip() or "Insufficient evidence in retrieved passages."

def handle_voice(audio_path):
    if not audio_path:
        return "", "No audio detected."
    t = transcribe_audio(audio_path)
    if t:
        return t, "Voice transcribed successfully."
    return "", "Could not transcribe. Try again."

def handle_pdf_upload(pdf_path):
    if not pdf_path:
        reset_pdf_index()
        return "No PDF loaded.", []
    try:
        info  = build_pdf_index(pdf_path)
        pages = extract_pdf_page_images(pdf_path, max_pages=4)
        msg   = f"Loaded: {info['name']} | {info['chunks']} chunks | {info['pages']} pages"
        return msg, pages
    except Exception as e:
        return f"Error: {e}", []

def search(query, audio_file, pdf_file, n_passages, language):
    if not query or not query.strip():
        return "", "", "", "", [], "Please enter a question."

    analysis   = classify_query(query)
    query_type = analysis["type"]

    results, routing = hybrid_search_with_pdf(query, top_k=10)

    type_info = (
        "Query type: " + query_type.upper() + "\n"
        "Strategy: "   + routing["strategy"].replace("_"," ").upper() + "\n"
        "PDF score: "  + str(routing["pdf_score"]) + "\n"
        "PDF weight: " + str(routing["pdf_weight"]) +
        " | Global: "  + str(routing["global_weight"]) + "\n"
        "Entities: "   + str(list(analysis["entities"].keys()) if analysis["entities"] else "None")
    )

    passages_md = ""
    image_paths = []

    for r in results[:int(n_passages)]:
        if r.get("from_pdf") and r.get("page"):
            label = f"**[PDF Page {r['page']}]**"
        else:
            label = f"**[{r.get('modality','text').upper()}]**"
        passages_md += f"{label}\n{str(r.get('contents',''))[:280]}...\n\n"
        if r.get("modality") == "image":
            ip = str(r.get("image_path",""))
            if ip and ip != "nan" and Path(ip).exists():
                image_paths.append(ip)

    if pdf_file and routing["pdf_weight"] > 0.3:
        pdf_imgs = extract_pdf_page_images(str(pdf_file), max_pages=2)
        image_paths = pdf_imgs + image_paths

    if query_type == "visual":
        img_res = search_dense_image_from_query(query, top_k=4)
        if img_res is not None and not img_res.empty:
            for _, row in img_res.iterrows():
                ip = str(row.get("image_path",""))
                if ip and ip != "nan" and Path(ip).exists():
                    image_paths.append(ip)

    context = format_pdf_context(results, n=int(n_passages))
    from src.prompts import build_prompt, clean_cot_answer
    import src.generation as gen
    import torch

    prompt = build_prompt(query, [{"contents": context, "modality":"text"}],
                          query_type=query_type, use_cot=True)
    tok, model = gen.load_llm()
    device = next(model.parameters()).device
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=3072).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=300, do_sample=False,
                             pad_token_id=tok.eos_token_id,
                             eos_token_id=tok.eos_token_id)
    full   = tok.decode(out[0], skip_special_tokens=True)
    answer = clean_answer(clean_cot_answer(full, query_type))

    lang_code = LANGUAGES.get(language)
    if lang_code:
        translated = translate_text(answer, lang_code)
        answer = "[" + language + "]\n" + translated + "\n\n[English]\n" + answer

    pdf_count = sum(1 for r in results if r.get("from_pdf"))
    stats = (
        "Strategy: " + routing["strategy"].replace("_"," ") + "\n"
        "PDF chunks: " + str(pdf_count) + " | Global: " + str(len(results)-pdf_count) + "\n"
        "Images found: " + str(len(image_paths)) + "\n"
        "Query type: " + query_type
    )

    return answer, passages_md, type_info, stats, image_paths[:6], "Done"


CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display:ital@0;1&display=swap');

:root {
    --navy:    #0D1B2A;
    --teal:    #1B4F72;
    --accent:  #2E86AB;
    --light:   #E8F4F8;
    --white:   #FFFFFF;
    --grey:    #6B7280;
    --border:  #D1E3EE;
    --success: #0F766E;
    --text:    #1A2332;
}

* { box-sizing: border-box; }

body, .gradio-container {
    font-family: 'DM Sans', sans-serif !important;
    background: #F0F6FA !important;
    color: var(--text) !important;
    max-width: 1400px !important;
    margin: 0 auto !important;
}

/* Header */
.rag-header {
    background: linear-gradient(135deg, var(--navy) 0%, var(--teal) 100%);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.rag-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(46,134,171,0.3) 0%, transparent 70%);
    border-radius: 50%;
}
.rag-header h1 {
    font-family: 'DM Serif Display', serif !important;
    font-size: 28px !important;
    color: white !important;
    margin: 0 0 8px 0 !important;
    font-weight: 400 !important;
    letter-spacing: -0.3px;
}
.rag-header p {
    color: rgba(255,255,255,0.75) !important;
    font-size: 14px !important;
    margin: 0 0 16px 0 !important;
    font-weight: 300 !important;
}
.pill-row { display: flex; flex-wrap: wrap; gap: 8px; }
.pill {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    color: white !important;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
    backdrop-filter: blur(4px);
}

/* Cards */
.card {
    background: white;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
}

/* Input styling */
.gradio-container textarea,
.gradio-container input[type=text] {
    background: white !important;
    color: var(--text) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
    padding: 12px 16px !important;
    transition: border-color 0.2s !important;
}
.gradio-container textarea:focus,
.gradio-container input[type=text]:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(46,134,171,0.12) !important;
    outline: none !important;
}

/* Labels */
label, .gradio-container label {
    color: var(--navy) !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: 0.3px !important;
    text-transform: uppercase !important;
}

/* Buttons */
.gradio-container button.primary {
    background: linear-gradient(135deg, var(--teal), var(--accent)) !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 12px 28px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 12px rgba(27,79,114,0.3) !important;
}
.gradio-container button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(27,79,114,0.4) !important;
}
.gradio-container button.secondary {
    background: white !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--teal) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    transition: all 0.2s !important;
}
.gradio-container button.secondary:hover {
    border-color: var(--accent) !important;
    background: var(--light) !important;
}

/* Example buttons */
.example-btn button {
    background: var(--light) !important;
    border: 1px solid var(--border) !important;
    border-radius: 20px !important;
    color: var(--teal) !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 6px 14px !important;
    transition: all 0.15s !important;
}
.example-btn button:hover {
    background: var(--accent) !important;
    color: white !important;
    border-color: var(--accent) !important;
}

/* Status bar */
.status-bar {
    background: linear-gradient(90deg, var(--light), white);
    border-left: 3px solid var(--accent);
    border-radius: 0 8px 8px 0;
    padding: 8px 16px;
    font-size: 13px;
    color: var(--teal);
    font-weight: 500;
}

/* Answer box */
.answer-box textarea {
    background: white !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 12px !important;
    font-size: 14px !important;
    line-height: 1.7 !important;
    color: var(--text) !important;
}

/* Analysis boxes */
.analysis-box textarea {
    background: #F7FBFD !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    font-size: 12.5px !important;
    font-family: 'DM Sans', monospace !important;
    color: var(--navy) !important;
}

/* Gallery */
.gradio-gallery {
    border: 1.5px solid var(--border) !important;
    border-radius: 12px !important;
    background: white !important;
    padding: 12px !important;
}

/* Divider */
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 16px;
    color: var(--navy);
    margin: 20px 0 12px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid var(--light);
}

/* Footer */
.rag-footer {
    background: white;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 24px;
    margin-top: 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
"""

with gr.Blocks(css=CSS, title="Clinical RAG | Akul Dahiya") as demo:

    gr.HTML("""
    <div class="rag-header">
        <h1>Multimodal Clinical RAG System</h1>
        <p>Intelligent biomedical search across text, images, audio and PDFs &mdash; powered by BioLORD + BiomedCLIP + Llama-3.2</p>
        <div class="pill-row">
            <span class="pill">679,137 passages</span>
            <span class="pill">P@10 = 0.93</span>
            <span class="pill">NDCG@10 = 0.99</span>
            <span class="pill">MRR = 1.00</span>
            <span class="pill">10,000 images</span>
            <span class="pill">Smart PDF routing</span>
            <span class="pill">10 languages</span>
            <span class="pill">Voice input</span>
        </div>
    </div>
    """)

    # Main search area
    with gr.Group():
        query_box = gr.Textbox(
            label="Medical Question",
            placeholder="Ask anything about biomedical research, or upload a PDF to chat with it...",
            lines=3,
            elem_classes=["answer-box"],
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="Voice Input",
                    type="filepath",
                    sources=["microphone","upload"],
                )
                with gr.Row():
                    transcribe_btn = gr.Button("Transcribe", variant="secondary", size="sm")
                    voice_status   = gr.Textbox(show_label=False, lines=1,
                                                placeholder="Voice status...",
                                                interactive=False, scale=2)

            with gr.Column(scale=1):
                pdf_input  = gr.File(label="Upload PDF", file_types=[".pdf"])
                pdf_status = gr.Textbox(show_label=False, lines=1,
                                        placeholder="PDF status...",
                                        interactive=False)

            with gr.Column(scale=1):
                n_passages = gr.Slider(minimum=1, maximum=10, value=5, step=1,
                                       label="Passages")
                language   = gr.Dropdown(choices=list(LANGUAGES.keys()),
                                         value="English", label="Output Language")
                search_btn = gr.Button("Search", variant="primary", size="lg")

    # Example queries
    gr.HTML('<p style="font-size:12px;font-weight:600;color:#6B7280;text-transform:uppercase;letter-spacing:0.5px;margin:16px 0 8px;">Quick Examples</p>')
    with gr.Row(elem_classes=["example-btn"]):
        gr.Button("COVID-19 + Healthcare", size="sm").click(
            fn=lambda: "COVID-19 impact on healthcare systems", outputs=query_box)
        gr.Button("Chest X-ray Pneumonia", size="sm").click(
            fn=lambda: "chest X-ray findings in pneumonia", outputs=query_box)
        gr.Button("Antibiotic Resistance", size="sm").click(
            fn=lambda: "antibiotic resistance mechanisms in bacteria", outputs=query_box)
        gr.Button("Telemedicine Rural", size="sm").click(
            fn=lambda: "benefits of telemedicine for rural patients", outputs=query_box)
        gr.Button("MRI Brain Scan", size="sm").click(
            fn=lambda: "MRI brain scan interpretation neurological disorders", outputs=query_box)
        gr.Button("PDF Findings", size="sm").click(
            fn=lambda: "What are the key findings in this paper?", outputs=query_box)

    status_box = gr.Textbox(
        value="Ready to search.",
        show_label=False,
        interactive=False,
        lines=1,
        elem_classes=["status-bar"],
    )

    # Results
    gr.HTML('<p class="section-title">Results</p>')
    with gr.Row():
        with gr.Column(scale=3):
            answer_box = gr.Textbox(
                label="Generated Answer",
                lines=11,
                interactive=False,
                elem_classes=["answer-box"],
            )
        with gr.Column(scale=1):
            analysis_box = gr.Textbox(
                label="Query Analysis",
                lines=6,
                interactive=False,
                elem_classes=["analysis-box"],
            )
            stats_box = gr.Textbox(
                label="Retrieval Stats",
                lines=5,
                interactive=False,
                elem_classes=["analysis-box"],
            )

    gr.HTML('<p class="section-title">Retrieved Passages</p>')
    passages_box = gr.Markdown()

    gr.HTML('<p class="section-title">Medical Images + PDF Pages</p>')
    image_gallery = gr.Gallery(
        label="",
        columns=4,
        height=260,
        object_fit="contain",
        elem_classes=["gradio-gallery"],
    )

    gr.HTML("""
    <div class="rag-footer">
        <div>
            <strong style="color:#0D1B2A;font-family:'DM Serif Display',serif;font-size:15px;">
                CS734 &mdash; Introduction to Information Retrieval
            </strong><br>
            <span style="color:#6B7280;font-size:13px;">Akul Dahiya &nbsp;&bull;&nbsp; Old Dominion University &nbsp;&bull;&nbsp; Spring 2026</span>
        </div>
        <div style="text-align:right;color:#6B7280;font-size:12px;line-height:1.8;">
            Pyserini BM25 &bull; BioLORD-2023-C &bull; FAISS &bull; BiomedCLIP<br>
            RRF Fusion &bull; MiniLM Reranker &bull; Llama-3.2-3B &bull; Whisper &bull; deep-translator
        </div>
    </div>
    """)

    # Wire events
    transcribe_btn.click(fn=handle_voice, inputs=[audio_input],
                         outputs=[query_box, voice_status])
    pdf_input.change(fn=handle_pdf_upload, inputs=[pdf_input],
                     outputs=[pdf_status, image_gallery])

    s_in  = [query_box, audio_input, pdf_input, n_passages, language]
    s_out = [answer_box, passages_box, analysis_box, stats_box, image_gallery, status_box]
    search_btn.click(fn=search, inputs=s_in, outputs=s_out)
    query_box.submit(fn=search, inputs=s_in, outputs=s_out)

tmp_dir = tempfile.gettempdir()
demo.launch(
    server_name="0.0.0.0",
    server_port=7895,
    share=True,
    show_error=True,
    allowed_paths=[
        "/home/aakul001/multimodal_rag/data/images/roco",
        "/home/aakul001/multimodal_rag/data/images/pmcvqa",
        tmp_dir,
    ]
)
