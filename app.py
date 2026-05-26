import gradio as gr
from transformers import pipeline

# Lightweight summarization model
summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6"
)

# Function
def summarize_article(article, max_len, min_len):

    if article.strip() == "":
        return "Please enter an article."

    summary = summarizer(
        article,
        max_length=max_len,
        min_length=min_len,
        do_sample=False
    )

    return summary[0]['summary_text']


# Gradio Interface
interface = gr.Interface(
    fn=summarize_article,

    inputs=[
        gr.Textbox(
            lines=15,
            label="Input Article",
            placeholder="Paste your article here..."
        ),

        gr.Slider(
            minimum=50,
            maximum=200,
            value=120,
            step=10,
            label="Max Length"
        ),

        gr.Slider(
            minimum=20,
            maximum=100,
            value=40,
            step=5,
            label="Min Length"
        )
    ],

    outputs=gr.Textbox(
        lines=8,
        label="Generated Summary"
    ),

    title="AI Article Summarizer",
    description="Summarize long articles using Hugging Face Transformers and Gradio.",

    theme=gr.themes.Soft()
)

interface.launch()