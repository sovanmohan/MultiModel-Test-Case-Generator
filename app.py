import os
import gradio as gr
from llm import response

with gr.Blocks() as llm:
    slider = gr.Slider(10, 100, render=False)
    chatbot = gr.ChatInterface(
        fn=response,
        multimodal=True,
        title="MultiModel LLM for Testcase generation",
        cache_examples=True,
    )

llm.launch(debug=True)
