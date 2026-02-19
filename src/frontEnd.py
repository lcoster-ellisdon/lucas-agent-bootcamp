import gradio as gr
import os
import asyncio
import question_extraction
from pageIndexAgents import interactive_query

with gr.Blocks(title="RFP Generator") as demo:
    gr.Markdown("# Agent-Based RFP Exploration Tool")
    
    # State to store extracted questions
    questions_state = gr.State([])
    
    with gr.Tabs():
        # Tab 1: Upload PDF and Extract Questions
        with gr.Tab("📄 Upload PDF"):
            gr.Markdown("### Extract Questions from PDF")
            
            with gr.Row():
                pdf_file = gr.File(
                    label="Upload PDF Document",
                    file_types=[".pdf"],
                    type="filepath"
                )
            
            extract_btn = gr.Button("Extract Questions", variant="primary", size="lg")
            loading_bar = gr.HTML()
            extraction_status = gr.Markdown()
            
            # Container for questions and checkboxes
            with gr.Column(visible=False) as questions_section:
                gr.Markdown("### Select Questions to Explore")
                selected_questions = gr.CheckboxGroup(
                    label="Questions",
                    choices=[],
                    info="Select one or more questions to explore"
                )
                explore_btn = gr.Button("🔍 Explore Selected Questions", variant="primary", size="lg")
                explore_output = gr.Markdown()
            
            def extract_questions_from_pdf(pdf_file_path):
                """Extract questions and return status + HTML for interactive question bars."""
                if pdf_file_path is None:
                    return "", "", [], gr.update(visible=False)

                filename = os.path.basename(pdf_file_path)

                # Simple animated indeterminate progress bar (HTML/CSS)
                loading_html = (
                    "<div style='padding:8px 0'>Processing...<div style='height:10px;background:#eee;border-radius:6px;overflow:hidden;margin-top:8px;'>"
                    "<div style='width:30%;height:100%;background:linear-gradient(90deg,#3b82f6,#60a5fa);animation:progress 1.2s linear infinite;'></div></div></div>"
                    "<style>@keyframes progress{0%{transform:translateX(-200%)}100%{transform:translateX(200%)}}</style>"
                )

                # First yield shows the loading bar
                yield loading_html, "", [], gr.update(visible=False)

                try:
                    questions = question_extraction.extract_questions_from_path(pdf_file_path)
                except Exception as e:
                    yield "", f"❌ Error during extraction: {e}", [], gr.update(visible=False)
                    return

                if not questions:
                    yield "", f"⚠️ No questions extracted from '{filename}'.", [], gr.update(visible=False)
                    return

                status = f"✅ **Successfully extracted {len(questions)} questions from '{filename}'**"

                # Final yield returns status and HTML with explore button visible
                yield "", status, questions, gr.update(visible=True)
            
            extract_btn.click(
                extract_questions_from_pdf,
                inputs=[pdf_file],
                outputs=[loading_bar, extraction_status, questions_state, questions_section]
            )
            
            def update_questions_display(questions):
                """Update the checkbox group with extracted questions."""
                if not questions:
                    return gr.update(choices=[])
                
                # Extract question text for display
                question_texts = [
                    q.get('question_text') or q.get('text') or '<no text>'
                    for q in questions
                ]
                
                return gr.update(choices=question_texts, value=[])
            
            questions_state.change(
                update_questions_display,
                inputs=[questions_state],
                outputs=[selected_questions]
            )
            
            def handle_explore_click(selected_questions_list):
                """Handle the explore button click - pass selected question strings directly to interactive_query."""
                if not selected_questions_list:
                    return "⚠️ Please select at least one question to explore."
                
                # selected_questions_list is already a list of question strings from the CheckboxGroup
                """
                try:
                    # Create a simple doc_registry (you may want to pass this from the PDF extraction)
                    doc_registry = {}
                    
                    # Run the async function with the question strings directly
                     results = asyncio.run(interactive_query(selected_questions_list, doc_registry))
                    
                    # Format results for display
                    output = "## Exploration Results\n\n"
                    for result in results:
                        output += f"### Question: {result['question']}\n"
                        output += f"**Top Documents:**\n"
                        for doc in result['ranked_docs'][:3]:
                            output += f"- Doc: {doc['doc_id']} (Score: {doc['score']:.3f})\n"
                        
                        if result['responses']:
                            output += f"\n**Answers:**\n"
                            for i, response in enumerate(result['responses'], 1):
                                output += f"{i}. {response.get('answer', 'No answer')}\n"
                                if response.get('sources'):
                                    sources_str = ', '.join([f"pg {s.get('page', '?')}" for s in response['sources']])
                                    output += f"   **Sources:** {sources_str}\n"
                        else:
                            output += "No answers found.\n"
                        
                        output += "\n---\n"
                    
                    return output
                except Exception as e:
                    return f"❌ Error during exploration: {str(e)}" 
                    """
            
            explore_btn.click(
                handle_explore_click,
                inputs=[selected_questions],
                outputs=[explore_output]
            )

if __name__ == "__main__":
    demo.launch(server_port=7860, share=True)