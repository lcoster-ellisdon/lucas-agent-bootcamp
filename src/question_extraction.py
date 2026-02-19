"""
Extract questions from PDFs using OpenAI GPT.

Processes PDFs in the datasets folder and generates questions in the format:
{
    "question_text": "...",
    "page": int,
    "topic": "...",
    "evidence_snippet": "...",
    "confidence": float
}
"""

import json
import os
import re
from pathlib import Path
from typing import Optional

import pypdf
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client (using Gemini via OpenAI-compatible endpoint)
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
)

DATASETS_DIR = Path(__file__).parent.parent / "datasets"
OUTPUT_FILE = Path(__file__).parent.parent / "extracted_questions.json"


def extract_text_from_pdf(pdf_path: str) -> dict[int, str]:
    """
    Extract text from PDF, organized by page number.
    
    Returns:
        dict: {page_number: page_text}
    """
    pages = {}
    try:
        with open(pdf_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text = page.extract_text()
                pages[page_num + 1] = text  # 1-indexed pages
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
    return pages


def extract_topics_from_text(text: str) -> list[str]:
    """
    Extract logical topic candidates from text headings.
    Filters out non-useful headings based on irrelevant keywords.
    """
    topics = set()
    
    # Keywords that indicate non-useful headings (filter out)
    irrelevant_keywords = {
        'date', 'revision', 'page', 'of', 'control',
        'sponsor', 'canada', 'montreal', 'inc', 'ltd', 'ltd.',
        'pharmaceutical', 'address', 'phone', 'fax', 'e-mail',
        'table of contents', 'part', 'health professional',
        'excel plastic', 'containers', 'molecular', 'weight',
        'company'
    }
    
    lines = text.split('\n')
    for line in lines:
        stripped = line.strip()
        
        # Basic filtering: reasonable length, starts with capital letter
        if not (5 < len(stripped) < 150 and stripped[0].isupper()):
            continue
        
        # Skip lines that are mostly numbers or dates
        if any(c.isdigit() for c in stripped) and sum(1 for c in stripped if c.isdigit()) > len(stripped) * 0.3:
            continue
        
        # Check if it contains irrelevant keywords - skip if it does
        lower_line = stripped.lower()
        if any(keyword in lower_line for keyword in irrelevant_keywords):
            continue
        
        # Keep this heading as a topic
        topics.add(stripped)
    
    return list(topics)[:5]  # Return top 5 topics


def generate_question_with_gpt(
    page_content: str,
    topic: str,
    page_num: int
) -> Optional[dict]:
    """
    Use Gemini API to generate a question from page content.
    """
    prompt = f"""Given the following content from a document, generate ONE specific, factual question that can be answered from this text.

Topic: {topic}
Content: {page_content[:1500]}

Generate a question in this exact JSON format:
{{
    "question_text": "the specific question here",
    "evidence_snippet": "a relevant quote from the content that answers the question",
    "confidence": 0.8
}}

Only respond with valid JSON, no additional text."""

    try:
        response = client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Extract JSON from response (handles markdown code blocks)
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON if no code block
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
            else:
                print(f"  No JSON found in response for page {page_num}")
                return None
        
        result = json.loads(json_str)
        result["topic"] = topic
        
        # Ensure confidence is a float
        if "confidence" not in result:
            result["confidence"] = 0.8
        else:
            result["confidence"] = float(result["confidence"])
        
        return result
            
    except json.JSONDecodeError as e:
        print(f"  Failed to parse JSON for page {page_num}: {e}")
        return None
    except Exception as e:
        print(f"  Error calling Gemini for page {page_num}: {e}")
        return None


def process_pdfs() -> list[dict]:
    """
    Process all PDFs in the datasets folder and extract questions.
    Optimized to process first N pages only and limit topics per page.
    """
    if not DATASETS_DIR.exists():
        print(f"Creating datasets directory at {DATASETS_DIR}")
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Please add your PDF files to {DATASETS_DIR}")
        return []
    
    pdf_files = list(DATASETS_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {DATASETS_DIR}")
        return []
    
    all_questions = []
    max_pages_per_pdf = 5  # Limit pages to process per PDF
    max_topics_per_page = 2  # Limit topics per page
    
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        
        # Extract text by page
        pages = extract_text_from_pdf(str(pdf_file))
        
        # Process only first N pages
        for page_num in sorted(pages.keys())[:max_pages_per_pdf]:
            page_text = pages[page_num]
            
            if not page_text.strip():
                continue
            
            # Extract topics from this page
            topics = extract_topics_from_text(page_text)
            
            if not topics:
                # Use a generic topic if none found
                topics = ["General Content"]
            
            # Limit topics per page
            topics = topics[:max_topics_per_page]
            
            # Generate questions for each topic on this page
            for topic in topics:
                print(f"  Page {page_num}, Topic: {topic[:50]}...")
                
                question = generate_question_with_gpt(page_text, topic, page_num)
                
                if question:
                    all_questions.append(question)
    
    return all_questions


def normalize_question(text: str) -> str:
    """
    Normalize question text for similarity comparison.
    Removes numbers, case-normalizes, and removes specific details.
    """
    import re
    # Remove numbers
    normalized = re.sub(r'\d+', '[NUM]', text.lower())
    # Remove specific ages/measurements
    normalized = re.sub(r'\b(year|month|week|day|age|old|mg|ml|gram|kg)\b', '', normalized)
    # Remove extra whitespace
    normalized = ' '.join(normalized.split())
    return normalized


def deduplicate_questions(questions: list[dict], similarity_threshold: float = 0.75) -> list[dict]:
    """
    Remove logically similar/duplicate questions.
    Keeps first occurrence of each unique question pattern.
    """
    from difflib import SequenceMatcher
    
    unique_questions = []
    seen_normalized = []
    
    for question in questions:
        q_text = question.get("question_text", "")
        normalized = normalize_question(q_text)
        
        # Check if this normalized question is similar to any already seen
        is_duplicate = False
        for seen in seen_normalized:
            # Calculate similarity ratio
            ratio = SequenceMatcher(None, normalized, seen).ratio()
            if ratio >= similarity_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_questions.append(question)
            seen_normalized.append(normalized)
    
    return unique_questions


def save_questions(questions: list[dict]) -> None:
    """Save extracted questions to JSON file."""
    with open(OUTPUT_FILE, "w") as f:
        json.dump(questions, f, indent=2)
    print(f"\nSaved {len(questions)} questions to {OUTPUT_FILE}")

def extract_questions_from_path(pdf_path: str, max_pages: int = None, max_topics_per_page: int = 2) -> list[dict]:
    """
    Extract questions from a single PDF file path and return a list of question dicts.

    This runs the same per-page/topic generation flow as `process_pdfs` but for
    a single file and returns deduplicated results.
    """
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return []

    all_questions: list[dict] = []

    # Extract text by page
    pages = extract_text_from_pdf(str(pdf_path))

    # Process only first N pages
    for page_num in sorted(pages.keys())[:max_pages]:
        page_text = pages[page_num]

        if not page_text or not page_text.strip():
            continue

        # Extract topics from this page
        topics = extract_topics_from_text(page_text)

        if not topics:
            topics = ["General Content"]

        # Limit topics per page
        topics = topics[:max_topics_per_page]

        # Generate questions for each topic on this page
        for topic in topics:
            print(f"  Page {page_num}, Topic: {topic[:50]}...")
            question = generate_question_with_gpt(page_text, topic, page_num)
            if question:
                all_questions.append(question)

    # Deduplicate before returning
    if all_questions:
        all_questions = deduplicate_questions(all_questions)

    return all_questions

def main():
    """Main entry point."""
    print("PDF Question Extraction Tool")
    print("=" * 50)
    print(f"Datasets directory: {DATASETS_DIR}")
    print(f"Output file: {OUTPUT_FILE}")
    print()
    
    questions = process_pdfs()
    
    if questions:
        # Remove duplicate/similar questions
        questions = deduplicate_questions(questions)
        save_questions(questions)
        print("\nSample question:")
        print(json.dumps(questions[0], indent=2))
    else:
        print("No questions extracted.")


if __name__ == "__main__":
    main()
