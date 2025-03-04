# src/main.py

from transformers import pipeline

def create_summarizer():
    
    print("MAIN")
    # This loads a pre-trained summarization model
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer

if __name__ == "__main__":
    summarizer = create_summarizer()
    example_text = """
    Team, we need to finish the new feature by Monday. 
    John, can you make sure the front-end is ready? 
    Sarah, handle the back-end integration. 
    We need testing by Wednesday.
    """
    summary = summarizer(example_text, max_length=60, min_length=10, do_sample=False)
    print("Summary:", summary[0]["summary_text"])
