from transformers import T5ForConditionalGeneration, T5Tokenizer
import nltk
from nltk.tokenize import sent_tokenize


nltk.download('punkt', quiet=True)

def create_t5_model(model_name="t5-base"):
    """
    Loads and returns the T5 tokenizer and model.
    """
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model

def split_text_with_overlap(text, max_words=700, overlap_words=100):
    """
    Splits text into overlapping chunks using sentence boundaries.
    Each chunk contains up to max_words words and overlaps the previous chunk by about overlap_words.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        sentence_word_count = len(sentence.split())
        if current_word_count + sentence_word_count <= max_words:
            current_chunk.append(sentence)
            current_word_count += sentence_word_count
        else:
            # Save the current chunk
            chunks.append(" ".join(current_chunk))
            # Create an overlap by taking the last sentences until reaching overlap_words
            overlap = []
            overlap_count = 0
            for s in reversed(current_chunk):
                s_words = len(s.split())
                if overlap_count + s_words <= overlap_words:
                    overlap.insert(0, s)
                    overlap_count += s_words
                else:
                    break
            current_chunk = overlap + [sentence]
            current_word_count = overlap_count + sentence_word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def summarize_chunk(text_chunk, tokenizer, model, max_length=150, min_length=30):
    """
    Summarizes a single text chunk using T5.
    """
    prompt = "summarize: " + text_chunk
    input_ids = tokenizer.encode(prompt, return_tensors="pt", truncation=True)
    summary_ids = model.generate(
        input_ids,
        max_length=max_length,
        min_length=min_length,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def generate_final_structured_summary(combined_summary, tokenizer, model, max_length=400):
    structured_prompt = (
        "Generate a structured meeting summary with the following format:\n"
        "Title: <title>\nSummary: <summary>\nExplanation: <detailed explanation>\n\n"
        "Meeting Summary: " + combined_summary
    )
    
    # Print the prompt for debugging.
    print("Structured prompt:\n", structured_prompt)
    
    input_ids = tokenizer.encode(structured_prompt, return_tensors="pt", truncation=True)
    output_ids = model.generate(
        input_ids,
        max_length=max_length,
        num_beams=4,
        early_stopping=True
    )
    final_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print("final_output\n", final_output)
    return final_output

def generate_summary_with_overlap(text, tokenizer, model, 
                                  chunk_max_words=700, chunk_overlap_words=100,
                                  max_length=150, min_length=30):
    """
    Splits the input text into overlapping chunks, summarizes each, then combines and re-summarizes them using a structured prompt.
    """
    chunks = split_text_with_overlap(text, max_words=chunk_max_words, overlap_words=chunk_overlap_words)
    print(f"Total chunks: {len(chunks)}")
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        summary = summarize_chunk(chunk, tokenizer, model, max_length=max_length, min_length=min_length)
        chunk_summaries.append(summary)
        print(f"Chunk {i+1} summary: {summary}")
    
    combined_summary_text = " ".join(chunk_summaries)
    final_summary = generate_final_structured_summary(combined_summary_text, tokenizer, model, max_length=250)
    return final_summary