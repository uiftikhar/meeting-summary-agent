import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from ingest import read_transcript
from summarizer import create_t5_model, generate_summary_with_overlap
from pathlib import Path
from feedback import store_feedback
def main():
    print("MAIN")
    # Get the directory o f the current file (src/file.py)
    current_dir = Path(__file__).resolve().parent
    print("The path to test.txt is:", current_dir)

    # Construct the path to test.txt: go up one level to root, then into data
    transcript_file = current_dir.parent / 'data' / 'Transcript-CPL-BTO-Tech-Handover.txt'

    transcript = read_transcript(transcript_file)

    # Create summarizer pipeline
    tokenizer, model = create_t5_model("t5-base")

    # Generate the summary
    final_summary = generate_summary_with_overlap(transcript, tokenizer, model)
    print("SUMMARY\n", final_summary)

    # Prompt feedback from user
    feedback = input("\n Is the summary correct> (y/n): ").strip().lower()
    if feedback != "y":
        user_corrected_summary = input("Enter the corrected summary: ").strip()
        store_feedback(transcript, generated_summary, user_corrected_summary)
    else:
        print("Summary accepted. No feedback provided.")


if __name__ == "__main__":
    main()
