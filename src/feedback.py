import os
import json

def store_feedback(transcript: str, generated_summary: str, user_corrected_summary: str, feedback_file="feedback.json"):
    """
    Store user feedback (the corrected summary) along with the transcript and generated summary.
    This data can later be used to fine-tune the summarizer or serve as a retrieval memory.
    """
    feedback_entry = {
        "transcript": transcript,
        "generated_summary": generated_summary,
        "user_corrected_summary": user_corrected_summary
    }
    
    # Load existing feedback if available
    if os.path.exists(feedback_file):
        with open(feedback_file, "r", encoding="utf-8") as f:
            feedback_data = json.load(f)
    else:
        feedback_data = []

    feedback_data.append(feedback_entry)

    # Write updated feedback to the file
    with open(feedback_file, "w", encoding="utf-8") as f:
        json.dump(feedback_data, f, indent=4)
    print("Feedback saved for future learning.")
