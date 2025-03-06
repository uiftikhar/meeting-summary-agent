def read_transcript(file_path: str) -> str:
    """
    Read and return the contents of a transcript file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        transcript = f.read()
    return transcript