import re
import json

def parse_inline_text(input_text):
    """
    Parses an inline text with voice/emotion markers. The idea is that whenever a marker is
    found (e.g., [voice="narrator" emotion="neutral"]), the subsequent text is associated with that metadata.
    Any marker that starts with 'end' (such as [endchunk]) is ignored.
    If no marker is present at the beginning, default values are used.
    """
    default_metadata = {"voice": "narrator", "emotion": "neutral"}
    segments = []
    current_metadata = default_metadata.copy()
    pos = 0
    # This regex finds any tag with [ ... ]
    marker_pattern = re.compile(r'\[(.*?)\]')

    for m in re.finditer(marker_pattern, input_text):
        start, end = m.span()
        # Any text before the marker is associated with the current metadata.
        if start > pos:
            text_segment = input_text[pos:start].strip()
            if text_segment:
                segments.append((text_segment, current_metadata["voice"], current_metadata["emotion"]))
        marker_content = m.group(1).strip()
        # If the marker is an end marker (i.e. starts with "end"), we ignore it.
        if marker_content.lower().startswith("end"):
            pos = end
            continue
        # Extract key="value" pairs from the marker.
        attrs = dict(re.findall(r'(\w+)\s*=\s*"([^"]+)"', marker_content))
        if "voice" in attrs:
            current_metadata["voice"] = attrs["voice"]
        if "emotion" in attrs:
            current_metadata["emotion"] = attrs["emotion"]
        pos = end

    # Any remaining text after the last marker.
    if pos < len(input_text):
        remaining = input_text[pos:].strip()
        if remaining:
            segments.append((remaining, current_metadata["voice"], current_metadata["emotion"]))
    return segments

def split_into_sentences(segment_text):
    """
    Splits the text into sentences using a simple regular expression.

    Note: This regex looks for punctuation marks (., !, ?) followed by whitespace.
    Depending on your text, you might want to use a more robust sentence tokenizer.
    """
    sentences = re.split(r'(?<=[.!?])\s+', segment_text)
    # Filter out any empty sentences
    return [s.strip() for s in sentences if s.strip()]

def create_chunks(segments, max_chars):
    """
    Given a list of segments (each a tuple of (text, voice, emotion)) produces a list of JSON-friendly
    chunks. The sentences are first extracted from each segment. Then they are aggregated in order,
    taking care to start a new chunk if (a) adding a sentence would cause the chunk to exceed max_chars or
    (b) the sentence's voice (or emotion) differs from the current chunk.
    """
    # Create a list of sentences, each with its associated metadata.
    sentences_with_meta = []
    for seg_text, seg_voice, seg_emotion in segments:
        for sentence in split_into_sentences(seg_text):
            sentences_with_meta.append({
                "sentence": sentence,
                "voice": seg_voice,
                "emotion": seg_emotion
            })

    chunks = []
    current_chunk = ""
    current_voice = None
    current_emotion = None
    chunk_index = 1

    for item in sentences_with_meta:
        sentence = item["sentence"]
        voice = item["voice"]
        emotion = item["emotion"]

        # If the voice or emotion is changing compared to the current chunk, force a new chunk.
        if current_voice is not None and (voice != current_voice or emotion != current_emotion):
            chunks.append({
                "chunk_id": f"{chunk_index:03d}",
                "voice": current_voice,
                "emotion": current_emotion,
                "text": current_chunk.strip()
            })
            chunk_index += 1
            current_chunk = ""
            current_voice = voice
            current_emotion = emotion

        # If starting a new chunk, set the metadata.
        if not current_chunk:
            current_voice = voice
            current_emotion = emotion

        # Determine whether adding the sentence would exceed the allowed maximum.
        # If the current chunk has content, we add a space.
        tentative = (current_chunk + " " + sentence).strip() if current_chunk else sentence
        if len(tentative) > max_chars and current_chunk:
            # If adding the sentence goes over max_chars, then finish current chunk and start a new chunk.
            chunks.append({
                "chunk_id": f"{chunk_index:03d}",
                "voice": current_voice,
                "emotion": current_emotion,
                "text": current_chunk.strip()
            })
            chunk_index += 1
            current_chunk = sentence  # start new chunk with the current sentence
            current_voice = voice
            current_emotion = emotion
        else:
            current_chunk = tentative

    # Append any remaining text as the final chunk.
    if current_chunk:
        chunks.append({
            "chunk_id": f"{chunk_index:03d}",
            "voice": current_voice,
            "emotion": current_emotion,
            "text": current_chunk.strip()
        })

    return chunks

def process_inline_text_to_json(input_text, max_chars=100):
    """
    Given an inline formatted text and a maximum character limit per chunk,
    returns the JSON representation of chunks.
    """
    segments = parse_inline_text(input_text)
    chunks = create_chunks(segments, max_chars)
    result = {"chunks": chunks}
    return json.dumps(result, indent=4)

if __name__ == "__main__":
    # Example input inline text.
    sample_text = '''
[voice="narrator" emotion="neutral"]
She said the following: [voice="character-01" emotion="angry"] "No, please leave me alone!" [voice="narrator" emotion="neutral"] and went back to her seat.

Then he shouted, [voice="character-01" emotion="angry"] "Get out now!"  [voice="narrator" emotion="neutral"] However, things were not so simple.
The day slowly came to an end.

[voice="character-01" emotion="angry"] Get out now! A very long text for demonstration purposes. Another sentence here. A very long text for demonstration purposes. Another sentence here."
    '''

    # For this example, we set a maximum of 100 characters per chunk.
    max_chars = 100
    output_json = process_inline_text_to_json(sample_text, max_chars)
    print(output_json)