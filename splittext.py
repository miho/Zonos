import nltk

#  ~/.local/bin/uv pip install nltk
nltk.download('punkt')
nltk.download('punkt_tab')

def split_text_into_sentences(text):
    """Splits text into sentences while respecting sentence boundaries."""
    return nltk.sent_tokenize(text)

def split_long_sentence(sentence, N):
    """Splits a long sentence into smaller chunks with a maximum length of N characters."""
    words = sentence.split()
    chunks = []
    current_chunk = ''
    for word in words:
        if len(current_chunk) + len(word) + 1 <= N:  # +1 for the space
            current_chunk += word + ' '
        else:
            chunks.append(current_chunk.rstrip())
            current_chunk = word + ' '
    chunks.append(current_chunk.rstrip())  # Add the last chunk
    return chunks

text = "Tropical Storm Gabrielle was a short-lived tropical cyclone that passed over North Carolina before tracking out to sea. The seventh named storm of the 2007 Atlantic hurricane season, Gabrielle developed as a subtropical cyclone on September 8 about 385 miles (620 km) southeast of Cape Lookout, North Carolina. Unfavorable wind shear affected the storm for much of its duration, although a temporary decrease in the shear allowed the cyclone to become a tropical storm. On September 9, Gabrielle made landfall at Cape Lookout National Seashore in the Outer Banks of North Carolina with winds of 60 mph (97 km/h). Turning to the northeast, the storm quickly weakened and dissipated on September 11. In advance of the storm, tropical cyclone watches and warnings were issued for coastal areas, while rescue teams and the U.S. Coast Guard were put on standby. The storm dropped heavy rainfall near its immediate landfall location but little precipitation elsewhere. Along the coast of North Carolina, high waves, rip currents, and storm surge were reported. Slight localized flooding was reported. Gusty winds also occurred, though no wind damage was reported. One person drowned in rough surf caused by the storm in Florida. Overall damage was minor."
sentences = split_text_into_sentences(text)

N = 300
chunks = []
chunk = ''
for sentence in sentences:
    if len(chunk) + len(sentence) <= N:
        chunk += " " + sentence
    else:
        if len(sentence) > N:  # Split long sentence
            long_chunks = split_long_sentence(sentence, N)
            chunks.extend(long_chunks)
        else:
            chunks.append(chunk)
            chunk = sentence
chunks.append(chunk)  # Add the last chunk

for sentence in sentences:
    print(sentence)

print('---')
for chunk in chunks:
    print("--> ", chunk)