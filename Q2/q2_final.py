import re
from collections import Counter
import nltk
from nltk.corpus import gutenberg, stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag

RESOURCES = [
    "gutenberg",
    "punkt",
    "punkt_tab",
    "stopwords",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
]

for resource in RESOURCES:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

raw_text = gutenberg.raw("austen-emma.txt")

#PART A : Text Preprocessing
raw_tokens = word_tokenize(raw_text)
before_total_tokens = len(raw_tokens)
before_unique_words = len(set(raw_tokens))
processed_text = re.sub(r"--", " -- ", raw_text)
processed_tokens = word_tokenize(processed_text)
stop_words = set(stopwords.words("english"))
preprocessed_tokens = []
for tok in processed_tokens:
    lower = tok.lower()
    if lower.isalpha() and lower not in stop_words:
        preprocessed_tokens.append(lower)
after_total_tokens = len(preprocessed_tokens)
freq_dist = Counter(preprocessed_tokens)
after_unique_words = len(freq_dist)
print("=" * 60)
print("PART A: Preprocessing statistics")
print("=" * 60)
print("BEFORE preprocessing:")
print(f"  Total tokens:   {before_total_tokens}")
print(f"  Unique words:   {before_unique_words}")
print("AFTER preprocessing:")
print(f"  Total tokens:   {after_total_tokens}")
print(f"  Unique words:   {after_unique_words}")
print()

#PART B: Word Frequency Analysis
top_10_words = freq_dist.most_common(10)
num_hapax = sum(1 for c in freq_dist.values() if c == 1)
print("=" * 60)
print("PART B: Word Frequency Analysis")
print("=" * 60)
print(f"{'Rank':<6}{'Word':<15}{'Frequency':<10}")
for i, (word, count) in enumerate(top_10_words, start=1):
    print(f"{i:<6}{word:<15}{count:<10}")
print(f"\nWords occurring exactly once : {num_hapax}")
print()

#PART C: POS Tagging
tagged = []
for sent in sent_tokenize(processed_text):
    tokens = word_tokenize(sent)
    tagged.extend(pos_tag(tokens))
NOUN_TAGS = {"NN", "NNS", "NNP", "NNPS"}
NOUN_TOKEN_RE = re.compile(r"^[A-Za-z]+\.?$")
raw_nouns = [
    word.lower()
    for word, tag in tagged
    if tag in NOUN_TAGS and NOUN_TOKEN_RE.match(word)
]
bare_forms = {w for w in raw_nouns if not w.endswith(".")}
def normalize(word):
    if word.endswith("."):
        core = word[:-1]
        return core if core in bare_forms else word
    return word

nouns_normalized = [normalize(w) for w in raw_nouns]
noun_freq = Counter(nouns_normalized)
top_10_nouns = noun_freq.most_common(10)
unique_nouns = sorted(noun_freq.keys())
print("=" * 60)
print("PART C: POS Tagging - Common Noun Analysis")
print("=" * 60)
print(f"{'Rank':<6}{'Noun':<15}{'Frequency':<10}")
for i, (word, count) in enumerate(top_10_nouns, start=1):
    print(f"{i:<6}{word:<15}{count:<10}")
print(f"\nTotal noun occurrences: {len(nouns_normalized)}")
print(f"Total unique nouns:     {len(unique_nouns)}")
print()

#save Output Files
with open("q2_top10_words.txt", "w", encoding="utf-8") as f:
    for word, count in top_10_words:
        f.write(f"{word}\t{count}\n")

with open("q2_top10_nouns.txt", "w", encoding="utf-8") as f:
    for word, count in top_10_nouns:
        f.write(f"{word}\t{count}\n")

with open("q2_unique_nouns.txt", "w", encoding="utf-8") as f:
    for noun in unique_nouns:
        f.write(noun + "\n")