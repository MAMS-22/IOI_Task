from transformers import AutoTokenizer
import json

MODEL = "bigscience/bloom-560m"
VOCAB = "telugu_vocab_verified.json"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

with open(VOCAB, encoding="utf-8") as f:
    vocab = json.load(f)

names = vocab["names"]

print(f"Testing {len(names)} names...\n")

single = []
multi = []

for name in names:
    ids = tokenizer.encode(name, add_special_tokens=False)

    if len(ids) == 1:
        single.append((name, ids[0]))
    else:
        multi.append((name, ids))

print("=" * 60)
print(f"SINGLE-TOKEN: {len(single)}/{len(names)}")
print("=" * 60)

for name, token_id in single:
    print(f"{name:12} -> {token_id}")

print("\n" + "=" * 60)
print(f"MULTI-TOKEN: {len(multi)}/{len(names)}")
print("=" * 60)

for name, ids in multi:
    print(f"{name:12} -> {ids}")

# Save the definitive verified vocabulary
verified_vocab = {
    "names": [name for name, _ in single],
    "places": [],
    "objects": []
}

with open("telugu_vocab.json", "w", encoding="utf-8") as f:
    json.dump(verified_vocab, f, ensure_ascii=False, indent=2)
    f.write("\n")

print("\nSaved definitive vocabulary to: telugu_vocab.json")