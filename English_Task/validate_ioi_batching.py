import torch
from transformer_lens import TransformerBridge
from transformer_lens.evals import IOIDataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

MODEL = "bigscience/bloom-560m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = TransformerBridge.boot_transformers(MODEL, device=DEVICE)
model.enable_compatibility_mode()
model.eval()

names = [
    "Mary", "John", "Susan", "David", "Sarah", "Michael",
    "Jessica", "Robert", "Linda", "William", "Karen", "James",
    "Elizabeth", "Daniel", "Barbara", "Matthew", "Nancy",
    "Anthony", "Margaret", "Mark",
]
names = [
    n for n in names
    if len(model.tokenizer.encode(" " + n, add_special_tokens=False)) == 1
]

ds = IOIDataset(
    tokenizer=model.tokenizer,
    names=names,
    num_samples=100,
    symmetric=True,
    seed=42,
)

def score(sample):
    prompt = sample["prompt"].unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        logits = model(prompt, return_type="logits")[0]

    io, s = sample["IO"].to(DEVICE), sample["S"].to(DEVICE)
    n = min(len(io), len(s))
    diff = torch.where(io[:n] != s[:n])[0]
    if len(diff) == 0:
        raise ValueError("IO and S have no differing token")

    j = diff[0].item()
    next_logits = logits[len(sample["prompt"]) + j - 1]
    io_logit = next_logits[io[j]].item()
    s_logit = next_logits[s[j]].item()

    return {
        "logit_diff": io_logit - s_logit,
        "correct": io_logit > s_logit,
        "zero_rank": io_logit >= next_logits.max().item() - 1e-7,
    }

# Reference: each prompt evaluated without padding.
individual = [score(ds[i]) for i in range(len(ds))]

# Batched: right-padded prompts, matching the current evaluation approach.
loader = DataLoader(
    ds,
    batch_size=8,
    shuffle=False,
    collate_fn=lambda batch: {
        "prompts": pad_sequence(
            [x["prompt"] for x in batch], batch_first=True
        ),
        "samples": batch,
    },
)

batched = []
with torch.inference_mode():
    for batch in loader:
        logits = model(
            batch["prompts"].to(DEVICE), return_type="logits"
        )
        for i, sample in enumerate(batch["samples"]):
            io, s = sample["IO"].to(DEVICE), sample["S"].to(DEVICE)
            n = min(len(io), len(s))
            diff = torch.where(io[:n] != s[:n])[0]
            if len(diff) == 0:
                raise ValueError("IO and S have no differing token")

            j = diff[0].item()
            pos = len(sample["prompt"]) + j - 1
            next_logits = logits[i, pos]
            io_logit = next_logits[io[j]].item()
            s_logit = next_logits[s[j]].item()

            batched.append({
                "logit_diff": io_logit - s_logit,
                "correct": io_logit > s_logit,
                "zero_rank": io_logit >= next_logits.max().item() - 1e-7,
            })

print("Samples:", len(ds))
print("Individual accuracy:",
      100 * sum(x["correct"] for x in individual) / len(individual))
print("Batched accuracy:",
      100 * sum(x["correct"] for x in batched) / len(batched))
print("Individual zero-rank:",
      100 * sum(x["zero_rank"] for x in individual) / len(individual))
print("Batched zero-rank:",
      100 * sum(x["zero_rank"] for x in batched) / len(batched))
print("Maximum absolute logit-difference change:",
      max(abs(a["logit_diff"] - b["logit_diff"])
          for a, b in zip(individual, batched)))
print("Prediction mismatches:",
      sum(a["correct"] != b["correct"] or
          a["zero_rank"] != b["zero_rank"]
          for a, b in zip(individual, batched)))