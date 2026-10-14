import torch
from transformer_lens import TransformerBridge
from transformer_lens.evals import IOIDataset
from torch.utils.data import DataLoader

MODEL_NAME = "bigscience/bloom-560m"

model = TransformerBridge.boot_transformers(
    MODEL_NAME,
    device="cuda" if torch.cuda.is_available() else "cpu",
)
model.enable_compatibility_mode()
model.eval()

# Keep only names that are single tokens when preceded by a space.
candidate_names = [
    "Mary", "John", "Susan", "David", "Sarah", "Michael",
    "Jessica", "Robert", "Linda", "William", "Karen", "James",
    "Elizabeth", "Daniel", "Barbara", "Matthew", "Nancy",
    "Anthony", "Margaret", "Mark",
]

names = [
    name for name in candidate_names
    if len(model.tokenizer.encode(
        " " + name, add_special_tokens=False
    )) == 1
]

print("Single-token names:", names)

dataset = IOIDataset(
    tokenizer=model.tokenizer,
    names=names,
    num_samples=100,
    symmetric=True,
    seed=42,
)

print("Dataset size:", len(dataset))

# Inspect a few examples before interpreting the scores.
for i in range(min(4, len(dataset))):
    sample = dataset[i]
    print("\nPrompt:", model.tokenizer.decode(sample["prompt"]))
    print("IO:", model.tokenizer.decode(sample["IO"]))
    print("S:", model.tokenizer.decode(sample["S"]))

from torch.nn.utils.rnn import pad_sequence

@torch.inference_mode()
def evaluate_ioi(model, dataset, batch_size=8):
    device = next(model.parameters()).device
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda samples: {
            "prompt": pad_sequence(
                [s["prompt"] for s in samples],
                batch_first=True,
            ),
            "IO": [s["IO"] for s in samples],
            "S": [s["S"] for s in samples],
            "prompt_length": [s["prompt"].shape[0] for s in samples],
        },
    )

    total_correct = 0
    total_logit_diff = 0.0
    total = 0

    for batch in loader:
        prompts = batch["prompt"].to(device)
        logits = model(prompts, return_type="logits")

        for i in range(logits.shape[0]):
            io = batch["IO"][i].to(device)
            s = batch["S"][i].to(device)
            prefix_length = batch["prompt_length"][i] - io.shape[0]

            min_len = min(io.shape[0], s.shape[0])
            io, s = io[:min_len], s[:min_len]

            differing = torch.where(io != s)[0]
            if len(differing) == 0:
                raise ValueError("IO and S targets have no differing token.")

            start_idx = differing[0].item()
            logit_idx = prefix_length + start_idx - 1

            next_logits = logits[i, logit_idx]
            correct_logit = next_logits[io[start_idx]]
            incorrect_logit = next_logits[s[start_idx]]

            total_logit_diff += (correct_logit - incorrect_logit).item()
            total_correct += int(correct_logit > incorrect_logit)
            total += 1

    return {
        "Logit Difference": total_logit_diff / total,
        "Accuracy": total_correct / total,
        "Samples": total,
    }

results = evaluate_ioi(model, dataset, batch_size=8)

print("\nTransformerLens-compatible IOI results")
for key, value in results.items():
    print(f"{key}: {value}")