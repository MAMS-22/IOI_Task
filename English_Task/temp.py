import torch
from transformer_lens import TransformerBridge

device = "cuda" if torch.cuda.is_available() else "cpu"

model = TransformerBridge.boot_transformers("bigscience/bloom-560m")
model.enable_compatibility_mode()

prompt = "When Mary and John went to the store, John gave a drink to"

print("Running forward pass...")

with torch.inference_mode():
    output = model(prompt)

print("Output type:", type(output))

if isinstance(output, torch.Tensor):
    print("Output shape:", output.shape)
    print("All values finite:", torch.isfinite(output).all().item())
    print("Last-position logits shape:", output[0, -1, :].shape)
print("Output:", output)