
import inspect
import json

import torch
from transformers import AutoTokenizer
from transformer_lens import TransformerBridge


MODEL_NAME = "bigscience/bloom-560m"


def safe_attr(obj, name):
    try:
        return getattr(obj, name)
    except Exception as exc:
        return f"<unavailable: {type(exc).__name__}: {exc}>"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 72)
    print("BLOOM-560M INTERPRETABILITY DIAGNOSTIC")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = TransformerBridge.boot_transformers(
        MODEL_NAME,
        device=device,
    )
    model.enable_compatibility_mode()
    model.eval()

    print(f"\nModel wrapper: {type(model)}")
    print(f"Model config: {safe_attr(model, 'cfg')}")

    cfg = safe_attr(model, "cfg")
    if not isinstance(cfg, str):
        for attr in ("n_layers", "n_heads", "d_model", "d_head"):
            print(f"cfg.{attr}: {safe_attr(cfg, attr)}")

    # Check relevant APIs without assuming they exist.
    print("\n--- Relevant methods and attributes ---")
    candidates = [
        "run_with_cache",
        "run_with_hooks",
        "add_hook",
        "reset_hooks",
        "hook_dict",
        "hook_points",
        "blocks",
        "W_O",
        "W_V",
        "W_Q",
        "W_K",
        "parameters",
    ]

    for name in candidates:
        value = safe_attr(model, name)
        if callable(value):
            try:
                signature = inspect.signature(value)
            except (TypeError, ValueError):
                signature = "(signature unavailable)"
            print(f"{name}: callable {signature}")
        elif isinstance(value, dict):
            print(f"{name}: dict with {len(value)} entries")
        elif isinstance(value, (list, tuple)):
            print(f"{name}: {type(value).__name__} with {len(value)} entries")
        else:
            print(f"{name}: {type(value).__name__}")

    # Find exposed TransformerLens-style hook points.
    hook_names = []
    hook_dict = safe_attr(model, "hook_dict")

    if isinstance(hook_dict, dict):
        hook_names = list(hook_dict.keys())
    else:
        hook_points = safe_attr(model, "hook_points")
        if isinstance(hook_points, (list, tuple)):
            hook_names = [
                getattr(hook, "name", str(hook))
                for hook in hook_points
            ]

    print("\n--- Hook points ---")
    print(f"Total discovered: {len(hook_names)}")

    keywords = (
        "attn", "hook_z", "hook_result", "pattern",
        "resid", "mlp", "q", "k", "v",
    )

    relevant = [
        name for name in hook_names
        if any(word in name.lower() for word in keywords)
    ]

    for name in relevant[:150]:
        print(name)

    if len(relevant) > 150:
        print(f"... and {len(relevant) - 150} more relevant hook names")

    # Run a short forward pass.
    text = (
        "John and Mary went to the store. "
        "John gave a drink to"
    )
    tokens = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
    )["input_ids"].to(device)

    print("\n--- Forward-pass test ---")
    print(f"Test text: {text}")
    print(f"Input shape: {tuple(tokens.shape)}")

    with torch.inference_mode():
        output = model(tokens)

    logits = output.logits if hasattr(output, "logits") else output

    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Logits dtype: {logits.dtype}")
    print(f"All logits finite: {bool(torch.isfinite(logits).all())}")

    # Test the cache API if it is available.
    print("\n--- Activation-cache test ---")
    run_with_cache = safe_attr(model, "run_with_cache")

    if callable(run_with_cache):
        try:
            with torch.inference_mode():
                cached_output, cache = run_with_cache(tokens)

            cached_logits = (
                cached_output.logits
                if hasattr(cached_output, "logits")
                else cached_output
            )

            print("run_with_cache succeeded.")
            print(f"Cache type: {type(cache)}")
            print(f"Cached output shape: {tuple(cached_logits.shape)}")

            try:
                cache_keys = list(cache.keys())
                print(f"Cached activation count: {len(cache_keys)}")
                for key in cache_keys[:40]:
                    print(f"  {key}")
            except Exception as exc:
                print(f"Could not list cache keys: {exc}")

        except Exception as exc:
            print(f"run_with_cache failed: {type(exc).__name__}: {exc}")
    else:
        print("run_with_cache is not exposed by this model wrapper.")

    # Inspect the underlying Hugging Face modules for fallback access.
    print("\n--- Underlying module inspection ---")
    try:
        for name, module in model.named_modules():
            lower = name.lower()
            if any(
                key in lower
                for key in ("self_attention", "attention", "query_key_value")
            ):
                print(f"{name}: {type(module).__name__}")
    except Exception as exc:
        print(f"named_modules unavailable: {exc}")

    print("\n--- Summary ---")
    summary = {
        "model": MODEL_NAME,
        "device": device,
        "wrapper_type": str(type(model)),
        "hook_count": len(hook_names),
        "relevant_hook_count": len(relevant),
        "logits_shape": list(logits.shape),
        "logits_finite": bool(torch.isfinite(logits).all()),
        "has_run_with_cache": callable(run_with_cache),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()