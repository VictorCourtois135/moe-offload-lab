"""
RabbitLLM Test Script
======================
Goal: verify that RabbitLLM works on your setup (RTX 2060, 6.4GB VRAM)
with a model that has a real reason to be streamed layer by layer —
i.e. a model too large to fit entirely in your VRAM.

Requirements:
    pip install rabbitllm

Usage:
    python test_rabbitllm.py
        -> tests Qwen2.5-7B-Instruct (~15GB download, doesn't fit
           in FP16 within 6.4GB of VRAM without RabbitLLM)

    python test_rabbitllm.py --model "Qwen/Qwen2.5-0.5B-Instruct"
        -> tests a smaller model (just to check compatibility,
           doesn't prove RabbitLLM's value since it would fit without it)

    python test_rabbitllm.py --compression 4bit --max-new-tokens 20
        -> compression and generation length options
"""

import argparse
import time
import sys


def patch_rabbitllm_missing_import():
    """
    Known bug in RabbitLLM 1.1.0: rabbitllm/utils/splitting.py calls
    huggingface_hub.snapshot_download(...) without having imported the
    huggingface_hub module, which raises a NameError. We fix this here by
    injecting the module into the namespace of the affected file, without
    having to modify the files installed in site-packages.
    """
    try:
        import huggingface_hub
        from rabbitllm.utils import splitting as _splitting_mod
        if not hasattr(_splitting_mod, "huggingface_hub"):
            _splitting_mod.huggingface_hub = huggingface_hub
            print("🩹 Patch applied: missing huggingface_hub import fixed "
                  "in rabbitllm.utils.splitting")
    except Exception as e:
        print(f"⚠️  Could not apply the automatic patch ({e}). "
              f"If loading fails with a NameError, see the README "
              f"for the manual fix.")


def print_step(msg):
    print(f"\n{'='*60}\n{msg}\n{'='*60}")


def check_gpu():
    print_step("1. Checking the GPU")
    try:
        import torch
        if not torch.cuda.is_available():
            print("❌ CUDA not available. RabbitLLM requires Linux/Windows + a CUDA GPU.")
            sys.exit(1)
        name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU detected: {name}")
        print(f"✅ Total VRAM: {total_vram:.1f} GB")
        return torch
    except ImportError:
        print("❌ PyTorch is not installed. First run: pip install torch")
        sys.exit(1)


def check_vram(torch, label=""):
    used = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    print(f"   [VRAM {label}] allocated: {used:.2f} GB | reserved: {reserved:.2f} GB")


def run_test(model_id, torch, compression, max_new_tokens):
    print_step(f"2. Loading the model: {model_id}")
    print(f"   Compression: {compression or 'none (full bfloat16)'}")
    print("   ⚠️  Note: the download always happens at full precision.")
    print("      Compression is only applied afterward, at load time.")

    try:
        from rabbitllm import AutoModel
    except ImportError:
        print("❌ RabbitLLM is not installed. Run: pip install rabbitllm")
        sys.exit(1)

    check_vram(torch, "before loading")

    t0 = time.time()
    try:
        model = AutoModel.from_pretrained(
            model_id,
            device="cuda:0",
            compression=compression,   # None, "4bit" or "8bit"
            profiling_mode=True,       # shows per-layer loading time
        )
    except Exception as e:
        print(f"\n❌ Model loading failed.")
        print(f"   Error: {e}")
        print("\n💡 Only Qwen2 and Qwen3 are tested and reliable with RabbitLLM today.")
        raise
    load_time = time.time() - t0
    print(f"✅ Model loaded in {load_time:.1f}s")
    check_vram(torch, "after loading")

    print_step("3. Generation test")
    prompt = "Explain what a Mixture of Experts model is in one sentence."
    print(f"Prompt: {prompt}")
    print(f"max_new_tokens: {max_new_tokens}")

    # RabbitLLM exposes a built-in tokenizer (model.tokenizer) — the text needs
    # to be tokenized before being sent to generate(), just like with a regular
    # transformers model.
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    input_text = model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    tokens = model.tokenizer(
        [input_text], return_tensors="pt", truncation=True, max_length=512
    )
    input_ids = tokens["input_ids"].to(model.device if hasattr(model, "device") else "cuda:0")
    attention_mask = tokens.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(input_ids.device)

    t0 = time.time()
    try:
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
        )
    except Exception as e:
        print(f"\n❌ Generation failed.")
        print(f"   Error: {e}")
        raise
    gen_time = time.time() - t0

    # Only decode the newly generated tokens (not the whole prompt echoed back)
    new_tokens = output_ids[0][input_ids.shape[-1]:]
    output_text = model.tokenizer.decode(new_tokens, skip_special_tokens=True)
    n_tokens = len(new_tokens)

    print(f"\n✅ Response generated in {gen_time:.1f}s ({n_tokens} tokens, "
          f"~{gen_time / max(n_tokens, 1):.1f}s/token):")
    print(f"   {output_text}")
    check_vram(torch, "after generation")

    print_step("✅ TEST PASSED")
    print(f"Model: {model_id}")
    print(f"Compression: {compression or 'none'}")
    print(f"Load time: {load_time:.1f}s")
    print(f"Generation time: {gen_time:.1f}s for {n_tokens} tokens")
    print("\nThis model is ready for the live demo.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace identifier of the model to test "
             "(default: Qwen2.5-7B-Instruct, ~15GB, doesn't fit in FP16 on 6.4GB of VRAM)"
    )
    parser.add_argument(
        "--compression", type=str, default=None, choices=[None, "4bit", "8bit"],
        help="Optional compression applied at load time (None, 4bit, 8bit)"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=40,
        help="Number of tokens to generate (reduce for a faster test/demo)"
    )
    args = parser.parse_args()

    torch = check_gpu()
    patch_rabbitllm_missing_import()
    run_test(args.model, torch, args.compression, args.max_new_tokens)