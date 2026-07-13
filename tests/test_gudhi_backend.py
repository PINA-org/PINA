"""
Smoke test for the MorphologicalBackend (Kornia fallback).
"""
import torch
from pina.topology.backends.morphological_backend import MorphologicalBackend

def test_morphological_backend():
    print("=" * 50)
    print("Testing MorphologicalBackend (Kornia Fallback)")
    print("=" * 50)

    data = torch.zeros(3, 32, 32)
    val = 1.0

    data[0, 5:10, 5:10] = val
    data[0, 20:25, 20:25] = val

    data[1, 5:25, 5:25] = val

    data[2, 4:28, 4:28] = val
    data[2, 10:22, 10:22] = 0.0

    backend = MorphologicalBackend(low_threshold=0.3, high_threshold=0.7)
    result = backend.compute(data)

    print(f"\nBackend: {result.backend_name}")
    print(f"Success: {result.success}")
    if not result.success:
        print(f"Error: {result.error_msg}")
        return

    print("\n--- Per-Sample Results ---")
    for i in range(3):
        print(f"Sample {i}: β₀ = {result.per_sample_beta_0[i]}, β₁ = {result.per_sample_beta_1[i]}")

    print("\n--- Metadata ---")
    print(f"Batch Size: {result.metadata.get('batch_size')}")
    print(f"low_threshold: {result.metadata.get('low_threshold')}")
    print(f"high_threshold: {result.metadata.get('high_threshold')}")
    print(f"β₁ Note: {result.metadata.get('note')}")

    expected_beta_0 = [2, 1, 1]
    expected_beta_1 = [None, None, None]

    print("\n--- Checking Results ---")
    assert result.per_sample_beta_0 == expected_beta_0, \
        f"Beta_0 mismatch: got {result.per_sample_beta_0}, expected {expected_beta_0}"
    assert result.per_sample_beta_1 == expected_beta_1, \
        f"Beta_1 mismatch: got {result.per_sample_beta_1}, expected {expected_beta_1}"

    print("\n✅ MorphologicalBackend test passed!")

if __name__ == "__main__":
    test_morphological_backend()