"""
Quick Test - Hanya test aggregation function tanpa full training
Lebih cepat untuk verifikasi perbaikan
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Import functions from phase7
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase7_sota_comparison import (
    bfedplc_aggregation, multi_krum, fedavg, detect_byzantine_updates
)

def create_dummy_updates(num_clients=10, model_size=1000, n_byzantine=2, attack_type='sign_flip'):
    """Create dummy model updates for testing"""
    updates = []
    
    # Create benign updates (small random values)
    for i in range(num_clients - n_byzantine):
        update = {
            'weight': torch.randn(model_size) * 0.1,
            'bias': torch.randn(10) * 0.1
        }
        updates.append(update)
    
    # Create Byzantine updates
    for i in range(n_byzantine):
        if attack_type == 'sign_flip':
            # Sign flip attack: multiply by -3.0
            update = {
                'weight': torch.randn(model_size) * 0.1 * -3.0,
                'bias': torch.randn(10) * 0.1 * -3.0
            }
        elif attack_type == 'random':
            # Random noise attack
            update = {
                'weight': torch.randn(model_size) * 10.0,
                'bias': torch.randn(10) * 10.0
            }
        else:
            update = {
                'weight': torch.randn(model_size) * 0.1,
                'bias': torch.randn(10) * 0.1
            }
        updates.append(update)
    
    return updates

def test_byzantine_detection():
    """Test Byzantine detection function"""
    print("=" * 70)
    print("TEST 1: Byzantine Detection Function")
    print("=" * 70)
    
    num_clients = 10
    n_byzantine = 2
    
    # Test sign_flip detection
    print("\n1. Testing sign_flip attack detection...")
    updates = create_dummy_updates(num_clients, model_size=100, n_byzantine=n_byzantine, attack_type='sign_flip')
    
    benign_indices, byzantine_indices = detect_byzantine_updates(
        updates, 
        threshold_factor=2.0,
        attack_type='sign_flip',
        n_byzantine=n_byzantine
    )
    
    print(f"   Total updates: {len(updates)}")
    print(f"   Benign detected: {len(benign_indices)} (indices: {benign_indices[:5]}...)")
    print(f"   Byzantine detected: {len(byzantine_indices)} (indices: {byzantine_indices})")
    
    # Check if Byzantine are detected (should detect at least some)
    expected_byzantine = list(range(num_clients - n_byzantine, num_clients))
    detected_correctly = len([i for i in byzantine_indices if i in expected_byzantine])
    
    print(f"   Expected Byzantine: {expected_byzantine}")
    print(f"   Correctly detected: {detected_correctly}/{n_byzantine}")
    
    if detected_correctly >= n_byzantine * 0.5:  # At least 50% detection
        print("   [OK] Detection working (>=50% correct)")
        detection_ok = True
    else:
        print("   [WARN] Detection needs improvement")
        detection_ok = False
    
    return detection_ok

def test_bfedplc_aggregation():
    """Test B-FedPLC aggregation function"""
    print("\n" + "=" * 70)
    print("TEST 2: B-FedPLC Aggregation Function")
    print("=" * 70)
    
    num_clients = 10
    n_byzantine = 2
    
    # Test with sign_flip attack
    print("\n1. Testing with sign_flip attack...")
    updates = create_dummy_updates(num_clients, model_size=100, n_byzantine=n_byzantine, attack_type='sign_flip')
    
    result = bfedplc_aggregation(
        updates,
        n_byzantine=n_byzantine,
        num_clusters=3,
        attack_type='sign_flip',
        verbose=True
    )
    
    if result is not None:
        print("   [OK] Aggregation successful")
        print(f"   Result keys: {list(result.keys())}")
        print(f"   Result shape (weight): {result['weight'].shape}")
        aggregation_ok = True
    else:
        print("   [FAIL] Aggregation failed (returned None)")
        aggregation_ok = False
    
    # Test with random attack
    print("\n2. Testing with random attack...")
    updates = create_dummy_updates(num_clients, model_size=100, n_byzantine=n_byzantine, attack_type='random')
    
    result = bfedplc_aggregation(
        updates,
        n_byzantine=n_byzantine,
        num_clusters=3,
        attack_type='random',
        verbose=True
    )
    
    if result is not None:
        print("   [OK] Aggregation successful")
        aggregation_ok = aggregation_ok and True
    else:
        print("   [FAIL] Aggregation failed")
        aggregation_ok = False
    
    return aggregation_ok

def test_fallback_logic():
    """Test fallback to Multi-Krum"""
    print("\n" + "=" * 70)
    print("TEST 3: Fallback Logic")
    print("=" * 70)
    
    num_clients = 10
    n_byzantine = 5  # High Byzantine fraction to trigger fallback
    
    print(f"\nTesting with high Byzantine fraction ({n_byzantine}/{num_clients})...")
    updates = create_dummy_updates(num_clients, model_size=100, n_byzantine=n_byzantine, attack_type='sign_flip')
    
    result = bfedplc_aggregation(
        updates,
        n_byzantine=n_byzantine,
        num_clusters=3,
        attack_type='sign_flip',
        verbose=True
    )
    
    if result is not None:
        print("   [OK] Fallback working (returned valid result)")
        fallback_ok = True
    else:
        print("   [FAIL] Fallback failed")
        fallback_ok = False
    
    return fallback_ok

def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("QUICK TEST: B-FedPLC Byzantine Detection Fix")
    print("=" * 70)
    print("\nThis test verifies the improved Byzantine detection and aggregation")
    print("without running full training (faster execution)")
    print()
    
    results = {}
    
    # Test 1: Detection
    results['detection'] = test_byzantine_detection()
    
    # Test 2: Aggregation
    results['aggregation'] = test_bfedplc_aggregation()
    
    # Test 3: Fallback
    results['fallback'] = test_fallback_logic()
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Byzantine Detection:  {'[PASS]' if results['detection'] else '[FAIL]'}")
    print(f"Aggregation Function: {'[PASS]' if results['aggregation'] else '[FAIL]'}")
    print(f"Fallback Logic:       {'[PASS]' if results['fallback'] else '[FAIL]'}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n[SUCCESS] ALL TESTS PASSED: Fix is working correctly!")
        print("\nNext step: Run full experiment with:")
        print("  python phase7_sota_comparison.py")
        return 0
    else:
        print("\n[WARNING] SOME TESTS FAILED: Review the output above")
        return 1

if __name__ == "__main__":
    sys.exit(main())
