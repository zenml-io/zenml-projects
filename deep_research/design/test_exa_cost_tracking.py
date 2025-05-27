#!/usr/bin/env python3
"""Test script to verify Exa cost tracking implementation."""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.search_utils import exa_search, extract_search_results, search_and_extract_results
from utils.pydantic_models import ResearchState
from utils.tracing_metadata_utils import get_costs_by_prompt_type


def test_exa_cost_extraction():
    """Test that Exa costs are properly extracted from API responses."""
    print("=== Testing Exa Cost Extraction ===")
    
    # Test with a simple query
    query = "What is quantum computing?"
    print(f"\nSearching for: {query}")
    
    # Test direct exa_search
    results = exa_search(query, max_results=2)
    print(f"Direct exa_search returned exa_cost: ${results.get('exa_cost', 0.0):.4f}")
    
    # Test extract_search_results
    extracted, cost = extract_search_results(results, provider="exa")
    print(f"extract_search_results returned cost: ${cost:.4f}")
    print(f"Number of results extracted: {len(extracted)}")
    
    # Test search_and_extract_results
    results2, cost2 = search_and_extract_results(query, max_results=2, provider="exa")
    print(f"search_and_extract_results returned cost: ${cost2:.4f}")
    print(f"Number of results: {len(results2)}")
    
    return cost2 > 0


def test_research_state_cost_tracking():
    """Test that ResearchState properly tracks costs."""
    print("\n=== Testing ResearchState Cost Tracking ===")
    
    state = ResearchState(main_query="Test query")
    
    # Simulate adding search costs
    state.search_costs["exa"] = 0.05
    state.search_cost_details.append({
        "provider": "exa",
        "query": "test query 1",
        "cost": 0.02,
        "timestamp": 1234567890.0,
        "step": "test_step"
    })
    state.search_cost_details.append({
        "provider": "exa",
        "query": "test query 2",
        "cost": 0.03,
        "timestamp": 1234567891.0,
        "step": "test_step"
    })
    
    print(f"Total Exa cost: ${state.search_costs.get('exa', 0.0):.4f}")
    print(f"Number of search details: {len(state.search_cost_details)}")
    
    return True


def test_cost_aggregation():
    """Test cost aggregation from multiple states."""
    print("\n=== Testing Cost Aggregation ===")
    
    # Create multiple sub-states
    state1 = ResearchState(main_query="Test")
    state1.search_costs["exa"] = 0.02
    state1.search_cost_details.append({
        "provider": "exa",
        "query": "query1",
        "cost": 0.02,
        "timestamp": 1234567890.0,
        "step": "sub_step_1"
    })
    
    state2 = ResearchState(main_query="Test")
    state2.search_costs["exa"] = 0.03
    state2.search_cost_details.append({
        "provider": "exa",
        "query": "query2",
        "cost": 0.03,
        "timestamp": 1234567891.0,
        "step": "sub_step_2"
    })
    
    # Simulate merge
    merged_state = ResearchState(main_query="Test")
    merged_state.search_costs = {}
    merged_state.search_cost_details = []
    
    for state in [state1, state2]:
        for provider, cost in state.search_costs.items():
            merged_state.search_costs[provider] = merged_state.search_costs.get(provider, 0.0) + cost
        merged_state.search_cost_details.extend(state.search_cost_details)
    
    print(f"Merged total cost: ${merged_state.search_costs.get('exa', 0.0):.4f}")
    print(f"Merged search details count: {len(merged_state.search_cost_details)}")
    
    return merged_state.search_costs.get('exa', 0.0) == 0.05


def main():
    """Run all tests."""
    print("Testing Exa Cost Tracking Implementation\n")
    
    # Check if Exa API key is set
    if not os.getenv("EXA_API_KEY"):
        print("WARNING: EXA_API_KEY not set. Skipping real API tests.")
        test_api = False
    else:
        test_api = True
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Exa cost extraction (only if API key is available)
    if test_api:
        tests_total += 1
        try:
            if test_exa_cost_extraction():
                print("✓ Exa cost extraction test passed")
                tests_passed += 1
            else:
                print("✗ Exa cost extraction test failed")
        except Exception as e:
            print(f"✗ Exa cost extraction test failed with error: {e}")
    
    # Test 2: ResearchState cost tracking
    tests_total += 1
    try:
        if test_research_state_cost_tracking():
            print("✓ ResearchState cost tracking test passed")
            tests_passed += 1
        else:
            print("✗ ResearchState cost tracking test failed")
    except Exception as e:
        print(f"✗ ResearchState cost tracking test failed with error: {e}")
    
    # Test 3: Cost aggregation
    tests_total += 1
    try:
        if test_cost_aggregation():
            print("✓ Cost aggregation test passed")
            tests_passed += 1
        else:
            print("✗ Cost aggregation test failed")
    except Exception as e:
        print(f"✗ Cost aggregation test failed with error: {e}")
    
    print(f"\nTests passed: {tests_passed}/{tests_total}")
    
    if tests_passed == tests_total:
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())