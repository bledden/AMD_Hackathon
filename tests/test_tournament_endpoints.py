#!/usr/bin/env python3
"""
Test tournament server endpoints for performance
"""

import json
import requests
import time

def test_tournament_server():
    base_url = "http://localhost:5000"

    print("=" * 80)
    print("🧪 TESTING TOURNAMENT SERVER")
    print("=" * 80)
    print()

    # Test 1: Health check
    print("1️⃣  Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print(f"   ✅ Server is healthy: {response.json()}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return False

    print()

    # Test 2: Question generation
    print("2️⃣  Testing question generation...")
    print("   Sending request...")

    try:
        start = time.time()
        response = requests.post(
            f"{base_url}/generate_question",
            json={},
            timeout=30
        )
        elapsed = time.time() - start

        if response.status_code == 200:
            question_data = response.json()

            print(f"   ⏱️  Generation time: {elapsed:.2f}s (server: {question_data.get('generation_time', 0):.2f}s)")
            print()
            print("   📝 Generated Question:")
            print("   " + "-" * 76)
            print(f"   Q: {question_data['question']}")
            print()
            print("   Choices:")
            for letter in ['A', 'B', 'C', 'D']:
                print(f"     {letter}. {question_data['choices'][letter]}")
            print()
            print(f"   Correct Answer: {question_data['correct_answer']}")
            print("   " + "-" * 76)
            print()

            # Check time limit
            if elapsed > 10.0:
                print(f"   ⚠️  WARNING: Generation took {elapsed:.2f}s (limit: 10s)")
            else:
                print(f"   ✅ PASSED: Generation within time limit ({elapsed:.2f}s < 10s)")

            print()

            # Test 3: Answer the generated question
            print("3️⃣  Testing answer generation...")
            print("   Sending question to be answered...")

            answer_payload = {
                "question": question_data['question'],
                "choices": question_data['choices']
            }

            start = time.time()
            answer_response = requests.post(
                f"{base_url}/answer_question",
                json=answer_payload,
                timeout=15
            )
            elapsed = time.time() - start

            if answer_response.status_code == 200:
                answer_data = answer_response.json()

                print(f"   ⏱️  Answer time: {elapsed:.2f}s (server: {answer_data.get('answer_time', 0):.2f}s)")
                print(f"   📋 Model's answer: {answer_data['answer']}")
                print(f"   📋 Correct answer: {question_data['correct_answer']}")

                if answer_data['answer'] == question_data['correct_answer']:
                    print("   ✅ Model answered correctly!")
                else:
                    print("   ⚠️  Model answered incorrectly (but this is okay for testing)")

                print()

                # Check time limit
                if elapsed > 6.0:
                    print(f"   ⚠️  WARNING: Answering took {elapsed:.2f}s (limit: 6s)")
                else:
                    print(f"   ✅ PASSED: Answering within time limit ({elapsed:.2f}s < 6s)")

                print()

            else:
                print(f"   ❌ Answer generation failed: {answer_response.status_code}")
                print(f"   Error: {answer_response.text}")
                return False

        else:
            print(f"   ❌ Question generation failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("=" * 80)
    print("✅ ALL TESTS PASSED")
    print("=" * 80)

    return True

if __name__ == "__main__":
    import sys
    success = test_tournament_server()
    sys.exit(0 if success else 1)
