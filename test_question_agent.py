#!/usr/bin/env python3
"""
Test Q-Agent: Generate a question and measure performance
"""

import json
import time
import subprocess
import sys

def test_question_agent():
    print("=" * 80)
    print("🧪 TESTING Q-AGENT (Question Generator)")
    print("=" * 80)
    print()

    # Start timer
    start_time = time.time()

    print("📤 Sending request to Q-Agent...")

    # Run Q-Agent with empty request (will generate default question)
    try:
        process = subprocess.Popen(
            ["python3", "/workspace/question_agent.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Send empty request
        stdout, stderr = process.communicate(input="", timeout=20)

        elapsed = time.time() - start_time

        print(f"⏱️  Generation time: {elapsed:.2f} seconds")
        print()

        # Check time limit
        if elapsed > 10.0:
            print(f"❌ FAILED: Generation took {elapsed:.2f}s (limit: 10s)")
            print()
        else:
            print(f"✅ PASSED: Generation within time limit ({elapsed:.2f}s < 10s)")
            print()

        # Parse output
        try:
            # Filter stderr to find actual JSON output in stdout
            question_data = json.loads(stdout.strip())

            print("📝 Generated Question:")
            print("-" * 80)
            print(f"Question: {question_data['question']}")
            print()
            print("Choices:")
            for letter in ['A', 'B', 'C', 'D']:
                print(f"  {letter}. {question_data['choices'][letter]}")
            print()
            print(f"Correct Answer: {question_data['correct_answer']}")
            print("-" * 80)
            print()

            # Validation
            valid = True
            errors = []

            if not question_data.get('question'):
                valid = False
                errors.append("Missing or empty question")

            if not all(letter in question_data.get('choices', {}) for letter in ['A', 'B', 'C', 'D']):
                valid = False
                errors.append("Missing one or more choices")

            if question_data.get('correct_answer') not in ['A', 'B', 'C', 'D']:
                valid = False
                errors.append("Invalid correct answer")

            if valid:
                print("✅ Question format validation: PASSED")
            else:
                print("❌ Question format validation: FAILED")
                for error in errors:
                    print(f"   - {error}")

            print()

        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON output: {e}")
            print("Raw stdout:", stdout[:500])
            print()

        # Show stderr if any (logs)
        if stderr:
            print("📋 Agent logs (stderr):")
            print(stderr[:1000])

        print()
        print("=" * 80)
        print(f"✅ Q-AGENT TEST COMPLETE (Time: {elapsed:.2f}s)")
        print("=" * 80)

        return elapsed <= 10.0

    except subprocess.TimeoutExpired:
        print(f"❌ FAILED: Q-Agent timed out (>20s)")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_question_agent()
    sys.exit(0 if success else 1)
