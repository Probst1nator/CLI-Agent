# `utils/_goldenstandardutil.py`
import markpickle  # Use the standard serializer
from typing import Dict, Any, Optional

# For standalone testing, we can create a mock if the real one isn't available.
# This allows the script to be run directly without depending on the full agent's path structure.
try:
    from agent.utils_manager.util_base import UtilBase
except ImportError:
    print("Warning: Could not import UtilBase. Using a mock class for standalone testing.")
    class UtilBase:
        pass

class GoldenStandardUtil(UtilBase):
    """
    A clear, concise description of what the utility does.
    This template serves as a 'golden standard' for creating new utilities.
    """

    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        """
        Provides standardized metadata for the tool, used for semantic search and hint generation.
        """
        return {
            "keywords": ["keyword1", "keyword2", "template", "example"],
            "use_cases": [
                "Demonstrate the standard structure for a new utility.",
                "Provide a template for developers to build upon."
            ],
            "arguments": {
                "arg1": "A mandatory string argument for the operation.",
                "arg2": "An optional integer argument that modifies the behavior."
            }
        }

    @staticmethod
    def _run_logic(arg1: str, arg2: Optional[int] = None) -> str:
        """
        The core implementation of the utility.
        It has explicit, type-hinted arguments and a clear return contract.
        It should handle all its own errors and always return a string serialized by markpickle.
        """
        try:
            # --- Main logic of the tool ---
            if not arg1:
                # Always return validation and business logic errors in the standard format.
                return markpickle.dumps({"error": "arg1 cannot be empty."})

            # Perform the work.
            processed_result = f"Processed {arg1} with value {arg2}"

            # Always return a successful result in the standard format.
            success_payload = {
                "result": {
                    "status": "Success",
                    "message": "The operation was successful.",
                    "data": processed_result
                }
            }
            return markpickle.dumps(success_payload)

        except Exception as e:
            # Catch any unexpected exceptions and format them as a standard error response.
            error_payload = {"error": f"An unexpected error occurred: {str(e)}"}
            return markpickle.dumps(error_payload)


def run(arg1: str, arg2: Optional[int] = None) -> str:
    """
    The public, module-level entry point for the agent.
    Its signature MUST EXACTLY MATCH the _run_logic method's signature to ensure
    compatibility with the agent's dynamic tool-calling mechanism.
    """
    return GoldenStandardUtil._run_logic(arg1=arg1, arg2=arg2)


# --- Minimal & Reproducible Test Showcase ---
# This block will only execute when the script is run directly, e.g., via `python utils/golden_standard_util.py`.
# It provides a simple, fast way to verify the utility's core functionality in isolation.
# To adapt for a new util, a developer only needs to update the `test_cases` list below.
if __name__ == "__main__":
    
    # Define all test cases in a simple, data-driven list of dictionaries.
    test_cases = [
        {
            "description": "Test 1: Successful execution with all arguments",
            "args": {"arg1": "Test String", "arg2": 123},
            "assertion": lambda res: "result" in res and res["result"]["data"] == "Processed Test String with value 123"
        },
        {
            "description": "Test 2: Successful execution with optional argument omitted",
            "args": {"arg1": "Another Test"},
            "assertion": lambda res: "result" in res and res["result"]["data"] == "Processed Another Test with value None"
        },
        {
            "description": "Test 3: Error handling for invalid input (empty string)",
            "args": {"arg1": ""},
            "assertion": lambda res: "error" in res and "arg1 cannot be empty" in res["error"]
        }
    ]

    print("="*50)
    print(f"   Running Self-Tests for {__name__}   ")
    print("="*50)
    
    passed_count = 0
    # Generic test runner that iterates through the defined cases.
    for i, test in enumerate(test_cases, 1):
        print(f"\n--- {test['description']} ---")
        try:
            # Execute the utility's run function with the test arguments.
            result_str = run(**test['args'])
            result_dict = markpickle.loads(result_str)
            
            print(f"Input: {test['args']}")
            print(f"Output: {result_dict}")
            
            # Check if the result meets the assertion criteria.
            if test['assertion'](result_dict):
                print("Status: PASSED ✔️")
                passed_count += 1
            else:
                print("Status: FAILED ❌ (Assertion logic failed)")
        except Exception as e:
            print(f"Status: FAILED ❌ (An unexpected exception occurred: {e})")

    # Final summary of the test run.
    print("\n" + "="*50)
    if passed_count == len(test_cases):
        print(f"  Summary: All {len(test_cases)} tests passed! ✅")
    else:
        print(f"  Summary: {passed_count}/{len(test_cases)} tests passed. Please review failures. ❌")
    print("="*50)
