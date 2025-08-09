from typing import Any, Dict

from py_classes.cls_util_base import UtilBase

class ExampleUtil(UtilBase):
    """
    A simple example utility that demonstrates the UtilBase implementation.
    
    This utility performs a basic calculation or text operation based on the inputs.
    """
    
    @staticmethod
    async def _run_logic(operation: str = "add", values: Dict[str, Any] = None) -> Any:
        """
        Perform a basic operation based on the input parameters.
        The run methods args are needed to be kept minimal to reduce usage complexity.
        
        Args:
            operation: The operation to perform ("add", "multiply", "concat")
            values: Dictionary of values to use in the operation
            
        Returns:
            The result of the operation
        """
        if values is None:
            values = {}
            
        if operation == "add":
            # Add all numeric values
            result = sum(v for v in values.values() if isinstance(v, (int, float)))
            return result
            
        elif operation == "multiply":
            # Multiply all numeric values
            result = 1
            for v in values.values():
                if isinstance(v, (int, float)):
                    result *= v
            return result
            
        elif operation == "concat":
            # Concatenate all string values
            result = ""
            for v in values.values():
                if isinstance(v, str):
                    result += v
            return result
            
        else:
            return f"Unknown operation: {operation}"


# Module-level run function for CLI-Agent compatibility
def run(operation: str = "add", values: Dict[str, Any] = None) -> Any:
    """
    Module-level wrapper for ExampleUtil._run_logic() to maintain compatibility with CLI-Agent.
    
    Args:
        operation (str): The operation to perform ("add", "multiply", "concat")
        values (Dict[str, Any]): Dictionary of values to use in the operation
        
    Returns:
        Any: The result of the operation
    """
    import asyncio
    return asyncio.run(ExampleUtil._run_logic(operation=operation, values=values)) 