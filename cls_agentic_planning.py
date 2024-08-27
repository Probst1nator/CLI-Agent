from typing import Callable, List
import inspect

class Parameter:
    """
    Represents a function parameter with its name, type, description, and required status.
    """

    def __init__(self, name: str, type: str, description: str, required: bool = True):
        """
        Initialize a Parameter object.

        Args:
            name (str): The name of the parameter.
            type (str): The type of the parameter.
            description (str): A description of the parameter.
            required (bool, optional): Whether the parameter is required. Defaults to True.
        """
        self.name = name
        self.type = type
        self.description = description
        self.required = required

    def __repr__(self) -> str:
        """
        Return a string representation of the Parameter object.

        Returns:
            str: A string representation of the Parameter.
        """
        return f"Parameter(name='{self.name}', type='{self.type}', description='{self.description}', required={self.required})"

class Tool:
    """
    Represents a tool (function) with its name, purpose, and parameters.
    """

    def __init__(self, function: Callable, purpose: str):
        """
        Initialize a Tool object.

        Args:
            function (Callable): The function that this tool represents.
            purpose (str): A description of the tool's purpose.
        """
        self.name = function.__name__
        self.purpose = purpose
        self.parameters = self._extract_parameters(function)

    def _extract_parameters(self, function: Callable) -> List[Parameter]:
        """
        Extract parameters from the given function.

        Args:
            function (Callable): The function to extract parameters from.

        Returns:
            List[Parameter]: A list of Parameter objects representing the function's parameters.
        """
        parameters = []
        # Get the function's signature
        signature = inspect.signature(function)
        for name, param in signature.parameters.items():
            # Determine parameter type
            param_type = str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any"
            # Determine if the parameter is required
            required = param.default == inspect.Parameter.empty
            # Get parameter description from docstring if available
            description = self._get_param_description(function, name)
            # Create a Parameter object and add it to the list
            parameters.append(Parameter(name, param_type, description, required))
        return parameters

    def _get_param_description(self, function: Callable, param_name: str) -> str:
        """
        Get the description of a parameter from the function's docstring.

        Args:
            function (Callable): The function containing the parameter.
            param_name (str): The name of the parameter.

        Returns:
            str: The description of the parameter, or "No description available" if not found.
        """
        docstring = inspect.getdoc(function)
        if docstring:
            lines = docstring.split('\n')
            for line in lines:
                if line.strip().startswith(f"{param_name}:"):
                    return line.split(':', 1)[1].strip()
        return "No description available"

    def __repr__(self) -> str:
        """
        Return a string representation of the Tool object.

        Returns:
            str: A string representation of the Tool.
        """
        params = '\n    '.join(repr(param) for param in self.parameters)
        return f"Tool(name='{self.name}', purpose='{self.purpose}',\n  parameters=[\n    {params}\n  ])"


# class AgenticPlanning:
#     def __init__(self, user_task: str, tools: List[Tool]):
        