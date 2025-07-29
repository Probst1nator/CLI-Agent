from abc import ABC, abstractmethod
from typing import Any, Type

class UtilBase(ABC):
    """
    Abstract base class for implementing utility functionality.
    
    This class defines the interface that all utility implementations 
    must follow. Subclasses need to implement the run method
    to provide specific utility behavior.
    """
    
    @staticmethod
    @abstractmethod
    def run() -> Any:
        """
        The actual implementation of the utility execution logic.
        
        This method must be overridden by subclasses to provide the
        specific functionality of the utility. Each subclass will
        define its own specific arguments.
        
        Returns:
            Any: The result of the utility execution. The return type depends on
                 the specific utility implementation.
                 
        Raises:
            May raise exceptions specific to the utility implementation.
        """
        pass
    
    @staticmethod
    def get_name(util_cls: Type['UtilBase']) -> str:
        """
        Get the name of a utility class.
        
        By default, this uses the class name with "Util" removed if it exists.
        
        Args:
            util_cls: The utility class to get the name of
            
        Returns:
            The name of the utility
        """
        class_name = util_cls.__name__
        # Remove "Util" suffix if present
        # if class_name.endswith("Util"):
        #     return class_name[:-4].lower()
        return class_name.lower()
    
    @staticmethod
    def get_description(util_cls: Type['UtilBase']) -> str:
        """
        Get the description of a utility class from its docstring.
        
        Args:
            util_cls: The utility class to get the description of
            
        Returns:
            The description of the utility from its docstring
        """
        import inspect
        docstring = inspect.getdoc(util_cls)
        return docstring or "No description available" 