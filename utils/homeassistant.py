import os
import requests
import json
from typing import Dict, Optional, Any, Literal
from dotenv import load_dotenv

# Import base class
from py_classes.cls_util_base import UtilBase

class HomeAssistant(UtilBase):
    """
    A utility for interacting with a Home Assistant server.
    This tool allows for calling services (e.g., turning on a light) and
    retrieving the state of entities.
    """

    @staticmethod
    def _get_config() -> tuple[str, str]:
        """
        Retrieves Home Assistant URL and a long-lived access token from environment variables.
        
        Returns:
            A tuple containing the Home Assistant URL and the access token.
            
        Raises:
            ValueError: If the required environment variables (HASS_URL, HASS_TOKEN) are not set.
        """
        # Load variables from .env file into the environment
        load_dotenv()

        hass_url = os.environ.get("HASS_URL")
        hass_token = os.environ.get("HASS_TOKEN")

        if not hass_url or not hass_token:
            raise ValueError(
                "HASS_URL and HASS_TOKEN environment variables must be set "
                "to connect to your Home Assistant instance."
            )
        return hass_url, hass_token

    @classmethod
    def _call_service(
        cls,
        domain: str,
        service: str,
        entity_id: Optional[str] = None,
        service_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Private method to call a service in Home Assistant.
        """
        hass_url, hass_token = cls._get_config()
        api_url = f"{hass_url.rstrip('/')}/api/services/{domain}/{service}"
        headers = {
            "Authorization": f"Bearer {hass_token}",
            "Content-Type": "application/json",
        }
        payload = service_data or {}
        if entity_id:
            payload['entity_id'] = entity_id

        response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=10)
        
        if response.status_code in [200, 201]:
            # The response body contains the state of entities that changed
            return (
                f"Successfully called service '{domain}.{service}'"
                f"{f' on entity {entity_id}' if entity_id else ''}. "
                f"Confirmation from HA: {response.text}"
            )
        else:
            # Raise an error for the calling code to handle
            raise RuntimeError(
                f"Error calling service '{domain}.{service}'. "
                f"Status: {response.status_code}. Details: {response.text}"
            )

    @classmethod
    def _get_state(cls, entity_id: Optional[str] = None) -> str:
        """
        Private method to retrieve the state of one or all entities from Home Assistant.
        """
        hass_url, hass_token = cls._get_config()
        # The endpoint is '/api/states/<entity_id>' for a single entity or '/api/states' for all
        endpoint = "states"
        if entity_id:
            endpoint = f"states/{entity_id}"
        
        api_url = f"{hass_url.rstrip('/')}/api/{endpoint}"
        headers = {"Authorization": f"Bearer {hass_token}"}
        
        response = requests.get(api_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            states = response.json()
            # If we received a list of all states, provide a summarized view.
            if isinstance(states, list):
                summary = [
                    f"- Entity ID: {s.get('entity_id')}, State: {s.get('state')}, Name: {s.get('attributes', {}).get('friendly_name', 'N/A')}"
                    for s in states
                ]
                # Return a manageable number of entities to avoid an overly long response
                if len(summary) > 75:
                    summary_text = '\n'.join(summary[:75])
                    summary_text += f"\n... and {len(summary) - 75} more entities."
                    return f"Found {len(summary)} entities. Showing the first 75:\n{summary_text}"
                return f"Found {len(summary)} entities:\n" + '\n'.join(summary)
            # If we received a single state object, return its full JSON representation.
            else:
                return f"State for '{entity_id}':\n{json.dumps(states, indent=2)}"
        else:
            # Raise an error for the calling code to handle
            raise RuntimeError(
                f"Error getting state for '{entity_id if entity_id else 'all entities'}'. "
                f"Status: {response.status_code}. Details: {response.text}"
            )

    @classmethod
    def run(
        cls,
        action: Literal["call_service", "get_state", "list_devices"],
        domain: Optional[str] = None,
        service: Optional[str] = None,
        entity_id: Optional[str] = None,
        service_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Interacts with a Home Assistant instance by calling services or getting states.

        Args:
            action: The action to perform. Must be one of 'call_service', 'get_state', or 'list_devices'.
            domain: (For 'call_service' action) The domain of the service (e.g., 'light', 'switch').
            service: (For 'call_service' action) The name of the service (e.g., 'turn_on', 'toggle').
            entity_id: Optional entity ID. For 'call_service', it's the target. For 'get_state',
                       it fetches a specific entity's state. Not used for 'list_devices'.
            service_data: (For 'call_service' action) Optional dictionary with extra data for the service.

        Returns:
            A string indicating the result of the action.
        """
        try:
            if action == "call_service":
                if not domain or not service:
                    raise ValueError("For the 'call_service' action, 'domain' and 'service' are required.")
                return cls._call_service(domain, service, entity_id, service_data)
            
            elif action == "get_state":
                if not entity_id:
                    return "To get the state of a specific device, you must provide its 'entity_id'. To get all devices, use the 'list_devices' action instead."
                return cls._get_state(entity_id)
                
            elif action == "list_devices":
                # 'list_devices' is a user-friendly alias for getting the state of all devices.
                return cls._get_state()

            else:
                # This should not happen if using Literal for type hinting, but it's good practice.
                return f"Error: Invalid action '{action}'. Must be 'call_service', 'get_state', or 'list_devices'."

        except (ValueError, RuntimeError, requests.exceptions.RequestException) as e:
            # Return a clear and actionable error message
            return f"Home Assistant tool error: {e}"
        except Exception as e:
            # Catch any other unexpected errors
            return f"An unexpected error occurred in the Home Assistant tool: {e}"

# --- Example Usage (for testing purposes) ---
if __name__ == "__main__":
    # To test this script, set the HASS_URL and HASS_TOKEN environment variables.
    # Example:
    # export HASS_URL="http://192.168.1.100:8123"
    # export HASS_TOKEN="your-long-lived-access-token"
    
    if not os.environ.get("HASS_URL") or not os.environ.get("HASS_TOKEN"):
        print("\nSkipping tests: HASS_URL and HASS_TOKEN environment variables are not set.")
    else:
        print("\n--- Test Case 1: Listing all devices (the 'fix') ---")
        try:
            result1 = HomeAssistant.run(action="list_devices")
            print(f"Result:\n{result1}")
        except Exception as e:
            print(f"Test Case 1 FAILED: {e}")

        print("\n--- Test Case 2: Getting state of a specific, known device ---")
        try:
            # Using 'sun.sun' as it is a common entity in Home Assistant
            result2 = HomeAssistant.run(action="get_state", entity_id="sun.sun")
            print(f"Result: {result2}")
        except Exception as e:
            print(f"Test Case 2 FAILED: {e}")

        print("\n--- Test Case 3: Calling a service (e.g., sending a notification) ---")
        try:
            result3 = HomeAssistant.run(
                action="call_service",
                domain="persistent_notification",
                service="create",
                service_data={
                    "notification_id": "tool_test_456",
                    "title": "Tool Test",
                    "message": "The updated HomeAssistant tool is working!"
                }
            )
            print(f"Result: {result3}")
        except Exception as e:
            print(f"Test Case 3 FAILED: {e}")

    print("\n--- Test Case 4: Invalid action call ---")
    try:
        # This will be caught by the run method's error handling
        result4 = HomeAssistant.run(action="invalid_action_name")
        print(f"Result: {result4}")
    except Exception as e:
        print(f"Test Case 4 FAILED with an unexpected exception: {e}")
        
    print("\n--- Finished testing ---")