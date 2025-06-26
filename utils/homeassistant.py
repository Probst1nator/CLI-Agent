import os
import requests
import json
from typing import Dict, Optional, Any, Literal, Set
from dotenv import load_dotenv
from datetime import datetime

# Import base class and globals
from py_classes.cls_util_base import UtilBase
from py_classes.globals import g

class HomeAssistant(UtilBase):
    """
    A utility for interacting with a Home Assistant server.
    This tool allows for calling services (e.g., turning on a light),
    retrieving the state of entities, and searching for entities.
    """
    
    _INTERACTED_ENTITIES_FILE = os.path.join(g.CLIAGENT_PERSISTENT_STORAGE_PATH, 'homeassistant_interacted_entities.json')

    @classmethod
    def _load_interacted_entities(cls) -> Dict[str, list]:
        """
        Load the usage history for entities from persistent storage.
        
        Returns:
            Dictionary mapping entity IDs to their usage history list.
        """
        try:
            if os.path.exists(cls._INTERACTED_ENTITIES_FILE):
                with open(cls._INTERACTED_ENTITIES_FILE, 'r') as f:
                    data = json.load(f)
                    return data.get('entities', {})
        except Exception as e:
            pass
        return {}

    @classmethod
    def _save_interacted_entities(cls, interacted_entities: Dict[str, list]) -> None:
        """
        Save the entity usage history to persistent storage, maintaining only latest 50 entries per entity.
        
        Args:
            interacted_entities: Dictionary mapping entity IDs to their usage history lists.
        """
        try:
            os.makedirs(os.path.dirname(cls._INTERACTED_ENTITIES_FILE), exist_ok=True)
            trimmed_entities = {}
            total_entries = 0
            
            for entity_id, history in interacted_entities.items():
                sorted_history = sorted(history, key=lambda x: x.get('timestamp', ''), reverse=True)
                trimmed_entities[entity_id] = sorted_history[:50]
                total_entries += len(trimmed_entities[entity_id])
            
            data = {
                'entities': trimmed_entities,
                'file_last_updated': datetime.now().isoformat(),
                'total_entities_tracked': len(trimmed_entities),
                'total_history_entries': total_entries
            }
            
            with open(cls._INTERACTED_ENTITIES_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            pass

    @classmethod
    def _track_entity_interaction(cls, entity_id: str, interaction_type: str = "state_query", 
                                 service_domain: Optional[str] = None, service_name: Optional[str] = None) -> None:
        """
        Track a new interaction with an entity by adding it to the usage history.
        """
        if not entity_id:
            return
            
        interacted_entities = cls._load_interacted_entities()
        current_time = datetime.now().isoformat()
        
        if interaction_type == "service_call" and service_domain and service_name:
            service_full_name = f"{service_domain}.{service_name}"
            history_entry = {
                "timestamp": current_time,
                "action": "service_call",
                "service": service_full_name,
                "details": f"Called {service_full_name}"
            }
        else:
            history_entry = {
                "timestamp": current_time,
                "action": "state_query",
                "service": None,
                "details": "Queried entity state"
            }
        
        if entity_id not in interacted_entities:
            interacted_entities[entity_id] = []
        
        interacted_entities[entity_id].insert(0, history_entry)
        interacted_entities[entity_id] = interacted_entities[entity_id][:50]
        cls._save_interacted_entities(interacted_entities)

    @staticmethod
    def _get_config() -> tuple[str, str]:
        """
        Retrieves Home Assistant URL and a long-lived access token from environment variables.
        """
        load_dotenv()
        hass_url = os.environ.get("HASS_URL")
        hass_token = os.environ.get("HASS_TOKEN")

        if not hass_url or not hass_token:
            raise ValueError(
                "HASS_URL and HASS_TOKEN environment variables must be set."
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
            if entity_id:
                cls._track_entity_interaction(entity_id, "service_call", domain, service)
            return (
                f"Successfully called service '{domain}.{service}'"
                f"{f' on entity {entity_id}' if entity_id else ''}. "
                f"Confirmation from HA: {response.text}"
            )
        else:
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
        endpoint = "states"
        if entity_id:
            endpoint = f"states/{entity_id}"
        
        api_url = f"{hass_url.rstrip('/')}/api/{endpoint}"
        headers = {"Authorization": f"Bearer {hass_token}"}
        
        response = requests.get(api_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            states = response.json()
            if isinstance(states, list):
                interacted_entities = cls._load_interacted_entities()
                interacted_states = []
                other_states = []
                
                for s in states:
                    entity_id_from_state = s.get('entity_id')
                    friendly_name = s.get('attributes', {}).get('friendly_name', 'N/A')
                    current_state = s.get('state')
                    
                    state_summary_parts = [f"- Entity ID: {entity_id_from_state}, State: {current_state}, Name: {friendly_name}"]
                    
                    if entity_id_from_state in interacted_entities:
                        history = interacted_entities[entity_id_from_state]
                        if history:
                            state_summary_parts.append(f"  └─ Usage History ({len(history)} total entries):")
                            preview_entries = history[:3]
                            for entry in preview_entries:
                                try:
                                    dt = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00'))
                                    formatted_time = dt.strftime("%m-%d %H:%M")
                                    state_summary_parts.append(f"    • {formatted_time}: {entry['details']}")
                                except:
                                    state_summary_parts.append(f"    • {entry['details']}")
                            if len(history) > 3:
                                state_summary_parts.append(f"    • ... and {len(history) - 3} more entries")
                        interacted_states.append('\n'.join(state_summary_parts))
                    else:
                        other_states.append('\n'.join(state_summary_parts))
                
                max_interacted = min(len(interacted_states), 10)
                max_others = max(0, 30 - max_interacted)
                
                selected_interacted = interacted_states[:max_interacted]
                selected_others = other_states[:max_others]
                
                response_parts = [f"Found {len(states)} entities total ({len(interacted_entities)} previously interacted)."]
                if selected_interacted:
                    response_parts.append(f"Showing {len(selected_interacted)} previously interacted entities first:")
                    response_parts.extend(selected_interacted)
                
                if selected_others:
                    if selected_interacted : # Add a separator if interacted entities were also shown
                         response_parts.append(f"\nThen showing {len(selected_others)} other entities:")
                    else: # Only other entities are shown
                         response_parts.append(f"Showing {len(selected_others)} entities:")
                    response_parts.extend(selected_others)
                
                total_shown = len(selected_interacted) + len(selected_others)
                remaining_total = len(states) - total_shown
                if remaining_total > 0:
                    response_parts.append(f"\n... and {remaining_total} more entities not shown.")
                
                return '\n'.join(response_parts)
            else:
                if entity_id:
                    cls._track_entity_interaction(entity_id, "state_query")
                return f"State for '{entity_id}':\n{json.dumps(states, indent=2)}"
        else:
            raise RuntimeError(
                f"Error getting state for '{entity_id if entity_id else 'all entities'}'. "
                f"Status: {response.status_code}. Details: {response.text}"
            )

    @classmethod
    def _search_entities_by_keyword(cls, keyword: str) -> str:
        """
        Private method to search for entities by a keyword in their entity_id or friendly_name.
        """
        hass_url, hass_token = cls._get_config()
        api_url = f"{hass_url.rstrip('/')}/api/states"
        headers = {"Authorization": f"Bearer {hass_token}"}
        
        response = requests.get(api_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            all_states = response.json()
            if not isinstance(all_states, list):
                return "Error: Received unexpected data format from Home Assistant when fetching all states."

            matching_entities_details = []
            lower_keyword = keyword.lower()

            for s in all_states:
                entity_id = s.get('entity_id', '')
                # Ensure friendly_name is a string for searching, even if it's None or not present.
                friendly_name_attr = s.get('attributes', {}).get('friendly_name')
                friendly_name = str(friendly_name_attr) if friendly_name_attr is not None else ''

                if lower_keyword in entity_id.lower() or lower_keyword in friendly_name.lower():
                    current_state = s.get('state', 'N/A')
                    display_name = friendly_name if friendly_name else 'N/A'
                    detail = f"- Entity ID: {entity_id}, State: {current_state}, Name: {display_name}"
                    matching_entities_details.append(detail)
            
            if not matching_entities_details:
                return f"No entities found matching keyword: '{keyword}'"
            
            # Sort results by entity_id for consistent output
            matching_entities_details.sort()
            
            return f"Found {len(matching_entities_details)} entities matching '{keyword}':\n" + '\n'.join(matching_entities_details)
        else:
            raise RuntimeError(
                f"Error searching entities. Failed to fetch all states. "
                f"Status: {response.status_code}. Details: {response.text}"
            )

    @classmethod
    def _get_entity_usage_history(cls, entity_id: str) -> str:
        """
        Get the full usage history for a specific entity.
        """
        interacted_entities = cls._load_interacted_entities()
        
        if entity_id not in interacted_entities or not interacted_entities[entity_id]:
            return f"No usage history found for entity '{entity_id}'."
        
        history = interacted_entities[entity_id]
        response_parts = [f"Usage History for '{entity_id}' ({len(history)} entries):"]
        response_parts.append("=" * 60)
        
        for i, entry in enumerate(history, 1):
            try:
                dt = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00'))
                formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                response_parts.append(f"{i:2d}. {formatted_time} - {entry['details']}")
            except:
                response_parts.append(f"{i:2d}. {entry['details']}")
        
        return '\n'.join(response_parts)

    @classmethod
    async def run(
        cls,
        action: Literal["call_service", "get_state", "list_devices", "get_usage_history", "search_entity_by_keyword"],
        domain: Optional[str] = None,
        service: Optional[str] = None,
        entity_id: Optional[str] = None,
        service_data: Optional[Dict[str, Any]] = None,
        keyword: Optional[str] = None
    ) -> str:
        """
        Interacts with a Home Assistant instance.

        Args:
            action: The action to perform. Must be one of 'call_service', 'get_state', 
                    'list_devices', 'get_usage_history', or 'search_entity_by_keyword'.
            domain: (For 'call_service' action) The domain of the service (e.g., 'light', 'switch').
            service: (For 'call_service' action) The name of the service (e.g., 'turn_on', 'toggle').
            entity_id: Optional entity ID. For 'call_service', it's the target. For 'get_state',
                       it fetches a specific entity's state. For 'get_usage_history', it's required.
            service_data: (For 'call_service' action) Optional dictionary with extra data for the service.
            keyword: (For 'search_entity_by_keyword' action) The keyword to search for in entity IDs and friendly names.

        Returns:
            A string indicating the result of the action.
        """
        try:
            if action == "call_service":
                if not domain or not service:
                    raise ValueError("For 'call_service', 'domain' and 'service' are required.")
                return cls._call_service(domain, service, entity_id, service_data)
            
            elif action == "get_state":
                if not entity_id:
                    return "For 'get_state', 'entity_id' is required. To list all devices, use 'list_devices'."
                return cls._get_state(entity_id)
                
            elif action == "list_devices":
                return cls._get_state() # Gets all states
            
            elif action == "get_usage_history":
                if not entity_id:
                    return "For 'get_usage_history', 'entity_id' is required."
                return cls._get_entity_usage_history(entity_id)

            elif action == "search_entity_by_keyword":
                if not keyword:
                    return "For 'search_entity_by_keyword', 'keyword' is required."
                return cls._search_entities_by_keyword(keyword)

            else:
                valid_actions = "'call_service', 'get_state', 'list_devices', 'get_usage_history', 'search_entity_by_keyword'"
                return f"Error: Invalid action '{action}'. Must be one of {valid_actions}."

        except (ValueError, RuntimeError, requests.exceptions.RequestException) as e:
            return f"Home Assistant tool error: {e}"
        except Exception as e:
            return f"An unexpected error occurred in the Home Assistant tool: {e}"

# --- Example Usage (for testing purposes) ---
if __name__ == "__main__":
    # To test this script, set the HASS_URL and HASS_TOKEN environment variables.
    # Example:
    # export HASS_URL="http://192.168.1.100:8123"
    # export HASS_TOKEN="your-long-lived-access-token"
    
    # Create dummy interacted_entities.json for testing if it doesn't exist
    if not os.path.exists(HomeAssistant._INTERACTED_ENTITIES_FILE):
        HomeAssistant._save_interacted_entities({
            "light.example_light": [{
                "timestamp": datetime.now().isoformat(),
                "action": "service_call",
                "service": "light.turn_on",
                "details": "Called light.turn_on"
            }]
        })

    if not os.environ.get("HASS_URL") or not os.environ.get("HASS_TOKEN"):
        print("\nSkipping tests: HASS_URL and HASS_TOKEN environment variables are not set.")
    else:
        print("\n--- Test Case 1: Listing all devices ---")
        try:
            result1 = HomeAssistant.run(action="list_devices")
            print(f"Result:\n{result1}")
        except Exception as e:
            print(f"Test Case 1 FAILED: {e}")

        print("\n--- Test Case 2: Getting state of a specific, known device (e.g., sun.sun) ---")
        # Most HA instances have 'sun.sun'. If not, replace with a known entity_id.
        test_entity_id = "sun.sun" 
        try:
            result2 = HomeAssistant.run(action="get_state", entity_id=test_entity_id)
            print(f"Result for {test_entity_id}: {result2}")
            # Test getting usage history for this entity (it might be empty if not interacted via tool)
            result_history = HomeAssistant.run(action="get_usage_history", entity_id=test_entity_id)
            print(f"\nUsage history for {test_entity_id}:\n{result_history}")

        except Exception as e:
            print(f"Test Case 2 FAILED: {e}")

        print("\n--- Test Case 3: Calling a service (persistent notification) ---")
        try:
            result3 = HomeAssistant.run(
                action="call_service",
                domain="persistent_notification",
                service="create",
                service_data={
                    "notification_id": f"tool_test_{datetime.now().strftime('%H%M%S')}",
                    "title": "Tool Test Notification",
                    "message": "The HomeAssistant tool's 'call_service' is working!"
                }
            )
            print(f"Result: {result3}")
        except Exception as e:
            print(f"Test Case 3 FAILED: {e}")

        print("\n--- Test Case 5: Search for entities by keyword (e.g., 'light') ---")
        search_keyword = "light" # Choose a keyword likely to have matches in your HA
        try:
            result5 = HomeAssistant.run(action="search_entity_by_keyword", keyword=search_keyword)
            print(f"Search results for '{search_keyword}':\n{result5}")
        except Exception as e:
            print(f"Test Case 5 FAILED: {e}")

        print("\n--- Test Case 6: Search for entities with a non-matching keyword ---")
        non_matching_keyword = "this_keyword_should_not_exist_in_any_entity_12345"
        try:
            result6 = HomeAssistant.run(action="search_entity_by_keyword", keyword=non_matching_keyword)
            print(f"Search results for '{non_matching_keyword}':\n{result6}")
        except Exception as e:
            print(f"Test Case 6 FAILED: {e}")
            
        print("\n--- Test Case 7: Search without providing keyword ---")
        try:
            result7 = HomeAssistant.run(action="search_entity_by_keyword")
            print(f"Result: {result7}")
        except Exception as e:
            print(f"Test Case 7 FAILED with an unexpected exception: {e}")


    print("\n--- Test Case 4: Invalid action call ---") # Renumbered from original
    try:
        result4 = HomeAssistant.run(action="invalid_action_name") # type: ignore
        print(f"Result: {result4}")
    except Exception as e:
        print(f"Test Case 4 FAILED with an unexpected exception: {e}")
        
    print("\n--- Finished testing ---")