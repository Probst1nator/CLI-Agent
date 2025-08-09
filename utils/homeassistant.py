import os
import requests
import json
import yaml
from typing import Dict, Optional, Any, Literal
from dotenv import load_dotenv
from datetime import datetime

# Import base class and globals
from py_classes.cls_util_base import UtilBase
from py_classes.globals import g

class HomeAssistant(UtilBase):
    """
    A utility for interacting with a Home Assistant server.
    This tool allows for calling services, retrieving the state of entities,
    searching for entities, and managing automations.
    Returns structured JSON responses for reliable communication.
    """
    
    @staticmethod
    def get_metadata() -> Dict[str, Any]:
        return {
            "keywords": ["smart home", "lights", "thermostat", "automation", "HA", "turn on", "turn off", "dim", "set temperature", "lock door"],
            "use_cases": [
                "Turn on the living room lights.",
                "What is the current temperature in the bedroom?",
                "Dim the office lights to 50%.",
                "Create an automation to turn off all lights when I leave home.",
                "List all available smart devices."
            ],
            "arguments": {
                "action": "The specific operation to perform (e.g., 'call_service', 'get_state').",
                "domain": "The service domain (e.g., 'light', 'switch').",
                "service": "The service name (e.g., 'turn_on', 'toggle').",
                "entity_id": "The unique ID of the device or entity to control (e.g., 'light.living_room_lamp')."
            },
            "code_examples": [
                {
                    "description": "Turn on a light",
                    "code": "from utils.homeassistant import HomeAssistant\nresult = HomeAssistant.run(action='call_service', domain='light', service='turn_on', entity_id='light.living_room')"
                },
                {
                    "description": "Get the state of a sensor",
                    "code": "from utils.homeassistant import HomeAssistant\nresult = HomeAssistant.run(action='get_state', entity_id='sensor.bedroom_temperature')"
                },
                {
                    "description": "List all devices",
                    "code": "from utils.homeassistant import HomeAssistant\nresult = HomeAssistant.run(action='list_devices')"
                }
            ]
        }
    
    _INTERACTED_ENTITIES_FILE = os.path.join(g.CLIAGENT_PERSISTENT_STORAGE_PATH, 'homeassistant_interacted_entities.json')
    _AUTOMATION_BACKUP_DIR = os.path.join(g.CLIAGENT_PERSISTENT_STORAGE_PATH, 'homeassistant_automation_backups')

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
        except Exception:
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
        except Exception:
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
        elif interaction_type == "automation_action":
            history_entry = {
                "timestamp": current_time,
                "action": "automation_action",
                "service": service_name,
                "details": f"Automation action: {service_name}"
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
    def _make_request(cls, method: str, endpoint: str, payload: Optional[Dict[str, Any]] = None) -> requests.Response:
        """
        Helper method to make HTTP requests to Home Assistant API.
        """
        hass_url, hass_token = cls._get_config()
        api_url = f"{hass_url.rstrip('/')}/api/{endpoint}"
        headers = {
            "Authorization": f"Bearer {hass_token}",
            "Content-Type": "application/json",
        }
        
        if method.upper() == "GET":
            return requests.get(api_url, headers=headers, timeout=10)
        elif method.upper() == "POST":
            return requests.post(api_url, headers=headers, data=json.dumps(payload) if payload else None, timeout=10)
        elif method.upper() == "PUT":
            return requests.put(api_url, headers=headers, data=json.dumps(payload) if payload else None, timeout=10)
        elif method.upper() == "DELETE":
            return requests.delete(api_url, headers=headers, timeout=10)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

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
        Returns JSON string with result or error.
        """
        try:
            payload = service_data or {}
            if entity_id:
                payload['entity_id'] = entity_id

            response = cls._make_request("POST", f"services/{domain}/{service}", payload)
            
            if response.status_code in [200, 201]:
                if entity_id:
                    cls._track_entity_interaction(entity_id, "service_call", domain, service)
                
                result = {
                    "result": {
                        "status": "Success",
                        "service": f"{domain}.{service}",
                        "entity_id": entity_id,
                        "message": f"Successfully called service '{domain}.{service}'"
                                 f"{f' on entity {entity_id}' if entity_id else ''}",
                        "ha_response": response.text
                    }
                }
                return json.dumps(result, indent=2)
            else:
                error_result = {
                    "error": f"Failed to call service '{domain}.{service}'. "
                           f"Status: {response.status_code}. Details: {response.text}"
                }
                return json.dumps(error_result, indent=2)
                
        except requests.exceptions.ConnectionError:
            error_result = {"error": "Cannot connect to Home Assistant. Check HASS_URL and network connection."}
            return json.dumps(error_result, indent=2)
        except requests.exceptions.Timeout:
            error_result = {"error": "Home Assistant request timed out. Server may be slow or unresponsive."}
            return json.dumps(error_result, indent=2)
        except ValueError as e:
            error_result = {"error": str(e)}
            return json.dumps(error_result, indent=2)
        except Exception as e:
            error_result = {"error": f"Unexpected error calling service: {str(e)}"}
            return json.dumps(error_result, indent=2)

    @classmethod
    def _get_state(cls, entity_id: Optional[str] = None) -> str:
        """
        Private method to retrieve the state of one or all entities from Home Assistant.
        Returns JSON string with result or error.
        """
        try:
            endpoint = "states"
            if entity_id:
                endpoint = f"states/{entity_id}"
            
            response = cls._make_request("GET", endpoint)
            
            if response.status_code == 200:
                states = response.json()
                if isinstance(states, list):
                    # Handle list of all states
                    if len(states) == 0:
                        result = {
                            "result": {
                                "status": "No entities found",
                                "message": "Home Assistant returned no entities. This might indicate an empty installation or authentication issues.",
                                "entity_count": 0
                            }
                        }
                        return json.dumps(result, indent=2)
                    
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
                        if selected_interacted: # Add a separator if interacted entities were also shown
                             response_parts.append(f"\nThen showing {len(selected_others)} other entities:")
                        else: # Only other entities are shown
                             response_parts.append(f"Showing {len(selected_others)} entities:")
                        response_parts.extend(selected_others)
                    
                    total_shown = len(selected_interacted) + len(selected_others)
                    remaining_total = len(states) - total_shown
                    if remaining_total > 0:
                        response_parts.append(f"\n... and {remaining_total} more entities not shown.")
                    
                    result = {
                        "result": {
                            "status": "Success",
                            "entity_count": len(states),
                            "entities_shown": total_shown,
                            "entities_data": '\n'.join(response_parts)
                        }
                    }
                    return json.dumps(result, indent=2)
                else:
                    # Handle single entity state
                    if entity_id:
                        cls._track_entity_interaction(entity_id, "state_query")
                    
                    result = {
                        "result": {
                            "status": "Success",
                            "entity_id": entity_id,
                            "state_data": states
                        }
                    }
                    return json.dumps(result, indent=2)
            elif response.status_code == 404:
                error_result = {"error": f"Entity '{entity_id}' not found in Home Assistant."}
                return json.dumps(error_result, indent=2)
            else:
                error_result = {
                    "error": f"Failed to get state for '{entity_id if entity_id else 'all entities'}'. "
                           f"Status: {response.status_code}. Details: {response.text}"
                }
                return json.dumps(error_result, indent=2)
                
        except requests.exceptions.ConnectionError:
            error_result = {"error": "Cannot connect to Home Assistant. Check HASS_URL and network connection."}
            return json.dumps(error_result, indent=2)
        except requests.exceptions.Timeout:
            error_result = {"error": "Home Assistant request timed out. Server may be slow or unresponsive."}
            return json.dumps(error_result, indent=2)
        except ValueError as e:
            error_result = {"error": str(e)}
            return json.dumps(error_result, indent=2)
        except Exception as e:
            error_result = {"error": f"Unexpected error getting state: {str(e)}"}
            return json.dumps(error_result, indent=2)

    @classmethod
    def _search_entities_by_keyword(cls, keyword: str) -> str:
        """
        Private method to search for entities by a keyword in their entity_id or friendly_name.
        Returns JSON string with result or error.
        """
        try:
            response = cls._make_request("GET", "states")
            
            if response.status_code == 200:
                all_states = response.json()
                if not isinstance(all_states, list):
                    error_result = {"error": "Received unexpected data format from Home Assistant when fetching all states."}
                    return json.dumps(error_result, indent=2)

                if len(all_states) == 0:
                    result = {
                        "result": {
                            "status": "No entities in Home Assistant",
                            "message": "Home Assistant returned no entities at all. This might indicate an empty installation or authentication issues.",
                            "matches_found": 0,
                            "keyword": keyword
                        }
                    }
                    return json.dumps(result, indent=2)

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
                    result = {
                        "result": {
                            "status": "No matches found",
                            "message": f"No entities found matching keyword '{keyword}'. Searched through {len(all_states)} entities.",
                            "matches_found": 0,
                            "keyword": keyword,
                            "total_entities_searched": len(all_states)
                        }
                    }
                    return json.dumps(result, indent=2)
                
                # Sort results by entity_id for consistent output
                matching_entities_details.sort()
                
                result = {
                    "result": {
                        "status": "Success",
                        "message": f"Found {len(matching_entities_details)} entities matching '{keyword}'",
                        "matches_found": len(matching_entities_details),
                        "keyword": keyword,
                        "total_entities_searched": len(all_states),
                        "matches": '\n'.join(matching_entities_details)
                    }
                }
                return json.dumps(result, indent=2)
                
            else:
                error_result = {
                    "error": f"Failed to search entities. Could not fetch all states. "
                           f"Status: {response.status_code}. Details: {response.text}"
                }
                return json.dumps(error_result, indent=2)
                
        except requests.exceptions.ConnectionError:
            error_result = {"error": "Cannot connect to Home Assistant. Check HASS_URL and network connection."}
            return json.dumps(error_result, indent=2)
        except requests.exceptions.Timeout:
            error_result = {"error": "Home Assistant request timed out. Server may be slow or unresponsive."}
            return json.dumps(error_result, indent=2)
        except ValueError as e:
            error_result = {"error": str(e)}
            return json.dumps(error_result, indent=2)
        except Exception as e:
            error_result = {"error": f"Unexpected error searching entities: {str(e)}"}
            return json.dumps(error_result, indent=2)

    @classmethod
    def _get_entity_usage_history(cls, entity_id: str) -> str:
        """
        Get the full usage history for a specific entity.
        Returns JSON string with result or error.
        """
        try:
            interacted_entities = cls._load_interacted_entities()
            
            if entity_id not in interacted_entities or not interacted_entities[entity_id]:
                result = {
                    "result": {
                        "status": "No history found",
                        "message": f"No usage history found for entity '{entity_id}'. This entity has not been interacted with via this tool.",
                        "entity_id": entity_id,
                        "history_entries": 0
                    }
                }
                return json.dumps(result, indent=2)
            
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
            
            result = {
                "result": {
                    "status": "Success",
                    "entity_id": entity_id,
                    "history_entries": len(history),
                    "history_data": '\n'.join(response_parts)
                }
            }
            return json.dumps(result, indent=2)
            
        except Exception as e:
            error_result = {"error": f"Error retrieving usage history for '{entity_id}': {str(e)}"}
            return json.dumps(error_result, indent=2)

    @classmethod
    def _manage_automations(cls, operation: str, automation_id: Optional[str] = None, 
                           automation_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Private method to manage automations (CRUD operations).
        Returns JSON string with result or error.
        """
        try:
            if operation == "list":
                response = cls._make_request("GET", "config/automation/config")
                if response.status_code == 200:
                    automations = response.json()
                    if not automations:
                        result = {
                            "result": {
                                "status": "No automations found",
                                "message": "No automations are currently configured in Home Assistant.",
                                "automation_count": 0
                            }
                        }
                        return json.dumps(result, indent=2)
                    
                    automation_summaries = []
                    for auto in automations:
                        auto_id = auto.get('id', 'N/A')
                        alias = auto.get('alias', 'Unnamed')
                        description = auto.get('description', 'No description')
                        mode = auto.get('mode', 'single')
                        trigger_count = len(auto.get('trigger', []))
                        action_count = len(auto.get('action', []))
                        
                        summary = f"- ID: {auto_id}, Name: {alias}"
                        summary += f"\n  Description: {description}"
                        summary += f"\n  Mode: {mode}, Triggers: {trigger_count}, Actions: {action_count}"
                        automation_summaries.append(summary)
                    
                    result = {
                        "result": {
                            "status": "Success",
                            "automation_count": len(automations),
                            "automations": '\n'.join(automation_summaries)
                        }
                    }
                    return json.dumps(result, indent=2)
                else:
                    error_result = {
                        "error": f"Failed to list automations. Status: {response.status_code}. Details: {response.text}"
                    }
                    return json.dumps(error_result, indent=2)

            elif operation == "get":
                if not automation_id:
                    error_result = {"error": "automation_id is required for 'get' operation."}
                    return json.dumps(error_result, indent=2)
                
                response = cls._make_request("GET", f"config/automation/config/{automation_id}")
                if response.status_code == 200:
                    automation = response.json()
                    result = {
                        "result": {
                            "status": "Success",
                            "automation_id": automation_id,
                            "automation_data": automation
                        }
                    }
                    return json.dumps(result, indent=2)
                elif response.status_code == 404:
                    error_result = {"error": f"Automation '{automation_id}' not found."}
                    return json.dumps(error_result, indent=2)
                else:
                    error_result = {
                        "error": f"Failed to get automation '{automation_id}'. Status: {response.status_code}. Details: {response.text}"
                    }
                    return json.dumps(error_result, indent=2)

            elif operation == "create":
                if not automation_config:
                    error_result = {"error": "automation_config is required for 'create' operation."}
                    return json.dumps(error_result, indent=2)
                
                response = cls._make_request("POST", "config/automation/config", automation_config)
                if response.status_code in [200, 201]:
                    result = {
                        "result": {
                            "status": "Success",
                            "message": f"Successfully created automation '{automation_config.get('alias', automation_config.get('id', 'Unknown'))}'",
                            "automation_config": automation_config
                        }
                    }
                    return json.dumps(result, indent=2)
                else:
                    error_result = {
                        "error": f"Failed to create automation. Status: {response.status_code}. Details: {response.text}"
                    }
                    return json.dumps(error_result, indent=2)

            elif operation == "update":
                if not automation_id or not automation_config:
                    error_result = {"error": "Both automation_id and automation_config are required for 'update' operation."}
                    return json.dumps(error_result, indent=2)
                
                response = cls._make_request("PUT", f"config/automation/config/{automation_id}", automation_config)
                if response.status_code == 200:
                    result = {
                        "result": {
                            "status": "Success",
                            "message": f"Successfully updated automation '{automation_id}'",
                            "automation_id": automation_id,
                            "automation_config": automation_config
                        }
                    }
                    return json.dumps(result, indent=2)
                elif response.status_code == 404:
                    error_result = {"error": f"Automation '{automation_id}' not found for update."}
                    return json.dumps(error_result, indent=2)
                else:
                    error_result = {
                        "error": f"Failed to update automation '{automation_id}'. Status: {response.status_code}. Details: {response.text}"
                    }
                    return json.dumps(error_result, indent=2)

            elif operation == "delete":
                if not automation_id:
                    error_result = {"error": "automation_id is required for 'delete' operation."}
                    return json.dumps(error_result, indent=2)
                
                response = cls._make_request("DELETE", f"config/automation/config/{automation_id}")
                if response.status_code == 200:
                    result = {
                        "result": {
                            "status": "Success",
                            "message": f"Successfully deleted automation '{automation_id}'",
                            "automation_id": automation_id
                        }
                    }
                    return json.dumps(result, indent=2)
                elif response.status_code == 404:
                    error_result = {"error": f"Automation '{automation_id}' not found for deletion."}
                    return json.dumps(error_result, indent=2)
                else:
                    error_result = {
                        "error": f"Failed to delete automation '{automation_id}'. Status: {response.status_code}. Details: {response.text}"
                    }
                    return json.dumps(error_result, indent=2)

            else:
                error_result = {"error": f"Invalid automation operation '{operation}'. Valid operations: list, get, create, update, delete"}
                return json.dumps(error_result, indent=2)

        except requests.exceptions.ConnectionError:
            error_result = {"error": "Cannot connect to Home Assistant. Check HASS_URL and network connection."}
            return json.dumps(error_result, indent=2)
        except requests.exceptions.Timeout:
            error_result = {"error": "Home Assistant request timed out. Server may be slow or unresponsive."}
            return json.dumps(error_result, indent=2)
        except ValueError as e:
            error_result = {"error": str(e)}
            return json.dumps(error_result, indent=2)
        except Exception as e:
            error_result = {"error": f"Unexpected error managing automations: {str(e)}"}
            return json.dumps(error_result, indent=2)

    @classmethod
    def _control_automation(cls, action: str, automation_entity_id: str) -> str:
        """
        Private method to control automation state (enable/disable/trigger).
        Returns JSON string with result or error.
        """
        try:
            valid_actions = ["enable", "disable", "trigger", "reload"]
            if action not in valid_actions:
                error_result = {"error": f"Invalid automation action '{action}'. Valid actions: {', '.join(valid_actions)}"}
                return json.dumps(error_result, indent=2)

            if action == "reload":
                # Reload all automations
                response = cls._make_request("POST", "services/automation/reload")
                if response.status_code in [200, 201]:
                    result = {
                        "result": {
                            "status": "Success",
                            "message": "Successfully reloaded all automations",
                            "action": "reload"
                        }
                    }
                    return json.dumps(result, indent=2)
                else:
                    error_result = {
                        "error": f"Failed to reload automations. Status: {response.status_code}. Details: {response.text}"
                    }
                    return json.dumps(error_result, indent=2)
            else:
                # Control specific automation
                service_map = {
                    "enable": "turn_on",
                    "disable": "turn_off",
                    "trigger": "trigger"
                }
                
                service = service_map[action]
                payload = {"entity_id": automation_entity_id}
                
                response = cls._make_request("POST", f"services/automation/{service}", payload)
                if response.status_code in [200, 201]:
                    cls._track_entity_interaction(automation_entity_id, "automation_action", service_name=f"automation.{service}")
                    
                    result = {
                        "result": {
                            "status": "Success",
                            "message": f"Successfully {action}d automation '{automation_entity_id}'",
                            "action": action,
                            "entity_id": automation_entity_id
                        }
                    }
                    return json.dumps(result, indent=2)
                else:
                    error_result = {
                        "error": f"Failed to {action} automation '{automation_entity_id}'. Status: {response.status_code}. Details: {response.text}"
                    }
                    return json.dumps(error_result, indent=2)

        except requests.exceptions.ConnectionError:
            error_result = {"error": "Cannot connect to Home Assistant. Check HASS_URL and network connection."}
            return json.dumps(error_result, indent=2)
        except requests.exceptions.Timeout:
            error_result = {"error": "Home Assistant request timed out. Server may be slow or unresponsive."}
            return json.dumps(error_result, indent=2)
        except ValueError as e:
            error_result = {"error": str(e)}
            return json.dumps(error_result, indent=2)
        except Exception as e:
            error_result = {"error": f"Unexpected error controlling automation: {str(e)}"}
            return json.dumps(error_result, indent=2)

    @classmethod
    def _backup_restore_automations(cls, operation: str, backup_name: Optional[str] = None) -> str:
        """
        Private method to backup or restore automations.
        Returns JSON string with result or error.
        """
        try:
            os.makedirs(cls._AUTOMATION_BACKUP_DIR, exist_ok=True)
            
            if operation == "backup":
                if not backup_name:
                    backup_name = f"automation_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Get all automations
                response = cls._make_request("GET", "config/automation/config")
                if response.status_code != 200:
                    error_result = {
                        "error": f"Failed to retrieve automations for backup. Status: {response.status_code}. Details: {response.text}"
                    }
                    return json.dumps(error_result, indent=2)
                
                automations = response.json()
                backup_file = os.path.join(cls._AUTOMATION_BACKUP_DIR, f"{backup_name}.yaml")
                
                with open(backup_file, 'w') as f:
                    yaml.dump(automations, f, default_flow_style=False, allow_unicode=True)
                
                result = {
                    "result": {
                        "status": "Success",
                        "message": f"Successfully backed up {len(automations)} automations",
                        "backup_name": backup_name,
                        "backup_file": backup_file,
                        "automation_count": len(automations)
                    }
                }
                return json.dumps(result, indent=2)

            elif operation == "restore":
                if not backup_name:
                    error_result = {"error": "backup_name is required for restore operation."}
                    return json.dumps(error_result, indent=2)
                
                backup_file = os.path.join(cls._AUTOMATION_BACKUP_DIR, f"{backup_name}.yaml")
                if not os.path.exists(backup_file):
                    error_result = {"error": f"Backup file '{backup_name}.yaml' not found in {cls._AUTOMATION_BACKUP_DIR}"}
                    return json.dumps(error_result, indent=2)
                
                with open(backup_file, 'r') as f:
                    automations = yaml.safe_load(f)
                
                if not isinstance(automations, list):
                    error_result = {"error": "Invalid backup file format. Expected a list of automations."}
                    return json.dumps(error_result, indent=2)
                
                success_count = 0
                failed_count = 0
                
                for automation in automations:
                    try:
                        response = cls._make_request("POST", "config/automation/config", automation)
                        if response.status_code in [200, 201]:
                            success_count += 1
                        else:
                            failed_count += 1
                    except Exception:
                        failed_count += 1
                
                result = {
                    "result": {
                        "status": "Success" if failed_count == 0 else "Partial Success",
                        "message": f"Restored {success_count}/{len(automations)} automations from backup '{backup_name}'",
                        "backup_name": backup_name,
                        "total_automations": len(automations),
                        "successful_restorations": success_count,
                        "failed_restorations": failed_count
                    }
                }
                return json.dumps(result, indent=2)

            elif operation == "list_backups":
                backup_files = []
                if os.path.exists(cls._AUTOMATION_BACKUP_DIR):
                    for file in os.listdir(cls._AUTOMATION_BACKUP_DIR):
                        if file.endswith('.yaml'):
                            backup_name = file[:-5]  # Remove .yaml extension
                            file_path = os.path.join(cls._AUTOMATION_BACKUP_DIR, file)
                            stat = os.stat(file_path)
                            created_time = datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                            backup_files.append(f"- {backup_name} (created: {created_time})")
                
                if not backup_files:
                    result = {
                        "result": {
                            "status": "No backups found",
                            "message": "No automation backups found in the backup directory.",
                            "backup_count": 0
                        }
                    }
                else:
                    result = {
                        "result": {
                            "status": "Success",
                            "backup_count": len(backup_files),
                            "backups": '\n'.join(backup_files)
                        }
                    }
                return json.dumps(result, indent=2)

            else:
                error_result = {"error": f"Invalid backup operation '{operation}'. Valid operations: backup, restore, list_backups"}
                return json.dumps(error_result, indent=2)

        except Exception as e:
            error_result = {"error": f"Unexpected error in backup/restore operation: {str(e)}"}
            return json.dumps(error_result, indent=2)

    @staticmethod
    def _run_logic(
        action: Literal[
            "call_service", "get_state", "list_devices", "get_usage_history", "search_entity_by_keyword",
            "list_automations", "get_automation", "create_automation", "update_automation", "delete_automation",
            "enable_automation", "disable_automation", "trigger_automation", "reload_automations",
            "backup_automations", "restore_automations", "list_automation_backups"
        ],
        domain: Optional[str] = None,
        service: Optional[str] = None,
        entity_id: Optional[str] = None,
        service_data: Optional[Dict[str, Any]] = None,
        keyword: Optional[str] = None,
        automation_id: Optional[str] = None,
        automation_config: Optional[Dict[str, Any]] = None,
        backup_name: Optional[str] = None
    ) -> str:
        """
        Interacts with a Home Assistant instance for entity control and automation management.

        Args:
            action: The action to perform. Entity actions: 'call_service', 'get_state', 'list_devices', 
                    'get_usage_history', 'search_entity_by_keyword'. Automation actions: 'list_automations',
                    'get_automation', 'create_automation', 'update_automation', 'delete_automation',
                    'enable_automation', 'disable_automation', 'trigger_automation', 'reload_automations',
                    'backup_automations', 'restore_automations', 'list_automation_backups'.
            domain: (For 'call_service') The domain of the service (e.g., 'light', 'switch').
            service: (For 'call_service') The name of the service (e.g., 'turn_on', 'toggle').
            entity_id: Entity ID for various operations. For automations, use automation entity ID (e.g., 'automation.my_automation').
            service_data: (For 'call_service') Optional dictionary with extra data for the service.
            keyword: (For 'search_entity_by_keyword') The keyword to search for.
            automation_id: (For automation CRUD operations) The automation configuration ID.
            automation_config: (For 'create_automation', 'update_automation') The automation configuration dictionary.
            backup_name: (For 'backup_automations', 'restore_automations') Name of the backup.

        Returns:
            A JSON string with a 'result' key on success, or an 'error' key on failure.
        """
        try:
            # Entity-related actions
            if action == "call_service":
                if not domain or not service:
                    error_result = {"error": "For 'call_service', 'domain' and 'service' are required."}
                    return json.dumps(error_result, indent=2)
                return HomeAssistant._call_service(domain, service, entity_id, service_data)
            
            elif action == "get_state":
                if not entity_id:
                    error_result = {"error": "For 'get_state', 'entity_id' is required. To list all devices, use 'list_devices'."}
                    return json.dumps(error_result, indent=2)
                return HomeAssistant._get_state(entity_id)
                
            elif action == "list_devices":
                return HomeAssistant._get_state() # Gets all states
            
            elif action == "get_usage_history":
                if not entity_id:
                    error_result = {"error": "For 'get_usage_history', 'entity_id' is required."}
                    return json.dumps(error_result, indent=2)
                return HomeAssistant._get_entity_usage_history(entity_id)

            elif action == "search_entity_by_keyword":
                if not keyword:
                    error_result = {"error": "For 'search_entity_by_keyword', 'keyword' is required."}
                    return json.dumps(error_result, indent=2)
                return HomeAssistant._search_entities_by_keyword(keyword)

            # Automation management actions
            elif action == "list_automations":
                return HomeAssistant._manage_automations("list")

            elif action == "get_automation":
                if not automation_id:
                    error_result = {"error": "For 'get_automation', 'automation_id' is required."}
                    return json.dumps(error_result, indent=2)
                return HomeAssistant._manage_automations("get", automation_id)

            elif action == "create_automation":
                if not automation_config:
                    error_result = {"error": "For 'create_automation', 'automation_config' is required."}
                    return json.dumps(error_result, indent=2)
                return HomeAssistant._manage_automations("create", automation_config=automation_config)

            elif action == "update_automation":
                if not automation_id or not automation_config:
                    error_result = {"error": "For 'update_automation', both 'automation_id' and 'automation_config' are required."}
                    return json.dumps(error_result, indent=2)
                return HomeAssistant._manage_automations("update", automation_id, automation_config)

            elif action == "delete_automation":
                if not automation_id:
                    error_result = {"error": "For 'delete_automation', 'automation_id' is required."}
                    return json.dumps(error_result, indent=2)
                return HomeAssistant._manage_automations("delete", automation_id)

            # Automation control actions
            elif action == "enable_automation":
                if not entity_id:
                    error_result = {"error": "For 'enable_automation', 'entity_id' (automation entity ID) is required."}
                    return json.dumps(error_result, indent=2)
                return HomeAssistant._control_automation("enable", entity_id)

            elif action == "disable_automation":
                if not entity_id:
                    error_result = {"error": "For 'disable_automation', 'entity_id' (automation entity ID) is required."}
                    return json.dumps(error_result, indent=2)
                return HomeAssistant._control_automation("disable", entity_id)

            elif action == "trigger_automation":
                if not entity_id:
                    error_result = {"error": "For 'trigger_automation', 'entity_id' (automation entity ID) is required."}
                    return json.dumps(error_result, indent=2)
                return HomeAssistant._control_automation("trigger", entity_id)

            elif action == "reload_automations":
                return HomeAssistant._control_automation("reload", "")

            # Backup/restore actions
            elif action == "backup_automations":
                return HomeAssistant._backup_restore_automations("backup", backup_name)

            elif action == "restore_automations":
                if not backup_name:
                    error_result = {"error": "For 'restore_automations', 'backup_name' is required."}
                    return json.dumps(error_result, indent=2)
                return HomeAssistant._backup_restore_automations("restore", backup_name)

            elif action == "list_automation_backups":
                return HomeAssistant._backup_restore_automations("list_backups")

            else:
                valid_actions = [
                    "call_service", "get_state", "list_devices", "get_usage_history", "search_entity_by_keyword",
                    "list_automations", "get_automation", "create_automation", "update_automation", "delete_automation",
                    "enable_automation", "disable_automation", "trigger_automation", "reload_automations",
                    "backup_automations", "restore_automations", "list_automation_backups"
                ]
                error_result = {"error": f"Invalid action '{action}'. Must be one of: {', '.join(valid_actions)}."}
                return json.dumps(error_result, indent=2)

        except Exception as e:
            error_result = {"error": f"Unexpected error in HomeAssistant tool: {str(e)}"}
            return json.dumps(error_result, indent=2)


# --- Example Usage (for testing purposes) ---
# Module-level run function for CLI-Agent compatibility  
def run(action, **kwargs) -> str:
    """
    Module-level wrapper for HomeAssistant._run_logic() to maintain compatibility with CLI-Agent.
    
    Args:
        action: The action to perform on Home Assistant
        **kwargs: Additional arguments for the action
        
    Returns:
        str: JSON string with result or error
    """
    return HomeAssistant._run_logic(action=action, **kwargs)


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
        print("Example JSON response for no configuration:")
        result = HomeAssistant._run_logic(action="list_devices")
        print(result)
    else:
        print("\n--- Test Case 1: Listing all automations ---")
        try:
            result1 = HomeAssistant._run_logic(action="list_automations")
            print(f"Result:\n{result1}")
        except Exception as e:
            print(f"Test Case 1 FAILED: {e}")

        print("\n--- Test Case 2: Creating a sample automation ---")
        sample_automation = {
            "id": "test_automation_from_python",
            "alias": "Test Automation from Python",
            "description": "A test automation created via Python",
            "trigger": [{
                "platform": "time",
                "at": "18:00:00"
            }],
            "action": [{
                "service": "persistent_notification.create",
                "data": {
                    "title": "Test Automation Triggered",
                    "message": "This notification was created by a Python-generated automation!"
                }
            }],
            "mode": "single"
        }
        
        try:
            result2 = HomeAssistant._run_logic(
                action="create_automation",
                automation_config=sample_automation
            )
            print(f"Result:\n{result2}")
        except Exception as e:
            print(f"Test Case 2 FAILED: {e}")

        print("\n--- Test Case 3: Getting the created automation ---")
        try:
            result3 = HomeAssistant._run_logic(
                action="get_automation",
                automation_id="test_automation_from_python"
            )
            print(f"Result:\n{result3}")
        except Exception as e:
            print(f"Test Case 3 FAILED: {e}")

        print("\n--- Test Case 4: Backup automations ---")
        try:
            result4 = HomeAssistant._run_logic(
                action="backup_automations",
                backup_name="test_backup"
            )
            print(f"Result:\n{result4}")
        except Exception as e:
            print(f"Test Case 4 FAILED: {e}")

        print("\n--- Test Case 5: List automation backups ---")
        try:
            result5 = HomeAssistant._run_logic(action="list_automation_backups")
            print(f"Result:\n{result5}")
        except Exception as e:
            print(f"Test Case 5 FAILED: {e}")

        print("\n--- Test Case 6: Search for automation entities ---")
        try:
            result6 = HomeAssistant._run_logic(
                action="search_entity_by_keyword",
                keyword="automation"
            )
            print(f"Result:\n{result6}")
        except Exception as e:
            print(f"Test Case 6 FAILED: {e}")

        print("\n--- Test Case 7: Clean up - Delete test automation ---")
        try:
            result7 = HomeAssistant._run_logic(
                action="delete_automation",
                automation_id="test_automation_from_python"
            )
            print(f"Result:\n{result7}")
        except Exception as e:
            print(f"Test Case 7 FAILED: {e}")

    print("\n--- Finished testing ---")