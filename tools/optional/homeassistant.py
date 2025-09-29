# tools/homeassistant.py
"""
This file implements the 'homeassistant' tool. It allows the agent to interact
with a Home Assistant server to control entities, read states, and manage automations.
It acts as a user-friendly wrapper around a powerful Home Assistant API client.
"""

import os
import requests
import json
import re
import logging
from typing import Dict, Optional, Any
from dotenv import load_dotenv
from datetime import datetime

# Assumes g is initialized and available from the project's core
from core.globals import g

# --- Helper Class: The Core API Logic ---
# This class contains the original, powerful logic from the utility file.
# It handles all direct communication with the Home Assistant server.
class _HomeAssistantAPI:
    """
    A utility for interacting with a Home Assistant server.
    This tool allows for calling services, retrieving the state of entities,
    searching for entities, and managing automations.
    Returns structured JSON responses for reliable communication.
    """
    _INTERACTED_ENTITIES_FILE = os.path.join(g.CLIAGENT_PERSISTENT_STORAGE_PATH, 'homeassistant_interacted_entities.json')
    _AUTOMATION_BACKUP_DIR = os.path.join(g.CLIAGENT_PERSISTENT_STORAGE_PATH, 'homeassistant_automation_backups')

    @classmethod
    def _load_interacted_entities(cls) -> Dict[str, list]:
        try:
            if os.path.exists(cls._INTERACTED_ENTITIES_FILE):
                with open(cls._INTERACTED_ENTITIES_FILE, 'r') as f:
                    return json.load(f).get('entities', {})
        except Exception:
            pass
        return {}

    @classmethod
    def _save_interacted_entities(cls, interacted_entities: Dict[str, list]) -> None:
        try:
            os.makedirs(os.path.dirname(cls._INTERACTED_ENTITIES_FILE), exist_ok=True)
            # Trim history to the latest 50 entries per entity
            trimmed_entities = {entity_id: sorted(history, key=lambda x: x.get('timestamp', ''), reverse=True)[:50] for entity_id, history in interacted_entities.items()}
            data = {'entities': trimmed_entities, 'file_last_updated': datetime.now().isoformat()}
            with open(cls._INTERACTED_ENTITIES_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass # Fail silently if saving fails

    @classmethod
    def _track_entity_interaction(cls, entity_id: str, interaction_type: str, details: str) -> None:
        if not entity_id: return
        interacted_entities = cls._load_interacted_entities()
        history_entry = {"timestamp": datetime.now().isoformat(), "action": interaction_type, "details": details}
        if entity_id not in interacted_entities:
            interacted_entities[entity_id] = []
        interacted_entities[entity_id].insert(0, history_entry)
        cls._save_interacted_entities(interacted_entities)

    @staticmethod
    def _get_config() -> tuple[str, str]:
        load_dotenv()
        hass_url = os.environ.get("HASS_URL")
        hass_token = os.environ.get("HASS_TOKEN")
        if not hass_url or not hass_token:
            raise ValueError("HASS_URL and HASS_TOKEN environment variables must be set.")
        return hass_url, hass_token

    @classmethod
    def check_reachability(cls) -> tuple[bool, str]:
        hass_url = ""
        try:
            hass_url, hass_token = cls._get_config()
            # Attempt to hit a simple, unauthenticated endpoint or the /api/ endpoint
            # The /api/ endpoint requires authentication but will return 200 if reachable and authenticated
            # or 401 if reachable but unauthenticated. Either way, it indicates reachability.
            api_url = f"{hass_url.rstrip('/')}/api/"
            headers = {"Authorization": f"Bearer {hass_token}", "Content-Type": "application/json"}
            response = requests.get(api_url, headers=headers, timeout=5)
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            return True, "Home Assistant is reachable and authenticated."
        except ValueError as e:
            return False, f"Configuration Error: {e}. Please set HASS_URL and HASS_TOKEN environment variables."
        except requests.exceptions.ConnectionError:
            return False, f"Connection Error: Could not connect to Home Assistant at {hass_url}. Check URL and network."
        except requests.exceptions.Timeout:
            return False, f"Timeout Error: Connection to Home Assistant at {hass_url} timed out."
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                return False, f"Authentication Error: Home Assistant at {hass_url} returned 401 Unauthorized. Check HASS_TOKEN."
            return False, f"HTTP Error: Home Assistant at {hass_url} returned {e.response.status_code}. Details: {e.response.text}"
        except Exception as e:
            return False, f"An unexpected error occurred during reachability check: {str(e)}"

    @classmethod
    def _make_request(cls, method: str, endpoint: str, payload: Optional[Dict] = None) -> requests.Response:
        hass_url, hass_token = cls._get_config()
        api_url = f"{hass_url.rstrip('/')}/api/{endpoint}"
        headers = {"Authorization": f"Bearer {hass_token}", "Content-Type": "application/json"}
        response = requests.request(method, api_url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        return response

    @classmethod
    def call_service(cls, domain: str, service: str, entity_id: Optional[str], service_data: Optional[Dict]) -> Dict:
        payload = service_data or {}
        if entity_id: payload['entity_id'] = entity_id
        response = cls._make_request("POST", f"services/{domain}/{service}", payload)

        if entity_id:
            details = f"Called service {domain}.{service}"
            if service_data: details += f" with data: {json.dumps(service_data)}"
            cls._track_entity_interaction(entity_id, "service_call", details)

        return {"status": "Success", "message": f"Service {domain}.{service} called successfully.", "response": response.json()}

    @classmethod
    def get_state(cls, entity_id: str) -> Dict:
        response = cls._make_request("GET", f"states/{entity_id}")
        cls._track_entity_interaction(entity_id, "get_state", "Queried entity state")
        return response.json()

    @classmethod
    def list_devices(cls) -> list:
        response = cls._make_request("GET", "states")
        states = response.json()
        interacted = cls._load_interacted_entities()
        # Prioritize previously interacted entities
        states.sort(key=lambda s: s.get('entity_id') in interacted, reverse=True)
        return states

    @classmethod
    def search(cls, keyword: str) -> list:
        response = cls._make_request("GET", "states")
        all_states = response.json()
        lower_keyword = keyword.lower()
        matches = [
            s for s in all_states
            if lower_keyword in s.get('entity_id', '').lower() or
               lower_keyword in s.get('attributes', {}).get('friendly_name', '').lower()
        ]
        return matches

# --- Tool Class: The Agent's Interface ---
# This class provides the standard tool contract and translates between
# the agent's XML requests and the API helper class.
class homeassistant:
    _is_enabled: bool = False
    _status_message: str = "Not initialized."

    @classmethod
    def initialize(cls) -> None:
        """Initialize the Home Assistant tool by testing connectivity."""
        # Skip if already initialized to prevent duplicate logging
        if hasattr(cls, '_initialized') and cls._initialized:
            return

        is_reachable, message = _HomeAssistantAPI.check_reachability()
        cls._is_enabled = is_reachable
        cls._status_message = message
        cls._initialized = True

        if not is_reachable:
            logging.warning(f"Home Assistant tool disabled: {message}")
        else:
            logging.info(f"Home Assistant tool enabled: {message}")

    @staticmethod
    def get_delim() -> str:
        return 'homeassistant'

    @classmethod
    def get_tool_info(cls) -> dict:
        if not cls._is_enabled:
            return {
                "name": "homeassistant",
                "description": f"Home Assistant tool is currently disabled: {cls._status_message}",
                "example": "Home Assistant is not available."
            }
        return {
            "name": "homeassistant",
            "description": "Interacts with a Home Assistant server to control devices, read sensor states, and search for entities. Requires an <action> tag to specify the operation.",
            "example": (
                "<!-- Simple On/Off -->\n"
                "<homeassistant><action>call_service</action><domain>light</domain><service>turn_on</service><entity_id>light.living_room</entity_id></homeassistant>\n\n"
                "<!-- Get Sensor State -->\n"
                "<homeassistant><action>get_state</action><entity_id>sensor.bedroom_temperature</entity_id></homeassistant>\n\n"
                "<!-- List All Devices -->\n"
                "<homeassistant><action>list_devices</action></homeassistant>\n\n"
                "<!-- Search for Devices -->\n"
                "<homeassistant><action>search</action><keyword>living room</keyword></homeassistant>\n\n"
                "<!-- Complex Service Call with Data -->\n"
                "<homeassistant><action>call_service</action><domain>light</domain><service>turn_on</service><entity_id>light.office</entity_id><service_data>{\"brightness\": 128, \"rgb_color\": [255, 100, 100]}</service_data></homeassistant>"
            )
        }

    @staticmethod
    def _parse_args(content: str) -> Dict[str, Any]:
        """Parses XML-like tags into a dictionary."""
        args = {}
        tags = ['action', 'domain', 'service', 'entity_id', 'keyword', 'service_data']
        for tag in tags:
            match = re.search(f'<{tag}>(.*?)</{tag}>', content, re.DOTALL)
            if match:
                value = match.group(1).strip()
                if tag == 'service_data':
                    try:
                        args[tag] = json.loads(value)
                    except json.JSONDecodeError:
                        raise ValueError(f"Invalid JSON in <service_data>: {value}")
                else:
                    args[tag] = value
        return args

    @staticmethod
    def run(content: str) -> str:
        """
        Parses XML content to determine the action and parameters,
        executes the command, and returns a formatted, user-friendly response.
        """
        try:
            args = homeassistant._parse_args(content)
            action = args.pop('action', None)

            if not action:
                return "Error: No <action> specified. Must be one of: call_service, get_state, list_devices, search."

            # --- Action Router ---
            if action == 'call_service':
                result = _HomeAssistantAPI.call_service(**args)
                return f"Success: {result['message']}"

            elif action == 'get_state':
                entity_id = args.get('entity_id')
                if not entity_id: return "Error: <entity_id> is required for get_state action."
                state_data = _HomeAssistantAPI.get_state(entity_id)
                state = state_data.get('state')
                attributes = state_data.get('attributes', {})
                friendly_name = attributes.get('friendly_name', entity_id)
                response = f"State for '{friendly_name}' ({entity_id}): {state}"
                # Add common attributes for more context
                if 'temperature' in attributes: response += f", Temperature: {attributes['temperature']}{attributes.get('unit_of_measurement', '')}"
                if 'brightness' in attributes: response += f", Brightness: {attributes['brightness']}"
                if 'humidity' in attributes: response += f", Humidity: {attributes['humidity']}%"
                return response

            elif action == 'list_devices':
                devices = _HomeAssistantAPI.list_devices()
                if not devices: return "No devices found."
                # Show a summarized list to avoid context overflow
                summary = [f"Found {len(devices)} devices. Showing up to 30 most relevant:"]
                for device in devices[:30]:
                    entity_id = device.get('entity_id')
                    friendly_name = device.get('attributes', {}).get('friendly_name', 'N/A')
                    state = device.get('state')
                    summary.append(f"- {entity_id} (Name: {friendly_name}, State: {state})")
                if len(devices) > 30:
                    summary.append(f"...and {len(devices) - 30} more.")
                return "\n".join(summary)

            elif action == 'search':
                keyword = args.get('keyword')
                if not keyword: return "Error: <keyword> is required for search action."
                matches = _HomeAssistantAPI.search(keyword)
                if not matches: return f"No devices found matching keyword: '{keyword}'"
                summary = [f"Found {len(matches)} devices matching '{keyword}':"]
                for device in matches:
                    entity_id = device.get('entity_id')
                    friendly_name = device.get('attributes', {}).get('friendly_name', 'N/A')
                    state = device.get('state')
                    summary.append(f"- {entity_id} (Name: {friendly_name}, State: {state})")
                return "\n".join(summary)

            else:
                return f"Error: Unknown action '{action}'. Supported actions: call_service, get_state, list_devices, search."

        except (ValueError, KeyError) as e:
            return f"Error: Invalid input or missing parameter. Details: {str(e)}"
        except requests.exceptions.HTTPError as e:
            return f"Error: Home Assistant API request failed. Status: {e.response.status_code}. Details: {e.response.text}"
        except requests.exceptions.RequestException as e:
            return f"Error: Could not connect to Home Assistant. Check network and HASS_URL. Details: {str(e)}"
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"