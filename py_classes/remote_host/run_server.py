#!/usr/bin/env python
"""
Run the CLI-Agent Remote Host server.

This script provides a command-line interface for starting the remote host server.
"""

import os
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the CLI-Agent Remote Host server")
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=int(os.environ.get("PORT", 5000)),
        help="Port to run the server on (default: 5000 or PORT env var)"
    )
    
    parser.add_argument(
        "--host", 
        type=str, 
        default=os.environ.get("HOST", "0.0.0.0"),
        help="Host to bind the server to (default: 0.0.0.0 or HOST env var)"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Run the server in debug mode"
    )
    
    parser.add_argument(
        "--vosk-model-path", 
        type=str,
        help="Path to the Vosk model directory (for wake word detection)"
    )
    
    return parser.parse_args()

def main():
    """Run the server."""
    # Parse arguments
    args = parse_args()
    
    # Set environment variables based on arguments
    if args.vosk_model_path:
        os.environ["VOSK_MODEL_PATH"] = args.vosk_model_path
        
    # Import the server module (do this after setting env vars)
    from py_classes.remote_host.cls_remote_host_server import app
    
    # Run the server
    logger.info(f"Starting CLI-Agent Remote Host server on {args.host}:{args.port}")
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )

if __name__ == "__main__":
    main() 