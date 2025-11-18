#!/usr/bin/env python3
"""
Simple HTTP server for testing Agent Management frontend
Usage: python3 serve.py [port]
Default port: 8000
"""

import http.server
import os
import socketserver
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000

# Change to frontend directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers for API access
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        super().end_headers()

    def log_message(self, format, *args):
        # Custom logging format
        print(f"[{self.log_date_time_string()}] {format % args}")

with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
    print("")
    print("╔═══════════════════════════════════════════════════════╗")
    print("║  Agent Management SaaS Frontend Server               ║")
    print("╠═══════════════════════════════════════════════════════╣")
    print("║                                                       ║")
    print(f"║  URL: http://localhost:{PORT}                         ║")
    print("║                                                       ║")
    print("║  Press Ctrl+C to stop                                 ║")
    print("║                                                       ║")
    print("╚═══════════════════════════════════════════════════════╝")
    print("")
    print(f"[INFO] Serving frontend from: {os.getcwd()}")
    print("[INFO] API calls will go to /api/agents/*")
    print("")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Server stopped")
        sys.exit(0)
