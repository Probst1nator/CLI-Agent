import os
import webbrowser
import threading
import logging
import socketserver
from typing import List
from http.server import SimpleHTTPRequestHandler
import json
import time
import socket

from py_classes.cls_chat import Chat
from py_classes.globals import g
from py_classes.cls_few_shot_factory import FewShotProvider

class HtmlServer:
    def __init__(self, proj_vscode_dir_path):
        self.server_thread = None
        self.log_file = os.path.join(proj_vscode_dir_path, "visualize_context_server.log")
        self.remote_user_input = ""
        self.host_route = ""
        self.last_update_time = 0
        self.port = 8000

    def get_local_ip(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('10.255.255.255', 1))
            IP = s.getsockname()[0]
        except Exception:
            IP = '127.0.0.1'
        finally:
            s.close()
        return IP

    class DynamicHandler(SimpleHTTPRequestHandler):
        html_content = ""

        def do_GET(self):
            if self.path == '/':
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(self.html_content.encode())
            elif self.path == '/check_update':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                update_available = time.time() - self.server.html_server.last_update_time < 5
                self.wfile.write(json.dumps({'update_available': update_available}).encode())
            else:
                super().do_GET()

        def do_POST(self):
            if self.path == '/process_input':
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                user_input = data['input']
                
                self.server.html_server.remote_user_input = user_input
                processed_result = f"Processed: {user_input}"
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'result': processed_result}).encode())
            else:
                super().do_POST()

    def start_server(self, port: int = 8000) -> None:
        self.port = port
        while True:
            try:
                class Handler(self.DynamicHandler):
                    pass
                
                with socketserver.TCPServer(("", self.port), Handler) as httpd:
                    httpd.html_server = self
                    
                    local_ip = self.get_local_ip()
                    print(f"Serving locally at http://localhost:{self.port}")
                    print(f"To access from another device on the network, use http://{local_ip}:{self.port}")
                    print(f"Server logs are being written to {self.log_file}")
                    httpd.serve_forever()
            except OSError as e:
                if e.errno == 98:  # Address already in use
                    print(f"Port {self.port} is already in use. Trying the next port...")
                    self.port += 1
                else:
                    raise

    def add_input_widget_to_html(self, base_html: str) -> str:
        input_widget_html = """
        <style>
        body {
            padding: 0;
            display: flex;
            flex-direction: column;
            min-height: 100vh;
        }
        #contentWrapper {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            overflow-y: auto;
            padding-bottom: 80px;
        }
        #mainContent {
            width: 100%;
            padding: 20px;
            padding-bottom: 30px;
            box-sizing: border-box;
        }
        #inputWidgetContainer {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: #f0f0f0;
            padding: 10px 0;
            z-index: 1000;
            box-shadow: rgba(0, 0, 0, 0.5) 0px -10px 20px;
        }
        #inputWidget {
            width: 90%;
            max-width: 800px;
            margin: 0 auto;
            background-color: white;
            border-radius: 12px;
            padding: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
            display: flex;
            align-items: center;
        }
        #userInput {
            flex-grow: 1;
            height: 36px;
            border: 1px solid #ccc;
            background-color: white;
            border-radius: 8px;
            padding: 8px 12px;
            font-size: 14px;
            outline: none;
            transition: border-color 0.3s ease;
        }
        #userInput:focus {
            border-color: #007bff;
        }
        #submitButton {
            width: 50px;
            height: 36px;
            margin-left: 8px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            transition: background-color 0.3s ease;
        }
        #submitButton:hover {
            background-color: #0056b3;
        }
        </style>
        <div id="inputWidgetContainer">
            <div id="inputWidget">
                <textarea id="userInput" placeholder="Type your message..."></textarea>
                <button id="submitButton">Send</button>
            </div>
        </div>
        <script>
        function submitInput() {
            var input = document.getElementById('userInput').value;
            fetch('/process_input', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({input: input}),
            })
            .then(response => response.json())
            .then(data => {
                console.log('Success:', data);
                // Handle the response here (e.g., update the page content)
            })
            .catch((error) => {
                console.error('Error:', error);
            });
            document.getElementById('userInput').value = '';
        }
        document.getElementById('submitButton').addEventListener('click', submitInput);
        document.getElementById('userInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                submitInput();
            }
        });

        // Wrap the existing content in a centered div
        document.addEventListener('DOMContentLoaded', function() {
            var body = document.body;
            var contentWrapper = document.createElement('div');
            contentWrapper.id = 'contentWrapper';
            var mainContent = document.createElement('div');
            mainContent.id = 'mainContent';
            while (body.firstChild) {
                if (body.firstChild.id !== 'inputWidgetContainer') {
                    mainContent.appendChild(body.firstChild);
                } else {
                    break;
                }
            }
            contentWrapper.appendChild(mainContent);
            body.insertBefore(contentWrapper, body.firstChild);
        });

        // Check for updates periodically
        setInterval(function() {
            fetch('/check_update')
                .then(response => response.json())
                .then(data => {
                    if (data.update_available) {
                        location.reload();
                    }
                })
                .catch(error => console.error('Error:', error));
        }, 5000);  // Check every 5 seconds
        </script>
        """
        return base_html.replace('</body>', f'{input_widget_html}</body>')

    def visualize_context(self, context_chat: 'Chat', preferred_models: List[str] = [], force_local: bool = False) -> None:
        base_html, chat = FewShotProvider.few_shot_GenerateHtmlPage(context_chat.get_messages_as_string(-3), preferred_models=preferred_models, force_local=force_local)
        html = self.add_input_widget_to_html(base_html)
        
        host_over_network: bool = True # currently can not receive input when run locally
        if host_over_network:
            logging.basicConfig(filename=self.log_file, level=logging.INFO,
                                format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
            self.DynamicHandler.html_content = html
            self.last_update_time = time.time()
            
            if self.server_thread is None or not self.server_thread.is_alive():
                self.server_thread = threading.Thread(target=self.start_server, daemon=True)
                self.server_thread.start()
                time.sleep(1)  # Give the server a moment to start
            self.host_route = f"{self.get_local_ip()}:{self.port}"
            webbrowser.open(self.host_route)
        else:
            file_path = os.path.join(os.path.dirname(self.log_file), "tmp_context_visualization.html")
            with open(file_path, "w") as file:
                file.write(html)
            self.host_route = 'file://' + os.path.realpath(file_path)
            webbrowser.open(self.host_route)

# Usage example
if __name__ == "__main__":
    visualizer = HtmlServer(g.PROJ_VSCODE_DIR_PATH)
    context_chat = Chat()  # Assume Chat is defined elsewhere
    visualizer.visualize_context(context_chat, host_over_network=True)