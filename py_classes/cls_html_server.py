import os
import webbrowser
import threading
import logging
import socketserver
from typing import List
from http.server import SimpleHTTPRequestHandler
import json
import socketserver
import webbrowser
import os
from typing import List
import threading
import logging

from py_classes.cls_chat import Chat
from py_classes.globals import g
from py_classes.cls_few_shot_factory import FewShotProvider


class HtmlServer:
    def __init__(self, proj_vscode_dir_path):
        self.server_thread = None
        self.log_file = os.path.join(proj_vscode_dir_path, "visualize_context_server.log")
        self.remote_user_input = ""

    def get_local_ip(self):
        import socket
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
            else:
                super().do_GET()

        def do_POST(self):
            if self.path == '/process_input':
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                user_input = data['input']
                
                processed_result = self.server.visualizer.process_user_input(user_input)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'result': processed_result}).encode())
            else:
                super().do_GET()

    def start_server(self, port: int = 8000) -> None:
        class Handler(self.DynamicHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.server.visualizer = self
        
        with socketserver.TCPServer(("", port), Handler) as httpd:
            local_ip = self.get_local_ip()
            print(f"Serving locally at http://localhost:{port}")
            print(f"To access from another device on the network, use http://{local_ip}:{port}")
            print(f"Server logs are being written to {self.log_file}")
            httpd.serve_forever()

    def process_user_input(self, input_string: str) -> str:
        self.remote_user_input = input_string
        return f"Processed: {input_string}"

    def generate_html_with_input_widget(self, base_html: str) -> str:
        input_widget_html = """
        <div style="position: fixed; bottom: 0; left: 0; right: 0; background-color: white; padding: 10px;">
            <textarea id="userInput" style="width: calc(100% - 70px); height: 50px; resize: vertical;"></textarea>
            <button id="submitButton" style="width: 60px; height: 56px; vertical-align: top;">Enter</button>
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
        </script>
        """
        return base_html.replace('</body>', f'{input_widget_html}</body>')

    def visualize_context(self, context_chat: 'Chat', preferred_models: List[str] = [], force_local: bool = False, host_over_network: bool = False) -> None:
        base_html, chat = FewShotProvider.few_shot_GenerateHtmlPage(context_chat.get_messages_as_string(-3), preferred_models=preferred_models, force_local=force_local)
        html = self.generate_html_with_input_widget(base_html)

        if host_over_network:
            # Set up logging
            logging.basicConfig(filename=self.log_file, level=logging.INFO,
                                format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
            
            # Update the HTML content in the handler
            self.DynamicHandler.html_content = html
            
            # Start the server if it's not already running
            if self.server_thread is None or not self.server_thread.is_alive():
                self.server_thread = threading.Thread(target=self.start_server, daemon=True)
                self.server_thread.start()
            
            # Open the browser
            webbrowser.open('http://localhost:8000')
        else:
            # Create a temporary file to store the HTML content
            file_path = os.path.join(os.path.dirname(self.log_file), "tmp_context_visualization.html")
            with open(file_path, "w") as file:
                file.write(html)
            # Open the temporary file in the default web browser
            webbrowser.open('file://' + os.path.realpath(file_path))

# Usage example
if __name__ == "__main__":
    visualizer = HtmlServer(g.PROJ_VSCODE_DIR_PATH)
    context_chat = Chat()  # Assume Chat is defined elsewhere
    visualizer.visualize_context(context_chat, host_over_network=True)