
from py_classes.cls_web_server import WebServer
from py_classes.cls_chat import Chat

# Create and configure the server
server = WebServer()
server.chat = Chat.load_from_json()  # Load the chat from the saved JSON file
app = server.app
