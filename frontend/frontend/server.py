"""
Simple HTTP Server to serve the frontend
Run this to start the web server on http://localhost:8000
"""

import http.server
import socketserver
import os
from pathlib import Path

# Change to html directory where frontend files are located
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'html'))

PORT = 8000

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        return super().end_headers()

    def do_GET(self):
        if self.path == '/':
            self.path = '/ht.html'
        return super().do_GET()

Handler = MyHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"🚀 Server running at http://localhost:{PORT}")
    print(f"Press Ctrl+C to stop the server")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n✅ Server stopped")