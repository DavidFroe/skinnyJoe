#!/usr/bin/env python3
import sys
import json
import urllib.request
import urllib.error
import socket
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

# --- KONFIGURATION ---
LMS_SERVER_URL = "http://127.0.0.1:1234/v1"
JIT_TIMEOUT_SECONDS = 900 
MAX_RETRIES = 10  # Wir hämmern bis zu 10 Mal an die Tür!

def log(msg): 
    print(f"[PROXY] {msg}", file=sys.stderr)

class LMStudioProxyHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def _send_json(self, code, data):
        body = json.dumps(data).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Connection', 'close')
        self.end_headers()
        self.wfile.write(body)

    def _sanitize_payload(self, payload_dict):
        if "max_tokens" in payload_dict and payload_dict["max_tokens"] <= 0:
            log("WARNUNG: Illegaler 'max_tokens' Wert erkannt (<= 0). Wird entfernt!")
            del payload_dict["max_tokens"]
        return payload_dict

    def _forward_request_to_lms(self, url, payload_dict):
        safe_payload = self._sanitize_payload(payload_dict)
        data_bytes = json.dumps(safe_payload).encode('utf-8')
        
        # --- DER EISERNE STOSSDÄMPFER ---
        for attempt in range(MAX_RETRIES):
            req = urllib.request.Request(url, data=data_bytes, method='POST')
            req.add_header('Content-Type', 'application/json')
            req.add_header('Accept', 'application/json')
            req.add_header('User-Agent', 'OpenAI-Python-SDK-Mock/1.0 (LMS-Proxy)')
            req.add_header('Connection', 'close') 
            req.add_header('Content-Length', str(len(data_bytes)))

            try:
                response = urllib.request.urlopen(req, timeout=JIT_TIMEOUT_SECONDS)
                # Wenn wir hier ankommen, hat LM Studio geantwortet! Wir brechen den Loop ab und geben die Daten zurück.
                return response.getcode(), response.getheaders(), response.read()
            
            except urllib.error.HTTPError as e:
                # Echte HTTP-Fehler (z.B. Context Window Full) geben wir direkt weiter, da hilft kein Neuversuch
                err_body = e.read().decode('utf-8', errors='ignore')
                log(f"LM Studio API Error (HTTP {e.code}): {err_body}")
                raise Exception(f"LM Studio Error {e.code}: {err_body}")
            
            except (urllib.error.URLError, ConnectionResetError, socket.error) as e:
                # Hier fangen wir den Bug von LM Studio ab ("Empty reply from server" / "Connection reset")
                log(f"LM Studio hat die Verbindung gedroppt (Bug). Versuch {attempt + 1}/{MAX_RETRIES} fehlgeschlagen.")
                if attempt < MAX_RETRIES - 1:
                    log("Warte 1 Sekunde und versuche es erneut...")
                    time.sleep(1)
                    continue  # Ab in die nächste Runde!
                else:
                    log(f"Endgültiger Abbruch nach {MAX_RETRIES} Versuchen.")
                    raise Exception(f"Verbindungsabbruch nach {MAX_RETRIES} Versuchen: {e}")
            
            except TimeoutError:
                log("Timeout beim Modell-Ladevorgang.")
                raise Exception("Timeout beim Modell-Ladevorgang.")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length) if content_length else b'{}'
        
        try: 
            payload = json.loads(post_data.decode('utf-8'))
        except json.JSONDecodeError: 
            return self._send_json(400, {"error": "Invalid JSON from Agent"})

        if self.path in ['/api/v1/ask', '/api/v1/infer']:
            log(f"Übersetze {self.path} für LM Studio...")
            
            lms_payload = {
                "model": "qwen3-30b-local", 
                "messages": [
                    {"role": "system", "content": payload.get("system_prompt", "Du bist ein hilfreicher KI-Agent.")},
                    {"role": "user", "content": payload.get("prompt", "")}
                ],
                "temperature": payload.get("temperature", 0.7),
                "stream": False
            }
            
            url = f"{LMS_SERVER_URL}/chat/completions"
            try:
                status, headers, body = self._forward_request_to_lms(url, lms_payload)
                res_data = json.loads(body.decode('utf-8'))
                answer = res_data["choices"][0]["message"]["content"]
                
                if self.path == '/api/v1/ask':
                    self._send_json(200, {"response": answer})
                else:
                    self._send_json(200, {"status": "success", "response": answer})
                log("Erfolgreich an Ella geantwortet!")
            except Exception as e:
                self._send_json(500, {"error": str(e)})

        elif self.path == '/v1/chat/completions':
            log("Direkter Durchlauf /v1/chat/completions...")
            url = f"{LMS_SERVER_URL}/chat/completions"
            try:
                status, headers, body = self._forward_request_to_lms(url, payload)
                self.send_response(status)
                
                for key, value in headers:
                    if key.lower() not in ['transfer-encoding', 'content-length', 'connection', 'content-encoding']:
                        self.send_header(key, value)
                        
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Connection', 'close')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                self.wfile.write(body)
                log("Direkter Durchlauf erfolgreich beendet!")
            except Exception as e:
                self._send_json(500, {"error": str(e)})
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == '/api/v1/model/active':
            self._send_json(200, {"status": "success", "active_model": {"id": 15, "model_key": "qwen3-30b-local", "mode": "fafr"}})
        elif self.path == '/api/v1/status':
            self._send_json(200, {"status": "success"})
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == '__main__':
    log("Gepanzerter LM Studio Proxy startet...")
    server = ThreadingHTTPServer(('0.0.0.0', 5050), LMStudioProxyHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("Proxy wird beendet.")
        server.server_close()
