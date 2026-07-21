import json
import subprocess
import logging
import atexit
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger("dbert.tools.mcp_client")

class MCPClient:
    def __init__(self, command_str: str):
        self.command_str = command_str
        self.process = None
        self.id_counter = 1

    def connect(self) -> bool:
        """Starts the MCP server process and performs the initialization handshake."""
        logger.info(f"Connecting to MCP server subprocess via command: {self.command_str}")
        try:
            import shlex
            cmd_args = shlex.split(self.command_str, posix=False)
            
            self.process = subprocess.Popen(
                cmd_args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                text=True,
                bufsize=1
            )
            
            # Send initialize JSON-RPC request
            init_req = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "dbert-agent",
                        "version": "0.1.0"
                    }
                }
            }
            res = self._send_request(init_req)
            if res and "result" in res:
                # Send initialized notification confirmation
                init_notif = {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized"
                }
                self._send_notification(init_notif)
                logger.info("Successfully connected and initialized stdio MCP server link.")
                atexit.register(self.close)
                return True
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
        return False

    def _next_id(self) -> int:
        self.id_counter += 1
        return self.id_counter

    def _send_notification(self, payload: Dict[str, Any]) -> None:
        if not self.process or self.process.poll() is not None:
            return
        try:
            line = json.dumps(payload) + "\n"
            self.process.stdin.write(line)
            self.process.stdin.flush()
        except Exception as e:
            logger.error(f"Failed to send MCP notification line: {e}")

    def _send_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.process or self.process.poll() is not None:
            return {}
        try:
            req_id = payload.get("id")
            line = json.dumps(payload) + "\n"
            self.process.stdin.write(line)
            self.process.stdin.flush()
            
            # Read stdout line-by-line waiting for the matching ID response
            while True:
                resp_line = self.process.stdout.readline()
                if not resp_line:
                    break
                try:
                    res_data = json.loads(resp_line)
                    if res_data.get("id") == req_id:
                        return res_data
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            logger.error(f"Error during MCP stdio roundtrip: {e}")
        return {}

    def list_tools(self) -> List[Dict[str, Any]]:
        """Queries the MCP server for its list of exposed tools."""
        req = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/list",
            "params": {}
        }
        res = self._send_request(req)
        if res and "result" in res:
            return res["result"].get("tools", [])
        return []

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Calls a tool on the MCP server and returns the parsed text response."""
        req = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
        res = self._send_request(req)
        if not res:
            return "Error: No response from MCP server."
            
        if "error" in res:
            return f"Error from MCP server: {res['error'].get('message', res['error'])}"
            
        result = res.get("result", {})
        content_items = result.get("content", [])
        text_outputs = []
        for item in content_items:
            if item.get("type") == "text":
                text_outputs.append(item.get("text", ""))
        return "\n".join(text_outputs)

    def close(self) -> None:
        """Shuts down the subprocess gracefully."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None
