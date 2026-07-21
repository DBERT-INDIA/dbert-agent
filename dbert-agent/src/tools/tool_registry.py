import logging
from typing import Callable, Dict, List, Any

logger = logging.getLogger("dbert.tools.tool_registry")

class ToolRegistry:
    def __init__(self):
        self.handlers: Dict[str, Callable] = {}
        self.schemas: Dict[str, Dict[str, Any]] = {}

    def register_tool(self, name: str, handler: Callable, description: str, parameters: Dict[str, Any]) -> None:
        """
        Registers a tool handler and its schema.
        Parameters schema should follow the JSON Schema object format.
        """
        self.handlers[name] = handler
        self.schemas[name] = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters
            }
        }
        logger.info(f"Registered tool: {name}")

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Returns the list of LiteLLM/OpenAI-compatible tool schemas."""
        return list(self.schemas.values())

    def execute_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Executes the handler associated with the tool name and returns string output."""
        if name not in self.handlers:
            logger.error(f"Tool execution requested for unregistered tool: {name}")
            return f"Error: Tool '{name}' is not registered."
            
        logger.info(f"Executing tool '{name}' with arguments: {arguments}")
        try:
            handler = self.handlers[name]
            result = handler(**arguments)
            return str(result)
        except Exception as e:
            logger.error(f"Error executing tool '{name}': {e}")
            return f"Error executing tool '{name}': {e}"

# Global registry instance
global_registry = ToolRegistry()

# ----------------- Standard Pre-registered Tools -----------------

def web_search_handler(query: str) -> str:
    """Wrapper that invokes DuckDuckGo HTML scraping search."""
    try:
        from src.research.web_search import search
        results = search(query)
        if not results:
            return "No web search results found."
        
        output = []
        for idx, item in enumerate(results):
            output.append(f"[{idx+1}] Title: {item['title']}\nLink: {item['url']}\nSnippet: {item['snippet']}\n")
        return "\n".join(output)
    except Exception as e:
        return f"Failed to execute web search: {e}"

# Register the default web search tool
global_registry.register_tool(
    name="web_search",
    handler=web_search_handler,
    description="Search the web for up-to-date info, news, or reference material.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query term."
            }
        },
        "required": ["query"]
    }
)
