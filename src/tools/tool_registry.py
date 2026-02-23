from typing import Dict, Callable, Any, Optional

from src.tools.product_tool import ProductTool
from src.tools.policy_tool import PolicyTool
from src.logger import GLOBAL_LOGGER as log
from src.exception.custom_exception import ProductAssistantException


class ToolRegistry:
    """
    Central registry for all available agent tools.

    Provides:
    - Tool lookup by name
    - Safe execution wrapper
    - Future extensibility
    """

    def __init__(self):
        try:
            log.info("Initializing ToolRegistry")

            self._tools: Dict[str, Callable] = {}

            # Register default tools
            self._register_default_tools()

            log.info(
                "ToolRegistry initialized",
                available_tools=list(self._tools.keys()),
            )

        except Exception as e:
            raise ProductAssistantException(
                "Failed initializing ToolRegistry", e
            )

    # -------------------------------------------------
    # DEFAULT TOOL REGISTRATION
    # -------------------------------------------------

    def _register_default_tools(self):

        self.register_tool("product_tool", ProductTool())
        self.register_tool("policy_tool", PolicyTool())

    # -------------------------------------------------
    # REGISTER NEW TOOL
    # -------------------------------------------------

    def register_tool(self, name: str, tool_instance: Any):

        if name in self._tools:
            log.warning("Tool already registered, overriding", tool=name)

        self._tools[name] = tool_instance

        log.info("Tool registered", tool=name)

    # -------------------------------------------------
    # GET TOOL
    # -------------------------------------------------

    def get_tool(self, name: str) -> Optional[Any]:
        return self._tools.get(name)

    # -------------------------------------------------
    # EXECUTE TOOL SAFELY
    # -------------------------------------------------

    def execute(
        self,
        name: str,
        **kwargs,
    ):
        """
        Executes tool safely and returns result.
        """

        tool = self.get_tool(name)

        if not tool:
            raise ProductAssistantException(
                f"Tool '{name}' not found in registry"
            )

        log.info("Executing tool from registry", tool=name)

        return tool.run(**kwargs)

    # -------------------------------------------------
    # LIST TOOLS
    # -------------------------------------------------

    def list_tools(self):
        return list(self._tools.keys())