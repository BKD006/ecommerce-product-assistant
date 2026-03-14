from typing import Dict, Callable, Any, Optional

from src.tools.product_tool import ProductTool
from src.tools.policy_tool import PolicyTool

from src.logger import GLOBAL_LOGGER as log
from src.exception.custom_exception import ProductAssistantException


class ToolRegistry:
    """
    Central registry for all available agent tools.

    Responsibilities:
    - Tool registration
    - Tool lookup
    - Safe execution
    - Extensibility for future tools
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
        """
        Register core tools available to the agent.
        """

        # Product search + brands + categories
        self.register_tool(
            "product_tool",
            ProductTool(),
        )

        # Policy RAG
        self.register_tool(
            "policy_tool",
            PolicyTool(),
        )

    # -------------------------------------------------
    # REGISTER TOOL
    # -------------------------------------------------

    def register_tool(
        self,
        name: str,
        tool_instance: Any,
    ):

        if name in self._tools:
            log.warning(
                "Tool already registered, overriding",
                tool=name,
            )

        self._tools[name] = tool_instance

        log.info(
            "Tool registered",
            tool=name,
        )

    # -------------------------------------------------
    # GET TOOL
    # -------------------------------------------------

    def get_tool(
        self,
        name: str,
    ) -> Optional[Any]:

        return self._tools.get(name)

    # -------------------------------------------------
    # EXECUTE TOOL
    # -------------------------------------------------

    def execute(
        self,
        name: str,
        **kwargs,
    ):

        tool = self.get_tool(name)

        if not tool:
            raise ProductAssistantException(
                f"Tool '{name}' not found in registry"
            )

        try:

            log.info(
                "Executing tool",
                tool=name,
                parameters=kwargs,
            )

            result = tool.run(**kwargs)

            log.info(
                "Tool execution completed",
                tool=name,
                result_count=len(result)
                if isinstance(result, list)
                else None,
            )

            return result

        except Exception as e:

            log.error(
                "Tool execution failed",
                tool=name,
                error=str(e),
            )

            raise ProductAssistantException(
                f"Execution failed for tool '{name}'",
                e,
            )

    # -------------------------------------------------
    # LIST TOOLS
    # -------------------------------------------------

    def list_tools(self):

        return list(self._tools.keys())