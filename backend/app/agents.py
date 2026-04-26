from openai import OpenAI
import os
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession

load_dotenv()
from app.tools import execute_tool
client= OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


_PAGE_PATH_DESC = (
    "Wiki path for this agent, e.g. notes/topic or topic (lowercase, `/` separators, "
    "no `.` or `..` segments; trailing `.md` on the last segment is optional and stripped)."
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "read_page",
            "description": (
                "Load the full body of an active page. Soft-deleted pages are not visible here—"
                "use get_page_metadata or page_exists first if unsure."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "page_path": {
                        "type": "string",
                        "description": _PAGE_PATH_DESC,
                    },
                },
                "required": ["page_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_pages",
            "description": "Return sorted paths of all active (non-deleted) pages for this agent.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_page",
            "description": (
                "Create or replace the entire page body and append a version snapshot. "
                "This is a full replace, not a patch. If the page was soft-deleted, it becomes active again."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "page_path": {
                        "type": "string",
                        "description": _PAGE_PATH_DESC,
                    },
                    "content": {
                        "type": "string",
                        "description": "Full new page content (replaces existing body).",
                    },
                },
                "required": ["page_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_page",
            "description": (
                "Soft-delete a page (hidden from list_pages and read_page). "
                "Cannot delete reserved paths: index, log, schema."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "page_path": {
                        "type": "string",
                        "description": _PAGE_PATH_DESC,
                    },
                },
                "required": ["page_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_pages",
            "description": (
                "Find pages whose body contains the query (case-insensitive substring over content, not titles-only). "
                "Returns matching paths only; use read_page for full text. Whitespace-only query returns no results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": "Substring to match inside page bodies.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max paths to return (default 100; server caps at 100).",
                    },
                },
                "required": ["search_query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "append_log_entry",
            "description": (
                "Append one entry to the reserved `log` page (creates it if missing). "
                "Prefer this over write_page for logs so history stays append-style."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "log_entry": {
                        "type": "string",
                        "description": "Non-empty text to append to the log page.",
                    },
                },
                "required": ["log_entry"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_page_metadata",
            "description": (
                "Return path, description, is_active, timestamps, and content_length without loading the full body. "
                "Works even when the page is soft-deleted."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "page_path": {
                        "type": "string",
                        "description": _PAGE_PATH_DESC,
                    },
                },
                "required": ["page_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "page_exists",
            "description": (
                "Check whether a page row exists. Soft-deleted pages still exist with is_active=false."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "page_path": {
                        "type": "string",
                        "description": _PAGE_PATH_DESC,
                    },
                },
                "required": ["page_path"],
            },
        },
    },
]
async def run_agent_loop(agent_id: int, query: str, tools: list, messages: list, db: AsyncSession):
    messages.append({"role": "user", "content": query})
    response = await client.responses.create(
        model="gpt-5.4-nano-2026-03-17", 
        max_tokens=4096,
        tools=tools, 
        input=messages, 
        stream=True,
        reasoning={"effort": "low"}
    )
    for item in response.items:
        if item.type == "text":
            messages.append({"role": "assistant", "content": item.text.value})
        elif item.type == "tool_use":
            tool_name = item.tool_use.function.name
            tool_args = item.tool_use.function.arguments
            result = await execute_tool(tool_name, tool_args, agent_id, db)
            messages.append({"role": "assistant", "content": result})
    return messages
