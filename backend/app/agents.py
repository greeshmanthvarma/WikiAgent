from collections.abc import AsyncGenerator
from openai import AsyncOpenAI, APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
import os
import asyncio
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession
import json
import time
load_dotenv()
from app.tools import execute_tool

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


_PAGE_PATH_DESC = (
    "Wiki path for this agent, e.g. notes/topic or topic (lowercase, `/` separators, "
    "no `.` or `..` segments; trailing `.md` on the last segment is optional and stripped)."
)
MAX_STEPS = 10
TIMEOUT = 240
TOOL_EXECUTION_TIMEOUT = 10
MAX_RETRIES = 3

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

def _event_type(event: object) -> str:
    if isinstance(event, dict):
        return str(event.get("type") or "")
    return str(getattr(event, "type", "") or "")


def _event_item(event: object) -> dict | object:
    if isinstance(event, dict):
        return event["item"]
    return getattr(event, "item")


async def run_agent_loop(
    agent_id: int,
    query: str,
    messages: list,
    db: AsyncSession,
    *,
    tool_definitions: list | None = None,
) -> AsyncGenerator[str | dict[str, object], None]:
    
    defs = tools if tool_definitions is None else tool_definitions
    input_list = messages + [{"role": "user", "content": query}]
    tool_rounds = 0
    t0 = time.time()

    while True:

        if time.time() - t0 > TIMEOUT:
            raise TimeoutError("Agent loop exceeded TIMEOUT")
        
        if MAX_STEPS > 0 and tool_rounds >= MAX_STEPS:
            stream = None
            for attempt in range(MAX_RETRIES):
                try:
                    stream = await client.responses.create(
                        model="gpt-5.4-nano-2026-03-17",
                        max_tokens=4096,
                        input=input_list
                        + [
                            {
                                "role": "user",
                                "content": (
                                    "Use the tool results and messages above to answer the original request. "
                                    "Do not ask for more tool calls."
                                ),
                            },
                        ],
                        tool_choice="none",
                        stream=True,
                        reasoning={"effort": "low"},
                    )
                    break
                except (APITimeoutError, InternalServerError, RateLimitError, APIConnectionError) as e:
                    if attempt < MAX_RETRIES - 1:
                        yield {
                            "type": "api_retry",
                            "code": getattr(e, "status_code", None),
                            "message": str(e),
                            "attempt": attempt + 1,
                            "max_attempts": MAX_RETRIES,
                        }
                        await asyncio.sleep(2**attempt)
                    else:
                        raise
            closing_text = ""
            async for event in stream:
                event_type = _event_type(event)
                if event_type == "response.output_text.delta":
                    delta = event["delta"] if isinstance(event, dict) else event.delta
                    if isinstance(delta, str) and delta:
                        closing_text += delta
                        yield delta
                elif event_type == "response.output_text.done":
                    final = event.get("text") if isinstance(event, dict) else getattr(event, "text", None)
                    if isinstance(final, str):
                        closing_text = final
                    if closing_text.strip():
                        input_list.append({"role": "assistant", "content": closing_text})
            return

        current_output_text = ""
        tools_used_this_turn = False
        
        stream = None
        for attempt in range(MAX_RETRIES):
            try:
                stream = await client.responses.create(
                    model="gpt-5.4-nano-2026-03-17",
                    max_tokens=4096,
                    tools=defs,
                    input=input_list,
                    stream=True,
                    reasoning={"effort": "low"},
                )
                break
            except (APITimeoutError, InternalServerError, RateLimitError, APIConnectionError) as e:
                if attempt < MAX_RETRIES - 1:
                    yield {
                        "type": "api_retry",
                        "code": getattr(e, "status_code", None),
                        "message": str(e),
                        "attempt": attempt + 1,
                        "max_attempts": MAX_RETRIES,
                    }
                    await asyncio.sleep(2**attempt)
                else:
                    raise

        async for event in stream:
            event_type = _event_type(event)

            if event_type == "response.output_text.delta":
                delta = event["delta"] if isinstance(event, dict) else event.delta
                if isinstance(delta, str) and delta:
                    current_output_text += delta
                    yield delta

            elif event_type == "response.output_text.done":
                final = event.get("text") if isinstance(event, dict) else getattr(event, "text", None)
                if isinstance(final, str):
                    current_output_text = final
                if current_output_text.strip():
                    input_list.append({"role": "assistant", "content": current_output_text})

            elif event_type == "response.output_item.done":
                item = _event_item(event)
                itype = item["type"] if isinstance(item, dict) else getattr(item, "type", None)
                if itype != "function_call":
                    continue
                tool_name = item["name"] if isinstance(item, dict) else item.name
                raw_args = item["arguments"] if isinstance(item, dict) else item.arguments
                call_id = item["call_id"] if isinstance(item, dict) else item.call_id

                if isinstance(raw_args, str):
                    args_str = raw_args
                    try:
                        tool_args = json.loads(args_str)
                    except json.JSONDecodeError:
                        input_list.append(
                            {
                                "type": "function_call",
                                "name": tool_name,
                                "call_id": call_id,
                                "arguments": args_str,
                            }
                        )
                        input_list.append(
                            {
                                "type": "function_call_output",
                                "call_id": call_id,
                                "output": json.dumps(
                                    {"error": "invalid_tool_arguments_json", "raw": args_str[:500]}
                                ),
                            }
                        )
                        tools_used_this_turn = True
                        continue
                else:
                    tool_args = raw_args
                    args_str = json.dumps(raw_args)

                input_list.append(
                    {
                        "type": "function_call",
                        "name": tool_name,
                        "call_id": call_id,
                        "arguments": args_str,
                    }
                )

                try:
                    result = await asyncio.wait_for(
                        execute_tool(tool_name, tool_args, agent_id=agent_id, db=db),
                        timeout=TOOL_EXECUTION_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    input_list.append(
                        {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": json.dumps({"error": "tool_execution_timeout"}),
                        }
                    )
                    tools_used_this_turn = True
                    continue

                out = json.dumps(result)
                input_list.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": out,
                    }
                )
                if "result" in result:
                    tool_rounds += 1
                tools_used_this_turn = True

        if not tools_used_this_turn:
            return

