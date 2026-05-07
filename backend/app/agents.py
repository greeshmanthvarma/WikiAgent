from collections.abc import AsyncGenerator
from openai import AsyncOpenAI, APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
import os
import asyncio
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession
import json
import time
import logging
load_dotenv()
from app.tools import execute_tool

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

logger = logging.getLogger(__name__)


_PAGE_PATH_DESC = (
    "Wiki path for this agent, e.g. notes/topic or topic (lowercase, `/` separators, "
    "no `.` or `..` segments; trailing `.md` on the last segment is optional and stripped)."
)
MAX_STEPS = 10
TIMEOUT = 240
TOOL_EXECUTION_TIMEOUT = 10
TURN_API_TIMEOUT = 120
MAX_RETRIES = 3

MAIN_WIKI_AGENT_INSTRUCTIONS = """
You are the wiki agent for a single user's personal knowledge base. All tools are scoped server-side to that wiki; you do not need to pass agent or user ids.

Your job:
- Answer questions using the wiki when tools help; prefer facts from stored pages over guessing.
- Keep the wiki coherent: read before you write, and write in small, purposeful edits.

Wiki layout and file discovery:
- Page paths are wiki slugs (e.g. `schema`, `index`, `notes/topic`). An optional trailing `.md` on the last segment refers to the same page; tools normalize this.
- **`schema`**: This wiki's conventions—structure, naming, and what belongs where. Read **schema** early before creating many new pages or reorganizing, so you follow this wiki's rules.
- **`index`**: Catalog or map of important pages. Read **index** when orienting yourself or after substantive adds/removals so you can keep it truthful (or tell the user it needs an update).
- **`log`**: Chronological operational history. Prefer **append_log_entry** for meaningful changes; avoid rewriting the whole log unless you are fixing a specific mistake.
- **Finding pages**: Use **list_pages** for the sorted list of active paths. Use **search_pages** to find pages whose *bodies* match a substring. Use **read_page** for full text. Use **page_exists** or **get_page_metadata** when you only need existence, soft-delete status, or size without loading the full body.

Tool discipline:
- Use tools when you need current page content, a list of pages, search hits, or to change the wiki. Do not fabricate tool results or page text.
- If unsure a path exists, use page_exists or get_page_metadata, or search_pages, before read_page / write_page.
- Paths: lowercase wiki paths with "/" segments (e.g. notes/topic). No "." or ".." segments. Optional trailing ".md" on the last segment is allowed and means the same page.
- Never delete the reserved pages **index**, **schema**, or **log**.
- Soft-deleted pages may still exist; use get_page_metadata / page_exists if read_page or list_pages behave unexpectedly.

Writing style:
- Answers: concise Markdown. Link to wiki pages by path when relevant (e.g. `[[some/page]]` or backticks) so the UI can wire citations.
- When you change the wiki, say what you changed at a high level when you reply to the user.

Safety and limits:
- If a tool returns an error JSON, acknowledge the limitation and either fix inputs (e.g. path) or choose another approach—do not pretend the action succeeded.
- If you cannot complete the task within the conversation constraints, say what is missing (e.g. which page to create or what to upload).

When the user message clearly asks only for an explanation and the wiki has no bearing, you may answer without tools—but default to verifying with read/search when the question might be grounded in stored notes.
"""

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


def _log_preview(s: str, limit: int = 200) -> str:
    s = s.replace("\n", " ") # replace newlines with spaces
    return s if len(s) <= limit else f"{s[:limit]}…({len(s)} chars)" # return the string if it's less than the limit, otherwise return the string up to the limit and an ellipsis

_RESPONSE_LIFECYCLE_EVENT_TYPES = frozenset({
    "response.completed",
    "response.failed",
    "response.incomplete",
})


def _event_response(event: object) -> object | None:
    if isinstance(event, dict):
        return event.get("response")
    return getattr(event, "response", None)


def _usage_summary(usage: object | None) -> dict[str, object] | None:
    if usage is None:
        return None
    if isinstance(usage, dict):
        out_details = usage.get("output_tokens_details")
        reasoning = None
        if isinstance(out_details, dict):
            reasoning = out_details.get("reasoning_tokens")
        return {
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "reasoning_tokens": reasoning,
        }
    out_details = getattr(usage, "output_tokens_details", None)
    reasoning = None
    if isinstance(out_details, dict):
        reasoning = out_details.get("reasoning_tokens")
    elif out_details is not None:
        reasoning = getattr(out_details, "reasoning_tokens", None)
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
        "reasoning_tokens": reasoning,
    }


def _log_response_lifecycle(agent_id: int, phase: str, event: object) -> None:
    event_type = _event_type(event)
    resp = _event_response(event)
    if resp is None:
        logger.warning(
            "response lifecycle event missing response payload (agent_id=%s phase=%s type=%s)",
            agent_id,
            phase,
            event_type,
        )
        return
    rid = resp.get("id") if isinstance(resp, dict) else getattr(resp, "id", None)
    model = resp.get("model") if isinstance(resp, dict) else getattr(resp, "model", None)
    status = resp.get("status") if isinstance(resp, dict) else getattr(resp, "status", None)

    if event_type == "response.completed":
        usage = resp.get("usage") if isinstance(resp, dict) else getattr(resp, "usage", None)
        logger.info(
            "response completed (agent_id=%s phase=%s response_id=%s model=%s status=%s usage=%s)",
            agent_id,
            phase,
            rid,
            model,
            status,
            _usage_summary(usage),
        )
    elif event_type == "response.failed":
        err = resp.get("error") if isinstance(resp, dict) else getattr(resp, "error", None)
        code = message = None
        if isinstance(err, dict):
            code = err.get("code")
            message = err.get("message")
        elif err is not None:
            code = getattr(err, "code", None)
            message = getattr(err, "message", None)
        logger.error(
            "response failed (agent_id=%s phase=%s response_id=%s model=%s code=%s message=%s)",
            agent_id,
            phase,
            rid,
            model,
            code,
            message,
        )
    elif event_type == "response.incomplete":
        inc = resp.get("incomplete_details") if isinstance(resp, dict) else getattr(resp, "incomplete_details", None)
        reason = None
        if isinstance(inc, dict):
            reason = inc.get("reason")
        elif inc is not None:
            reason = getattr(inc, "reason", None)
        logger.warning(
            "response incomplete (agent_id=%s phase=%s response_id=%s model=%s status=%s reason=%s)",
            agent_id,
            phase,
            rid,
            model,
            status,
            reason,
        )
    else:
        logger.warning(
            "unexpected response lifecycle event (agent_id=%s phase=%s type=%s)",
            agent_id,
            phase,
            event_type,
        )


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
            logger.error("agent loop exceeded TIMEOUT (agent_id=%s, limit_s=%s)", agent_id, TIMEOUT)
            raise TimeoutError("Agent loop exceeded TIMEOUT")
        
        if MAX_STEPS > 0 and tool_rounds >= MAX_STEPS:
            logger.info(
                "agent synthesis turn (agent_id=%s, tool_rounds=%s, max_steps=%s)",
                agent_id,
                tool_rounds,
                MAX_STEPS,
            )
            for attempt in range(MAX_RETRIES):
                try:
                    async with asyncio.timeout(TURN_API_TIMEOUT):
                        stream = await client.responses.create(
                            model="gpt-5.4-nano-2026-03-17",
                            instructions=MAIN_WIKI_AGENT_INSTRUCTIONS,
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
                        closing_text = ""
                        async for event in stream:
                            event_type = _event_type(event)
                            if event_type == "response.output_text.delta":
                                delta = event["delta"] if isinstance(event, dict) else event.delta
                                if isinstance(delta, str) and delta:
                                    closing_text += delta
                                    yield delta
                            elif event_type == "response.output_text.done":
                                final = (
                                    event.get("text") if isinstance(event, dict) else getattr(event, "text", None)
                                )
                                if isinstance(final, str):
                                    closing_text = final
                                if closing_text.strip():
                                    input_list.append({"role": "assistant", "content": closing_text})
                            elif event_type in _RESPONSE_LIFECYCLE_EVENT_TYPES:
                                _log_response_lifecycle(agent_id, "synthesis", event)
                    return
                except TimeoutError:
                    logger.error(
                        "synthesis turn exceeded TURN_API_TIMEOUT (agent_id=%s limit_s=%s attempt=%s)",
                        agent_id,
                        TURN_API_TIMEOUT,
                        attempt + 1,
                    )
                    raise TimeoutError(
                        f"Agent synthesis turn exceeded TURN_API_TIMEOUT "
                        f"({TURN_API_TIMEOUT}s, agent_id={agent_id})"
                    ) from None
                except (APITimeoutError, InternalServerError, RateLimitError, APIConnectionError) as e:
                    if attempt < MAX_RETRIES - 1:
                        logger.warning(
                            "responses.create retry (phase=synthesis agent_id=%s attempt=%s/%s): %s",
                            agent_id,
                            attempt + 1,
                            MAX_RETRIES,
                            e,
                        )
                        yield {
                            "type": "api_retry",
                            "code": getattr(e, "status_code", None),
                            "message": str(e),
                            "attempt": attempt + 1,
                            "max_attempts": MAX_RETRIES,
                        }
                        await asyncio.sleep(2**attempt)
                    else:
                        logger.error(
                            "synthesis turn max retries exhausted (agent_id=%s, last_error=%s)",
                            agent_id,
                            e,
                        )
                        raise

        current_output_text = ""
        tools_used_this_turn = False
        
        stream = None
        for attempt in range(MAX_RETRIES):
            try:
                async with asyncio.timeout(TURN_API_TIMEOUT):
                    stream = await client.responses.create(
                        model="gpt-5.4-nano-2026-03-17",
                        instructions=MAIN_WIKI_AGENT_INSTRUCTIONS,
                        max_tokens=4096,
                        tools=defs,
                        input=input_list,
                        stream=True,
                        reasoning={"effort": "low"},
                    )
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
                                    logger.warning(
                                        "invalid tool arguments JSON (agent_id=%s tool=%s preview=%s)",
                                        agent_id,
                                        tool_name,
                                        _log_preview(args_str, 120),
                                    )
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

                            logger.info(
                                "tool call (agent_id=%s name=%s args_len=%s)",
                                agent_id,
                                tool_name,
                                len(args_str),
                            )
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
                                logger.error(
                                    "tool execution timeout (agent_id=%s tool=%s limit_s=%s)",
                                    agent_id,
                                    tool_name,
                                    TOOL_EXECUTION_TIMEOUT,
                                )
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
                            logger.info(
                                "tool done (agent_id=%s tool=%s result_keys=%s out_len=%s)",
                                agent_id,
                                tool_name,
                                list(result.keys()) if isinstance(result, dict) else type(result).__name__,
                                len(out),
                            )
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

                        elif event_type in _RESPONSE_LIFECYCLE_EVENT_TYPES:
                            _log_response_lifecycle(agent_id, "tool_turn", event)
                break
            except TimeoutError:
                logger.error(
                    "tool turn exceeded TURN_API_TIMEOUT (agent_id=%s limit_s=%s attempt=%s)",
                    agent_id,
                    TURN_API_TIMEOUT,
                    attempt + 1,
                )
                raise TimeoutError(
                    f"Agent tool turn exceeded TURN_API_TIMEOUT ({TURN_API_TIMEOUT}s, agent_id={agent_id})"
                ) from None
            except (APITimeoutError, InternalServerError, RateLimitError, APIConnectionError) as e:
                if attempt < MAX_RETRIES - 1:
                    logger.warning(
                        "responses.create/stream retry (phase=tool_turn agent_id=%s attempt=%s/%s): %s",
                        agent_id,
                        attempt + 1,
                        MAX_RETRIES,
                        e,
                    )
                    yield {
                        "type": "api_retry",
                        "code": getattr(e, "status_code", None),
                        "message": str(e),
                        "attempt": attempt + 1,
                        "max_attempts": MAX_RETRIES,
                    }
                    await asyncio.sleep(2**attempt)
                else:
                    logger.error(
                        "tool turn max retries exhausted (agent_id=%s last_error=%s)",
                        agent_id,
                        e,
                    )
                    raise

        if not tools_used_this_turn:
            logger.info("agent loop end without tools (agent_id=%s)", agent_id)
            return

