from __future__ import annotations

import json
import os
import re
import threading
import uuid
from datetime import datetime
from enum import IntEnum
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class LogLevel(IntEnum):
    OFF = 0
    CRITICAL = 1
    BRIEF = 2
    NORMAL = 3
    FULL = 4

LISTEN_HOST = os.getenv("LISTEN_HOST", "127.0.0.1")
LISTEN_PORT = int(os.getenv("LISTEN_PORT", 11435))

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "127.0.0.1")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", 11434))

LOG_FILE = os.getenv("LOG_FILE", "ollama_proxy.log")

LOG_LEVEL = os.getenv("LOG_LEVEL", "CRITICAL")

_LOG_LEVEL_MAP = {
    "OFF": LogLevel.OFF,
    "CRITICAL": LogLevel.CRITICAL,
    "BRIEF": LogLevel.BRIEF,
    "NORMAL": LogLevel.NORMAL,
    "FULL": LogLevel.FULL,
}
CURRENT_LOG_LEVEL = _LOG_LEVEL_MAP.get(LOG_LEVEL.upper(), LogLevel.NORMAL)

ENABLE_TOOL_INJECTION = os.getenv("ENABLE_TOOL_INJECTION", "True").lower() in ("true", "1", "yes", "t")
TOOLS_FILE = os.getenv("TOOLS_FILE", "ollama_tools.json")
TOOL_INJECTION_EXCLUDE_PATTERN = os.getenv("TOOL_INJECTION_EXCLUDE_PATTERN")

REPLACE_TITLE_PROMPT: str | None = (
    "Given the past chat concentrating on the "
    "user prompts, please reply with a title for "
    "the chat that is 3-4 words in length, all words "
    "used should be directly related to the content of "
    "the chat, avoid using verbs unless they are "
    "directly related to the content of the chat, no "
    "additional text or explanation, you don't need "
    "ending punctuation."
)
MIN_REPLACE_TITLE_PROMPT_LENGTH = 100

_request_counter = 0
_counter_lock = threading.Lock()


def next_request_id() -> int:
    global _request_counter
    with _counter_lock:
        _request_counter += 1
        return _request_counter


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def log(text: str = "", level: LogLevel = LogLevel.NORMAL) -> None:
    if CURRENT_LOG_LEVEL < level:
        return
    line = text.rstrip("\n")
    with open(LOG_FILE, "a", encoding="utf-8") as file:
        file.write(line + "\n")
    print(line)


def pretty_json(value: object) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def save_tools(tools: list[object]) -> None:
    if not ENABLE_TOOL_INJECTION:
        return
    try:
        with open(TOOLS_FILE, "w", encoding="utf-8") as file:
            json.dump(tools, file, indent=2, ensure_ascii=False)
            file.write("\n")
    except OSError as exc:
        log(f"WARNING: Could not save tools to {TOOLS_FILE!r}: {exc}", LogLevel.CRITICAL)


def load_saved_tools() -> list[object] | None:
    if not ENABLE_TOOL_INJECTION:
        return None
    try:
        with open(TOOLS_FILE, "r", encoding="utf-8") as file:
            tools = json.load(file)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        log(f"WARNING: Could not load tools from {TOOLS_FILE!r}: {exc}", LogLevel.CRITICAL)
        return None

    if not isinstance(tools, list):
        log(f"WARNING: {TOOLS_FILE!r} does not contain a JSON array.", LogLevel.CRITICAL)
        return None
    return tools


def get_message_text(data: dict[str, object]) -> str:
    messages = data.get("messages")
    if not isinstance(messages, list):
        return ""

    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
    return "\n".join(parts)


def tool_injection_is_excluded(data: dict[str, object]) -> bool:
    pattern = TOOL_INJECTION_EXCLUDE_PATTERN
    if not pattern:
        return False
    try:
        regex = re.compile(pattern, re.DOTALL)
    except re.error as exc:
        log(f"WARNING: Invalid TOOL_INJECTION_EXCLUDE_PATTERN: {exc}", LogLevel.CRITICAL)
        return False

    message_text = get_message_text(data)
    if regex.search(message_text):
        return True
    return False


def title_prompt_replacement_is_valid() -> bool:
    prompt = REPLACE_TITLE_PROMPT
    if not prompt or len(prompt) < MIN_REPLACE_TITLE_PROMPT_LENGTH:
        return False
    return True


def apply_title_prompt_replacement(data: dict[str, object]) -> bool:
    if not title_prompt_replacement_is_valid():
        return False
    messages = data.get("messages")
    if not isinstance(messages, list):
        return False

    if messages:
        last_message = messages[-1]
        if isinstance(last_message, dict) and isinstance(last_message.get("content"), str):
            last_message["content"] = REPLACE_TITLE_PROMPT
            last_message["role"] = "system"
            return True
    return False


def inject_saved_tools(data: dict[str, object]) -> bool:
    if not ENABLE_TOOL_INJECTION or "tools" in data:
        return False

    if tool_injection_is_excluded(data):
        log("", LogLevel.BRIEF)
        log("TOOL INJECTION SKIPPED: request matched TOOL_INJECTION_EXCLUDE_PATTERN", LogLevel.BRIEF)
        log(f"PATTERN: {TOOL_INJECTION_EXCLUDE_PATTERN!r}", LogLevel.BRIEF)
        return False

    tools = load_saved_tools()
    if tools is None:
        return False

    data["tools"] = tools
    return True


def ensure_user_message(data: dict[str, object]) -> bool:
    """Injects a synthetic user message if Continue.dev drops it during a tool loop."""
    messages = data.get("messages")
    if not isinstance(messages, list) or not messages:
        return False
        
    has_user = any(isinstance(m, dict) and m.get("role") == "user" for m in messages)
    if not has_user:
        insert_idx = 1 if (isinstance(messages[0], dict) and messages[0].get("role") == "system") else 0
        messages.insert(insert_idx, {
            "role": "user",
            "content": "Please continue."
        })
        log("INJECTED SYNTHETIC USER MESSAGE TO BYPASS QWEN TEMPLATE CRASH", LogLevel.BRIEF)
        return True

    return False


def print_request(request_id: int, path: str, body: bytes) -> dict[str, object] | None:
    log("", LogLevel.BRIEF)
    log("=" * 80, LogLevel.BRIEF)
    log(f"REQUEST #{request_id}  {timestamp()}", LogLevel.BRIEF)
    log(path, LogLevel.BRIEF)
    log("=" * 80, LogLevel.BRIEF)

    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        log("REQUEST BODY IS NOT JSON:", LogLevel.CRITICAL)
        log(body.decode("utf-8", errors="replace"), LogLevel.CRITICAL)
        return None

    if not isinstance(data, dict):
        log("REQUEST JSON is not an object:", LogLevel.CRITICAL)
        log(pretty_json(data), LogLevel.CRITICAL)
        return None

    log(f"MODEL: {data.get('model')!r}", LogLevel.BRIEF)
    log(f"STREAM: {data.get('stream')!r}", LogLevel.BRIEF)

    tools = data.get("tools")
    log("", LogLevel.BRIEF)
    log(f"TOOLS PRESENT: {tools is not None}", LogLevel.BRIEF)
    log(f"TOOL COUNT: {len(tools) if isinstance(tools, list) else 0}", LogLevel.BRIEF)

    if isinstance(tools, list):
        save_tools(tools)
        log("", LogLevel.FULL)
        log("ADVERTISED TOOLS:", LogLevel.FULL)
        log("-" * 80, LogLevel.FULL)

        for index, tool in enumerate(tools, 1):
            if not isinstance(tool, dict):
                log(f"[{index}] {tool!r}", LogLevel.FULL)
                continue

            function = tool.get("function", {})
            if not isinstance(function, dict):
                function = {}

            name = function.get("name")
            description = function.get("description")
            parameters = function.get("parameters")

            log(f"[{index}] {name}", LogLevel.FULL)
            if description:
                log(f"    description: {description}", LogLevel.FULL)
            if parameters is not None:
                log("    parameters:", LogLevel.FULL)
                for line in pretty_json(parameters).splitlines():
                    log("        " + line, LogLevel.FULL)

    messages = data.get("messages", [])
    log("", LogLevel.BRIEF)
    log(f"MESSAGE COUNT: {len(messages) if isinstance(messages, list) else 0}", LogLevel.BRIEF)
    log("", LogLevel.BRIEF)
    log("MESSAGES:", LogLevel.BRIEF)
    log("-" * 80, LogLevel.BRIEF)

    if isinstance(messages, list) and messages:
        messages_to_print = messages
        if CURRENT_LOG_LEVEL <= LogLevel.BRIEF:
            log("(History omitted due to BRIEF log level. Showing last message only.)", LogLevel.BRIEF)
            messages_to_print = [messages[-1]]

        for index, message in enumerate(messages_to_print):
            if CURRENT_LOG_LEVEL <= LogLevel.BRIEF:
                log("MESSAGE [LATEST]", LogLevel.BRIEF)
            else:
                log(f"MESSAGE [{index}]", LogLevel.NORMAL)

            log_level_for_msg = LogLevel.BRIEF if CURRENT_LOG_LEVEL <= LogLevel.BRIEF else LogLevel.NORMAL
            
            if not isinstance(message, dict):
                log(pretty_json(message), log_level_for_msg)
                continue

            log(f"  role: {message.get('role')!r}", log_level_for_msg)
            content = message.get("content")
            if content:
                log("  content:", log_level_for_msg)
                for line in str(content).splitlines():
                    log("    " + line, log_level_for_msg)

            if "tool_calls" in message:
                log("  tool_calls:", log_level_for_msg)
                log(pretty_json(message["tool_calls"]), log_level_for_msg)
            if "tool_name" in message:
                log(f"  tool_name: {message['tool_name']!r}", log_level_for_msg)

    log("", LogLevel.FULL)
    log("FULL REQUEST JSON:", LogLevel.FULL)
    log("-" * 80, LogLevel.FULL)
    log(pretty_json(data), LogLevel.FULL)
    return data


def summarize_response_chunk(chunk: dict[str, object], response_state: dict[str, object]) -> None:
    message = chunk.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content:
            response_state["content"].append(content)

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            response_state["tool_calls"].extend(tool_calls)

    if chunk.get("done"):
        response_state["done"] = True


def print_response(request_id, response_state):
    tool_calls = response_state.get("tool_calls", [])
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict): continue
            func = call.get("function", {})
            if isinstance(func, dict):
                t_name = func.get("name")
                if t_name in ("run_terminal_command", "project_run_terminal_command"):
                    log("=" * 80, LogLevel.CRITICAL)
                    log(f"CRITICAL TOOL CALL IN RESPONSE: {t_name} (Request #{request_id})", LogLevel.CRITICAL)
                    log(pretty_json(func.get("arguments", {})), LogLevel.CRITICAL)
                    log("=" * 80, LogLevel.CRITICAL)

    log("-" * 80, LogLevel.BRIEF)
    log(f"RESPONSE #{request_id} SUMMARY  {timestamp()}", LogLevel.BRIEF)
    log("-" * 80, LogLevel.BRIEF)

    content = "".join(response_state["content"]) 
    log("ASSISTANT CONTENT:", LogLevel.BRIEF)
    if content:
        log(content, LogLevel.BRIEF)
    else:
        log("(none)", LogLevel.BRIEF)

    log("", LogLevel.BRIEF)
    log(f"STRUCTURED TOOL CALL COUNT: {len(tool_calls)}", LogLevel.BRIEF)

    if tool_calls:
        log("", LogLevel.NORMAL)
        log("STRUCTURED TOOL CALLS:", LogLevel.NORMAL)
        log(pretty_json(tool_calls), LogLevel.NORMAL)
        log("", LogLevel.BRIEF)
        log("STRUCTURED TOOL NAMES:", LogLevel.BRIEF)
        for call in tool_calls:
            if not isinstance(call, dict): continue
            function = call.get("function", {})
            if isinstance(function, dict):
                log(f"  {function.get('name')!r}", LogLevel.BRIEF)

    if "<function=" in content:
        log("", LogLevel.CRITICAL)
        log("!!! TEXTUAL FUNCTION-CALL SYNTAX DETECTED !!!", LogLevel.CRITICAL)
        log("The assistant response contains '<function=...>' as CONTENT.", LogLevel.CRITICAL)
        log("This may mean the model generated a textual tool call rather than returning native tool_calls.", LogLevel.CRITICAL)

    log("", LogLevel.FULL)
    log("RAW RECONSTRUCTED ASSISTANT CONTENT:", LogLevel.FULL)
    log(repr(content), LogLevel.FULL)


# ==============================================================================
# STREAM PARSER (Pure Data Evaluation, No I/O)
# ==============================================================================

class QwenStreamParser:
    """
    A state machine that processes text chunks one by one.
    It yields ('text', string) for normal content, or ('tool', dict) when an XML
    tool call is completely intercepted and parsed.
    """
    PREFIX = "<function="

    def __init__(self, available_tools: set[str]):
        self.available_tools = available_tools
        self.buffer = ""
        self.state = "NORMAL" # States: NORMAL, CHECKING_PREFIX, INSIDE_XML

    def process_content(self, content: str):
        if not content:
            return

        if self.state == "NORMAL":
            if self.PREFIX in content:
                parts = content.split(self.PREFIX, 1)
                if parts[0]:
                    cleaned = parts[0].replace("</tool_call>", "")
                    if cleaned:
                        yield ("text", cleaned)
                
                self.buffer = self.PREFIX + parts[1]
                self.state = "INSIDE_XML"
                yield from self._process_buffer()
            else:
                partial_found = False
                for i in range(len(self.PREFIX) - 1, 0, -1):
                    if content.endswith(self.PREFIX[:i]):
                        text_before = content[:-i]
                        cleaned = text_before.replace("</tool_call>", "")
                        if cleaned:
                            yield ("text", cleaned)
                            
                        self.buffer = self.PREFIX[:i]
                        self.state = "CHECKING_PREFIX"
                        partial_found = True
                        break
                
                if not partial_found:
                    cleaned = content.replace("</tool_call>", "")
                    if cleaned:
                        yield ("text", cleaned)

        elif self.state == "CHECKING_PREFIX":
            self.buffer += content
            if self.buffer.startswith(self.PREFIX):
                self.state = "INSIDE_XML"
                yield from self._process_buffer()
            elif not self.PREFIX.startswith(self.buffer):
                # False alarm. Pop the first character, yield it, and re-evaluate the rest
                first_char = self.buffer[0]
                remainder = self.buffer[1:]
                self.buffer = ""
                self.state = "NORMAL"
                
                cleaned = first_char.replace("</tool_call>", "")
                if cleaned:
                    yield ("text", cleaned)
                    
                if remainder:
                    yield from self.process_content(remainder)

        elif self.state == "INSIDE_XML":
            self.buffer += content
            yield from self._process_buffer()

    def _process_buffer(self):
        end_tag = "</function>"
        if end_tag not in self.buffer:
            return
            
        # We have a full XML block. Attempt extraction.
        pattern = r"^<function=([a-zA-Z0-9_]+)>\s*(.*?)\s*</function>"
        match = re.search(pattern, self.buffer, re.DOTALL)
        
        # Calculate exactly where this block ends so we can process the remainder
        end_idx = self.buffer.find(end_tag) + len(end_tag)
        remainder = self.buffer[end_idx:].lstrip()
        
        # Strip trailing </tool_call> if the model emitted it right after </function>
        if remainder.startswith("</tool_call>"):
            remainder = remainder[len("</tool_call>"):].lstrip()
            
        if match:
            name = match.group(1)
            args_str = match.group(2).strip() or "{}"
            
            # Only yield if it's an available tool and valid JSON.
            # Otherwise, it skips the yield, effectively "trashing" the hallucination.
            if name in self.available_tools:
                try:
                    args_dict = json.loads(args_str)
                    if isinstance(args_dict, dict):
                        yield ("tool", {
                            "name": name,
                            "arguments": args_dict
                        })
                except json.JSONDecodeError:
                    pass 

        # Reset state and process anything that came after the XML block
        self.buffer = ""
        self.state = "NORMAL"
        if remainder:
            yield from self.process_content(remainder)

    def finalize(self):
        if self.buffer:
            if self.state == "CHECKING_PREFIX":
                # Flush the incomplete prefix
                cleaned = self.buffer.replace("</tool_call>", "")
                if cleaned:
                    yield ("text", cleaned)
            # If state is INSIDE_XML, it's an incomplete tool block. We trash it!
            
        self.buffer = ""
        self.state = "NORMAL"


# ==============================================================================
# ORCHESTRATOR / HTTP HANDLER
# ==============================================================================

class OllamaProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format_string: str, *args: object) -> None:
        pass

    def do_GET(self) -> None: self.forward_request()
    def do_POST(self) -> None: self.forward_request()
    def do_PUT(self) -> None: self.forward_request()
    def do_DELETE(self) -> None: self.forward_request()
    def do_PATCH(self) -> None: self.forward_request()
    def do_HEAD(self) -> None: self.forward_request()

    def _read_request_body(self) -> bytes | None:
        content_length = self.headers.get("Content-Length")
        if content_length is None: return b""
        try:
            length = int(content_length)
        except ValueError:
            self.send_error(400, "Invalid Content-Length")
            return None
        return self.rfile.read(length)

    def _prepare_request(self, request_id: int, body: bytes) -> tuple[bytes, dict[str, object] | None]:
        request_data: dict[str, object] | None = None
        is_chat = self.path.startswith("/api/chat")
        
        if not (is_chat and body):
            self._log_non_chat_request(request_id, body)
            return body, request_data

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            self._log_non_chat_request(request_id, body)
            return body, request_data

        if not isinstance(parsed, dict):
            self._log_non_chat_request(request_id, body)
            return body, request_data

        request_data = parsed
        modified = False

        if ensure_user_message(request_data):
            modified = True

        if inject_saved_tools(request_data):
            modified = True
            log("", LogLevel.BRIEF)
            log(f"REQUEST #{request_id}: INJECTED SAVED TOOLS", LogLevel.BRIEF)
            log(f"INJECTED TOOL COUNT: {len(request_data['tools'])}", LogLevel.BRIEF)
        elif tool_injection_is_excluded(request_data):
            if apply_title_prompt_replacement(request_data):
                modified = True
                log("", LogLevel.BRIEF)
                log(f"REQUEST #{request_id}: REPLACED TITLE PROMPT", LogLevel.BRIEF)

        if modified:
            body = json.dumps(request_data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

        print_request(request_id, self.path, body)
        return body, request_data

    def _log_non_chat_request(self, request_id: int, body: bytes) -> None:
        log("", LogLevel.BRIEF)
        log("=" * 80, LogLevel.BRIEF)
        log(f"REQUEST #{request_id}  {timestamp()}", LogLevel.BRIEF)
        log(f"{self.command} {self.path}", LogLevel.BRIEF)
        log("=" * 80, LogLevel.BRIEF)
        if not body: return
        try:
            log(pretty_json(json.loads(body)), LogLevel.FULL)
        except json.JSONDecodeError:
            log(body.decode("utf-8", errors="replace"), LogLevel.FULL)

    def _build_forward_headers(self) -> dict[str, str]:
        excluded_headers = {"host", "content-length", "connection", "transfer-encoding"}
        headers = {k: v for k, v in self.headers.items() if k.lower() not in excluded_headers}
        headers["Host"] = f"{OLLAMA_HOST}:{OLLAMA_PORT}"
        return headers

    def _send_response_headers(self, response: object) -> None:
        self.send_response(response.status, response.reason)
        excluded_headers = {"connection", "transfer-encoding", "content-length"}
        for key, value in response.getheaders():
            if key.lower() not in excluded_headers:
                self.send_header(key, value)
        self.send_header("Connection", "close")
        self.end_headers()

    @staticmethod
    def _get_xml_tool_names(request_data: dict[str, object] | None) -> set[str]:
        if not isinstance(request_data, dict): return set()
        tools = request_data.get("tools", [])
        if not isinstance(tools, list): return set()
        
        names = set()
        for tool in tools:
            if not isinstance(tool, dict): continue
            func = tool.get("function", {})
            if not isinstance(func, dict): continue
            name = func.get("name")
            if isinstance(name, str):
                names.add(name)
        return names

    @staticmethod
    def _should_intercept_xml(is_chat: bool, request_data: dict[str, object] | None) -> bool:
        if not is_chat or not isinstance(request_data, dict): return False
        model_name = str(request_data.get("model", "")).lower()
        tools = request_data.get("tools", [])
        return "qwen" in model_name and isinstance(tools, list) and bool(tools)

    @staticmethod
    def _model_name(request_data: dict[str, object] | None) -> str:
        if not isinstance(request_data, dict): return ""
        return str(request_data.get("model", "")).lower()

    def _write_chunk(self, chunk: bytes) -> None:
        self.wfile.write(chunk)
        self.wfile.flush()

    def _write_json_chunk(self, chunk: dict[str, object]) -> None:
        encoded = (json.dumps(chunk, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        self._write_chunk(encoded)

    def _build_ollama_chunk(self, model_name: str, content: str = "", tool_calls: list = None, done: bool = False) -> dict[str, object]:
        """Constructs a perfectly clean Ollama JSON payload."""
        msg: dict[str, object] = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return {
            "model": model_name,
            "created_at": timestamp(),
            "message": msg,
            "done": done
        }

    def _log_response_chunk(self, request_id: int, message: dict[str, object]) -> None:
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            log("", LogLevel.NORMAL)
            log(f"RESPONSE #{request_id}: NATIVE TOOL CALL CHUNK", LogLevel.NORMAL)
            log(pretty_json(tool_calls), LogLevel.NORMAL)

        content = message.get("content")
        if isinstance(content, str) and content:
            log(f"RESPONSE #{request_id}: CONTENT CHUNK: {content!r}", LogLevel.NORMAL)

    def _stream_response(self, request_id: int, response: object, request_data: dict[str, object] | None) -> dict[str, object]:
        is_chat = self.path.startswith("/api/chat")
        model_name = self._model_name(request_data)
        should_intercept_xml = self._should_intercept_xml(is_chat, request_data)
        available_tool_names = self._get_xml_tool_names(request_data) if should_intercept_xml else set()

        response_state: dict[str, object] = {"content": [], "tool_calls": [], "done": False}
        parser = QwenStreamParser(available_tool_names)

        while True:
            chunk_bytes = response.readline()
            if not chunk_bytes:
                break

            # 1. Passthrough Logic (Non-Chat or Non-Qwen)
            if not should_intercept_xml:
                self._write_chunk(chunk_bytes)
                if is_chat:
                    try:
                        parsed = json.loads(chunk_bytes)
                        if isinstance(parsed, dict):
                            summarize_response_chunk(parsed, response_state)
                            msg = parsed.get("message")
                            if isinstance(msg, dict):
                                self._log_response_chunk(request_id, msg)
                    except Exception:
                        pass
                continue

            # 2. Parse Incoming Qwen Chunk
            try:
                parsed = json.loads(chunk_bytes)
            except Exception:
                self._write_chunk(chunk_bytes)
                continue

            if not isinstance(parsed, dict) or "message" not in parsed:
                self._write_chunk(chunk_bytes)
                continue

            message = parsed.get("message", {})
            
            # --- Native Tool Calls (Pass-through intact) ---
            native_calls = message.get("tool_calls")
            if isinstance(native_calls, list) and native_calls:
                response_state["tool_calls"].extend(native_calls)
                self._log_response_chunk(request_id, message)
                self._write_chunk(chunk_bytes)
                continue

            # --- Text Content / XML Interception ---
            content = message.get("content")
            is_done = parsed.get("done", False)

            if isinstance(content, str) and content:
                for action, payload in parser.process_content(content):
                    if action == "text":
                        out_chunk = self._build_ollama_chunk(model_name, content=payload)
                        self._write_json_chunk(out_chunk)
                        response_state["content"].append(payload)
                        
                        log(f"RESPONSE #{request_id}: CONTENT CHUNK: {payload!r}", LogLevel.NORMAL)

                    elif action == "tool":
                        formatted_call = {
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "type": "function",
                            "function": payload
                        }
                        out_chunk = self._build_ollama_chunk(model_name, tool_calls=[formatted_call])
                        self._write_json_chunk(out_chunk)
                        response_state["tool_calls"].append(formatted_call)

                        log("", LogLevel.CRITICAL)
                        log("=" * 80, LogLevel.CRITICAL)
                        log(f"*** PROXY ACTION (Request #{request_id}): INTERCEPTED QWEN XML TOOL CALL ***", LogLevel.CRITICAL)
                        log(f"EXTRACTED TOOLS:\n{pretty_json([formatted_call])}", LogLevel.NORMAL)
                        log("=" * 80, LogLevel.CRITICAL)
                        log("", LogLevel.CRITICAL)

            # --- End of Stream ---
            if is_done:
                for action, payload in parser.finalize():
                    if action == "text":
                        out_chunk = self._build_ollama_chunk(model_name, content=payload)
                        self._write_json_chunk(out_chunk)
                        response_state["content"].append(payload)
                        
                done_chunk = self._build_ollama_chunk(model_name, done=True)
                self._write_json_chunk(done_chunk)
                response_state["done"] = True
                break

        if is_chat:
            print_response(request_id, response_state)

        return response_state

    def forward_request(self) -> None:
        request_id = next_request_id()
        body = self._read_request_body()
        if body is None: return

        body, request_data = self._prepare_request(request_id, body)
        connection = HTTPConnection(OLLAMA_HOST, OLLAMA_PORT, timeout=600)

        try:
            connection.request(self.command, self.path, body=body, headers=self._build_forward_headers())
            response = connection.getresponse()
            self._send_response_headers(response)
            self._stream_response(request_id, response, request_data)

            log("", LogLevel.BRIEF)
            log(f"REQUEST #{request_id} COMPLETE", LogLevel.BRIEF)
            log("", LogLevel.BRIEF)

        except Exception as exc:
            log("", LogLevel.CRITICAL)
            log(f"REQUEST #{request_id} PROXY ERROR:", LogLevel.CRITICAL)
            log(repr(exc), LogLevel.CRITICAL)
            log("", LogLevel.CRITICAL)
            try:
                self.send_error(502, f"Proxy error: {exc}")
            except Exception: pass
        finally:
            connection.close()


def main() -> None:
    server = ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), OllamaProxyHandler)
    print("\n" + "=" * 70)
    print("Ollama debugging proxy")
    print("=" * 70)
    print(f"Listen    : http://{LISTEN_HOST}:{LISTEN_PORT}")
    print(f"Ollama    : http://{OLLAMA_HOST}:{OLLAMA_PORT}")
    print(f"Log       : {LOG_FILE}")
    print(f"Log Level : {LOG_LEVEL} (Level {CURRENT_LOG_LEVEL})")
    print(f"Injection : {ENABLE_TOOL_INJECTION}")
    print(f"Exclude   : {TOOL_INJECTION_EXCLUDE_PATTERN!r}")
    print(f"Tools     : {TOOLS_FILE}\n")
    print("Press Ctrl+C to stop.")
    print("=" * 70 + "\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping proxy...")
    finally:
        server.server_close()