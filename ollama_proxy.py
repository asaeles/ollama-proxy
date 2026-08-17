from __future__ import annotations

import json
import re
import threading
from datetime import datetime
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = 11435

OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434

LOG_FILE = "ollama_proxy.log"

# Set to False to disable all proxy logging.
ENABLE_LOGGING = False

# Set to True to remember the latest advertised tools and inject them into
# subsequent /api/chat requests when Continue omits the "tools" property.
ENABLE_TOOL_INJECTION = True

# File containing the most recently captured tools array.
TOOLS_FILE = "ollama_tools.json"

# If this regular expression matches the textual content of a request's
# messages, saved tools will NOT be injected into that request.
#
# Set to None or "" to disable the exclusion.
TOOL_INJECTION_EXCLUDE_PATTERN = (
    # r"asldknalsjdhgasdmvnsljdfnbgu9wrhtfpokjsdfvljnzdoufhgoierhglzjsbfg"
    r"Given the following.+title for the chat"
)

# When TOOL_INJECTION_EXCLUDE_PATTERN matches a request, the matched
# message content is normally left untouched (tools are simply not
# injected). If REPLACE_TITLE_PROMPT is set to a string of at least
# MIN_REPLACE_TITLE_PROMPT_LENGTH characters, that string is used to
# replace the request's message content instead. This is intended for
# the Continue "generate a title" request, whose prompt embeds the
# full prior turn and can otherwise confuse tool-calling models.
#
# Set to None or "" to disable.
REPLACE_TITLE_PROMPT: str | None = (
    "Given the past chat concentrating on the first "
    "user prompt, please reply with a title for "
    "the chat that is 3-4 words in length, all words "
    "used should be directly related to the content of "
    "the chat, avoid using verbs unless they are "
    "directly related to the content of the chat, no "
    "additional text or explanation, you don't need "
    "ending punctuation."
)

# Minimum length REPLACE_TITLE_PROMPT must meet to be considered valid.
MIN_REPLACE_TITLE_PROMPT_LENGTH = 100


_request_counter = 0
_counter_lock = threading.Lock()


def next_request_id() -> int:
    global _request_counter

    with _counter_lock:
        _request_counter += 1
        return _request_counter


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(
        timespec="milliseconds"
    )


def log(text: str = "") -> None:
    if not ENABLE_LOGGING:
        return

    line = text.rstrip("\n")

    with open(
        LOG_FILE,
        "a",
        encoding="utf-8",
    ) as file:
        file.write(line + "\n")

    print(line)


def pretty_json(value: object) -> str:
    return json.dumps(
        value,
        indent=2,
        ensure_ascii=False,
    )


def save_tools(tools: list[object]) -> None:
    """
    Persist the latest tool definitions received from Continue.
    """
    if not ENABLE_TOOL_INJECTION:
        return

    try:
        with open(
            TOOLS_FILE,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                tools,
                file,
                indent=2,
                ensure_ascii=False,
            )
            file.write("\n")

    except OSError as exc:
        log(
            f"WARNING: Could not save tools to "
            f"{TOOLS_FILE!r}: {exc}"
        )


def load_saved_tools() -> list[object] | None:
    """
    Load the most recently saved tool definitions.
    """
    if not ENABLE_TOOL_INJECTION:
        return None

    try:
        with open(
            TOOLS_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            tools = json.load(file)

    except FileNotFoundError:
        return None

    except (
        OSError,
        json.JSONDecodeError,
    ) as exc:
        log(
            f"WARNING: Could not load tools from "
            f"{TOOLS_FILE!r}: {exc}"
        )
        return None

    if not isinstance(tools, list):
        log(
            f"WARNING: {TOOLS_FILE!r} does not contain "
            "a JSON array."
        )
        return None

    return tools


def get_message_text(
    data: dict[str, object],
) -> str:
    """
    Extract textual message content from an Ollama chat request.

    Each message's content is joined with newlines. Non-textual message
    fields are ignored because the exclusion pattern is intended to match
    the actual conversational text.
    """
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


def tool_injection_is_excluded(
    data: dict[str, object],
) -> bool:
    """
    Return True when the request matches the configured injection
    exclusion pattern.
    """
    pattern = TOOL_INJECTION_EXCLUDE_PATTERN

    if not pattern:
        return False

    try:
        regex = re.compile(
            pattern,
            re.DOTALL,
        )

    except re.error as exc:
        log(
            "WARNING: Invalid "
            "TOOL_INJECTION_EXCLUDE_PATTERN: "
            f"{exc}"
        )
        return False

    message_text = get_message_text(data)

    if regex.search(message_text):
        return True

    return False


def title_prompt_replacement_is_valid() -> bool:
    """
    Return True when REPLACE_TITLE_PROMPT is set to a usable
    replacement string.
    """
    prompt = REPLACE_TITLE_PROMPT

    if not prompt:
        return False

    if len(prompt) < MIN_REPLACE_TITLE_PROMPT_LENGTH:
        return False

    return True


def apply_title_prompt_replacement(
    data: dict[str, object],
) -> bool:
    """
    Replace the content of every textual message with
    REPLACE_TITLE_PROMPT.

    Intended to run only after tool_injection_is_excluded() has
    matched, so this is expected to operate on a single-message
    "generate a title" style request.
    """
    if not title_prompt_replacement_is_valid():
        return False

    messages = data.get("messages")

    if not isinstance(messages, list):
        return False

    replaced = False

    for message in messages:
        if not isinstance(message, dict):
            continue

        if isinstance(message.get("content"), str):
            message["content"] = REPLACE_TITLE_PROMPT
            replaced = True

    return replaced


def inject_saved_tools(
    data: dict[str, object],
) -> bool:
    """
    Inject saved tools when the request does not contain a 'tools'
    property and does not match the exclusion pattern.

    An explicitly supplied empty tools array is not replaced.
    """
    if not ENABLE_TOOL_INJECTION:
        return False

    if "tools" in data:
        return False

    if tool_injection_is_excluded(data):
        log()
        log(
            "TOOL INJECTION SKIPPED: "
            "request matched "
            "TOOL_INJECTION_EXCLUDE_PATTERN"
        )
        log(
            f"PATTERN: "
            f"{TOOL_INJECTION_EXCLUDE_PATTERN!r}"
        )
        return False

    tools = load_saved_tools()

    if tools is None:
        return False

    data["tools"] = tools

    return True


def print_request(
    request_id: int,
    path: str,
    body: bytes,
) -> dict[str, object] | None:
    log()
    log("=" * 80)
    log(
        f"REQUEST #{request_id}  "
        f"{timestamp()}"
    )
    log(path)
    log("=" * 80)

    try:
        data = json.loads(body)

    except json.JSONDecodeError:
        log("REQUEST BODY IS NOT JSON:")
        log(
            body.decode(
                "utf-8",
                errors="replace",
            )
        )
        return None

    if not isinstance(data, dict):
        log("REQUEST JSON is not an object:")
        log(pretty_json(data))
        return None

    log(
        f"MODEL: {data.get('model')!r}"
    )
    log(
        f"STREAM: {data.get('stream')!r}"
    )

    tools = data.get("tools")

    log()
    log(
        f"TOOLS PRESENT: "
        f"{tools is not None}"
    )
    log(
        f"TOOL COUNT: "
        f"{len(tools) if isinstance(tools, list) else 0}"
    )

    if isinstance(tools, list):
        save_tools(tools)

        log()
        log("ADVERTISED TOOLS:")
        log("-" * 80)

        for index, tool in enumerate(
            tools,
            1,
        ):
            if not isinstance(tool, dict):
                log(
                    f"[{index}] {tool!r}"
                )
                continue

            function = tool.get(
                "function",
                {},
            )

            if not isinstance(
                function,
                dict,
            ):
                function = {}

            name = function.get("name")
            description = function.get(
                "description"
            )
            parameters = function.get(
                "parameters"
            )

            log(
                f"[{index}] {name}"
            )

            if description:
                log(
                    f"    description: "
                    f"{description}"
                )

            if parameters is not None:
                log("    parameters:")

                for line in pretty_json(
                    parameters
                ).splitlines():
                    log(
                        "        " + line
                    )

    messages = data.get(
        "messages",
        [],
    )

    log()
    log(
        f"MESSAGE COUNT: "
        f"{len(messages) if isinstance(messages, list) else 0}"
    )

    log()
    log("MESSAGES:")
    log("-" * 80)

    if isinstance(messages, list):
        for index, message in enumerate(
            messages
        ):
            log(
                f"MESSAGE [{index}]"
            )

            if not isinstance(
                message,
                dict,
            ):
                log(
                    pretty_json(message)
                )
                continue

            log(
                f"  role: "
                f"{message.get('role')!r}"
            )

            content = message.get(
                "content"
            )

            if content:
                log("  content:")

                for line in str(
                    content
                ).splitlines():
                    log(
                        "    " + line
                    )

            if "tool_calls" in message:
                log("  tool_calls:")
                log(
                    pretty_json(
                        message[
                            "tool_calls"
                        ]
                    )
                )

            if "tool_name" in message:
                log(
                    f"  tool_name: "
                    f"{message['tool_name']!r}"
                )

    log()
    log("FULL REQUEST JSON:")
    log("-" * 80)
    log(pretty_json(data))

    return data


def summarize_response_chunk(
    chunk: dict[str, object],
    response_state: dict[str, object],
) -> None:
    message = chunk.get("message")

    if isinstance(
        message,
        dict,
    ):
        content = message.get(
            "content"
        )

        if (
            isinstance(
                content,
                str,
            )
            and content
        ):
            response_state[
                "content"
            ].append(
                content
            )  # type: ignore[union-attr]

        tool_calls = message.get(
            "tool_calls"
        )

        if (
            isinstance(
                tool_calls,
                list,
            )
            and tool_calls
        ):
            response_state[
                "tool_calls"
            ].extend(
                tool_calls
            )  # type: ignore[union-attr]

    if chunk.get("done"):
        response_state["done"] = True


def print_response(
    request_id: int,
    response_state: dict[str, object],
) -> None:
    log()
    log("-" * 80)
    log(
        f"RESPONSE #{request_id} SUMMARY  "
        f"{timestamp()}"
    )
    log("-" * 80)

    content = "".join(
        response_state["content"]
    )  # type: ignore[arg-type]

    log("ASSISTANT CONTENT:")

    if content:
        log(content)
    else:
        log("(none)")

    tool_calls = response_state[
        "tool_calls"
    ]

    log()
    log(
        f"STRUCTURED TOOL CALL COUNT: "
        f"{len(tool_calls)}"
    )

    if tool_calls:
        log()
        log(
            "STRUCTURED TOOL CALLS:"
        )
        log(
            pretty_json(
                tool_calls
            )
        )

        log()
        log(
            "STRUCTURED TOOL NAMES:"
        )

        for call in tool_calls:
            if not isinstance(
                call,
                dict,
            ):
                continue

            function = call.get(
                "function",
                {},
            )

            if isinstance(
                function,
                dict,
            ):
                log(
                    f"  "
                    f"{function.get('name')!r}"
                )

    if "<function=" in content:
        log()
        log(
            "!!! TEXTUAL FUNCTION-CALL "
            "SYNTAX DETECTED !!!"
        )
        log(
            "The assistant response contains "
            "'<function=...>' as CONTENT."
        )
        log(
            "This may mean the model generated "
            "a textual tool call rather than "
            "Ollama returning a native "
            "tool_calls object."
        )

    log()
    log(
        "RAW RECONSTRUCTED ASSISTANT CONTENT:"
    )
    log(repr(content))


class OllamaProxyHandler(
    BaseHTTPRequestHandler
):
    protocol_version = "HTTP/1.1"

    def log_message(
        self,
        format_string: str,
        *args: object,
    ) -> None:
        # Suppress BaseHTTPRequestHandler's
        # normal access logging.
        pass

    def do_GET(self) -> None:
        self.forward_request()

    def do_POST(self) -> None:
        self.forward_request()

    def do_PUT(self) -> None:
        self.forward_request()

    def do_DELETE(self) -> None:
        self.forward_request()

    def do_PATCH(self) -> None:
        self.forward_request()

    def do_HEAD(self) -> None:
        self.forward_request()

    def forward_request(self) -> None:
        request_id = next_request_id()

        content_length = (
            self.headers.get(
                "Content-Length"
            )
        )

        body = b""

        if content_length is not None:
            try:
                length = int(
                    content_length
                )

            except ValueError:
                self.send_error(
                    400,
                    "Invalid Content-Length",
                )
                return

            body = self.rfile.read(
                length
            )

        #
        # Inspect and potentially modify
        # Ollama chat requests before forwarding.
        #
        if (
            self.path.startswith(
                "/api/chat"
            )
            and body
        ):
            try:
                request_data = (
                    json.loads(body)
                )

            except json.JSONDecodeError:
                request_data = None

            if isinstance(
                request_data,
                dict,
            ):
                injected = (
                    inject_saved_tools(
                        request_data
                    )
                )

                modified = injected

                if injected:
                    log()
                    log(
                        f"REQUEST #{request_id}: "
                        "INJECTED SAVED TOOLS"
                    )
                    log(
                        "INJECTED TOOL COUNT: "
                        f"{len(request_data['tools'])}"
                    )

                elif tool_injection_is_excluded(
                    request_data
                ):
                    replaced = (
                        apply_title_prompt_replacement(
                            request_data
                        )
                    )

                    if replaced:
                        modified = True

                        log()
                        log(
                            f"REQUEST #{request_id}: "
                            "REPLACED TITLE PROMPT"
                        )

                if modified:
                    body = json.dumps(
                        request_data,
                        ensure_ascii=False,
                        separators=(
                            ",",
                            ":",
                        ),
                    ).encode(
                        "utf-8"
                    )

            #
            # Log AFTER injection so the log
            # represents what is actually sent
            # to Ollama.
            #
            print_request(
                request_id,
                self.path,
                body,
            )

        else:
            log()
            log("=" * 80)
            log(
                f"REQUEST #{request_id}  "
                f"{timestamp()}"
            )
            log(
                f"{self.command} "
                f"{self.path}"
            )
            log("=" * 80)

            if body:
                try:
                    log(
                        pretty_json(
                            json.loads(body)
                        )
                    )

                except json.JSONDecodeError:
                    log(
                        body.decode(
                            "utf-8",
                            errors="replace",
                        )
                    )

        #
        # Forward the request to Ollama.
        #
        connection = HTTPConnection(
            OLLAMA_HOST,
            OLLAMA_PORT,
            timeout=600,
        )

        excluded_headers = {
            "host",
            "content-length",
            "connection",
            "transfer-encoding",
        }

        forward_headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower()
            not in excluded_headers
        }

        #
        # Content-Length is intentionally removed
        # because the request body may have changed.
        #
        forward_headers["Host"] = (
            f"{OLLAMA_HOST}:"
            f"{OLLAMA_PORT}"
        )

        try:
            connection.request(
                self.command,
                self.path,
                body=body,
                headers=forward_headers,
            )

            response = (
                connection.getresponse()
            )

            excluded_response_headers = {
                "connection",
                "transfer-encoding",
                "content-length",
            }

            self.send_response(
                response.status,
                response.reason,
            )

            for key, value in (
                response.getheaders()
            ):
                if (
                    key.lower()
                    not in excluded_response_headers
                ):
                    self.send_header(
                        key,
                        value,
                    )

            self.send_header(
                "Connection",
                "close",
            )

            self.end_headers()

            is_chat = (
                self.path.startswith(
                    "/api/chat"
                )
            )

            response_state: dict[
                str,
                object,
            ] = {
                "content": [],
                "tool_calls": [],
                "done": False,
            }

            #
            # Ollama streams /api/chat as
            # newline-delimited JSON.
            #
            while True:
                chunk = response.readline()

                if not chunk:
                    break

                #
                # Forward the original response
                # bytes unchanged.
                #
                self.wfile.write(
                    chunk
                )
                self.wfile.flush()

                if not is_chat:
                    continue

                try:
                    parsed = json.loads(
                        chunk
                    )

                except (
                    json.JSONDecodeError,
                    UnicodeDecodeError,
                ):
                    log(
                        f"RESPONSE #{request_id}: "
                        f"NON-JSON CHUNK: "
                        f"{chunk!r}"
                    )
                    continue

                if not isinstance(
                    parsed,
                    dict,
                ):
                    continue

                summarize_response_chunk(
                    parsed,
                    response_state,
                )

                message = parsed.get(
                    "message"
                )

                if not isinstance(
                    message,
                    dict,
                ):
                    continue

                tool_calls = message.get(
                    "tool_calls"
                )

                if (
                    isinstance(
                        tool_calls,
                        list,
                    )
                    and tool_calls
                ):
                    log()
                    log(
                        f"RESPONSE #{request_id}: "
                        "NATIVE TOOL CALL CHUNK"
                    )
                    log(
                        pretty_json(
                            tool_calls
                        )
                    )

                content = message.get(
                    "content"
                )

                if (
                    isinstance(
                        content,
                        str,
                    )
                    and content
                ):
                    log(
                        f"RESPONSE #{request_id}: "
                        f"CONTENT CHUNK: "
                        f"{content!r}"
                    )

            if is_chat:
                print_response(
                    request_id,
                    response_state,
                )

            log()
            log(
                f"REQUEST #{request_id} "
                "COMPLETE"
            )
            log()

        except Exception as exc:
            log()
            log(
                f"REQUEST #{request_id} "
                "PROXY ERROR:"
            )
            log(repr(exc))
            log()

            try:
                self.send_error(
                    502,
                    f"Proxy error: {exc}",
                )

            except Exception:
                pass

        finally:
            connection.close()


def main() -> None:
    server = ThreadingHTTPServer(
        (
            LISTEN_HOST,
            LISTEN_PORT,
        ),
        OllamaProxyHandler,
    )

    print()
    print("=" * 70)
    print("Ollama debugging proxy")
    print("=" * 70)
    print(
        f"Listen    : "
        f"http://{LISTEN_HOST}:"
        f"{LISTEN_PORT}"
    )
    print(
        f"Ollama    : "
        f"http://{OLLAMA_HOST}:"
        f"{OLLAMA_PORT}"
    )
    print(
        f"Log       : "
        f"{LOG_FILE}"
    )
    print(
        f"Logging   : "
        f"{ENABLE_LOGGING}"
    )
    print(
        f"Injection : "
        f"{ENABLE_TOOL_INJECTION}"
    )
    print(
        f"Exclude   : "
        f"{TOOL_INJECTION_EXCLUDE_PATTERN!r}"
    )
    print(
        f"Tools     : "
        f"{TOOLS_FILE}"
    )
    print()
    print(
        "Press Ctrl+C to stop."
    )
    print("=" * 70)
    print()

    try:
        server.serve_forever()

    except KeyboardInterrupt:
        print(
            "\nStopping proxy..."
        )

    finally:
        server.server_close()


if __name__ == "__main__":
    main()