import json
import os
import sys

PROTECTED_DIRS = ("pina", "tests", "docs")
FILE_KEYS = ("file_path", "filePath", "notebook_path", "notebookPath", "path", "file")


def normalize(path):
    return path.replace("\\", "/").rstrip("/")


def relative_to_project(path, root):
    path = normalize(path)
    root = normalize(root)
    if path.lower().startswith(root.lower() + "/"):
        path = path[len(root) + 1 :]
    elif path.startswith("/") or (len(path) > 1 and path[1] == ":"):
        return None
    if path.startswith("./"):
        path = path[2:]
    return path


def is_protected(path, root):
    relative = relative_to_project(path, root)
    if not relative:
        return False
    return any(relative == d or relative.startswith(f"{d}/") for d in PROTECTED_DIRS)


def input_paths(tool_input):
    for key in FILE_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str):
            yield value
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    yield item


def main():
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    tool_input = payload.get("tool_input") or {}
    if not isinstance(tool_input, dict):
        return 0

    root = os.environ.get("CLAUDE_PROJECT_DIR") or payload.get("cwd") or os.getcwd()

    for path in input_paths(tool_input):
        if is_protected(path, root):
            relative = relative_to_project(path, root)
            print(
                json.dumps(
                    {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "ask",
                            "permissionDecisionReason": (
                                f"Modifying source file {relative}. "
                                "Approve this change before continuing."
                            ),
                        }
                    }
                )
            )
            return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
