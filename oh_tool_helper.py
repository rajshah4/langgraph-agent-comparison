"""Decorator helper for OpenHands SDK tools.

Provides a @simple_tool decorator that wraps plain Python functions
into the OpenHands SDK's Action/Observation/Executor/ToolDefinition
classes automatically — similar to LangGraph's @tool decorator.

Usage:

    from oh_tool_helper import simple_tool, tool_spec

    @simple_tool
    def get_schema() -> str:
        \"\"\"Get the database schema.\"\"\"
        return db.get_table_info()

    @simple_tool(read_only=False, destructive=True)
    def update_profile(field: str, new_value: str) -> str:
        \"\"\"Update a profile field.

        Args:
            field: The field to update.
            new_value: The new value.
        \"\"\"
        ...

    # Use with Agent:
    agent = Agent(llm=llm, tools=[tool_spec(get_schema), tool_spec(update_profile)])
"""

from __future__ import annotations

import inspect
import re
import sys
import types
from collections.abc import Sequence
from typing import Any, get_type_hints

from pydantic import Field, create_model

from openhands.sdk import Action, Observation
from openhands.sdk.tool import (
    Tool,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)

# Module reference for injecting generated classes at module level
_THIS_MODULE = sys.modules[__name__]


def _parse_docstring_args(docstring: str | None) -> dict[str, str]:
    """Extract parameter descriptions from a Google-style Args section."""
    if not docstring:
        return {}

    args: dict[str, str] = {}
    in_args = False
    current_name: str | None = None
    current_desc_lines: list[str] = []

    for line in docstring.split("\n"):
        stripped = line.strip()

        if stripped == "Args:":
            in_args = True
            continue

        if in_args:
            # End of Args section: blank line or non-indented line
            if not stripped and not current_name:
                break
            if stripped and not line[0].isspace():
                break

            # Check for a new "name: description" entry
            match = re.match(r"(\w+)(?:\s*\([^)]*\))?\s*:\s*(.*)", stripped)
            if match:
                # Save previous param
                if current_name:
                    args[current_name] = " ".join(current_desc_lines)
                current_name = match.group(1)
                current_desc_lines = [match.group(2)] if match.group(2) else []
            elif current_name and stripped:
                # Continuation line for the current parameter
                current_desc_lines.append(stripped)

    # Save the last parameter
    if current_name:
        args[current_name] = " ".join(current_desc_lines)

    return args


def _make_module_level_class(name: str, bases: tuple, body: dict) -> type:
    """Create a class that appears to be defined at module level.

    The OpenHands SDK's DiscriminatedUnionMixin rejects classes whose
    __qualname__ contains '<locals>' (closures). This helper creates
    classes via `type()` and patches __module__ and __qualname__ so
    they look like top-level definitions in *this* module.
    """
    cls = type(name, bases, body)
    cls.__module__ = _THIS_MODULE.__name__
    cls.__qualname__ = name
    # Also inject into module namespace for potential deserialization
    setattr(_THIS_MODULE, name, cls)
    return cls


def simple_tool(
    fn=None,
    *,
    title: str | None = None,
    read_only: bool = True,
    destructive: bool = False,
    idempotent: bool | None = None,
):
    """Create an OpenHands SDK tool from a plain Python function.

    Can be used as a bare decorator or with keyword arguments:

        @simple_tool
        def my_tool(param: str) -> str: ...

        @simple_tool(read_only=False, destructive=True)
        def my_tool(param: str) -> str: ...

    The function's docstring becomes the tool description (first line)
    and the Args section provides field descriptions.

    Returns the original function with ``_tool_name`` and ``_tool_title``
    attributes attached.
    """

    def _decorator(func):
        tool_title = title or func.__name__
        reg_name = f"_st_{func.__name__}"  # registration key

        # Unique class names derived from function name
        action_cls_name = f"ST_{func.__name__}_Action"
        obs_cls_name = f"ST_{func.__name__}_Observation"
        tool_cls_name = f"ST_{func.__name__}_ToolDef"
        executor_cls_name = f"ST_{func.__name__}_Executor"

        # ── Description from docstring ────────────────────────────────
        docstring = (func.__doc__ or func.__name__).strip()
        description = docstring.split("\n")[0]  # first line
        arg_descs = _parse_docstring_args(func.__doc__)

        # ── Build dynamic Action class from signature ─────────────────
        sig = inspect.signature(func)
        try:
            hints = get_type_hints(func)
        except Exception:
            hints = {}

        field_defs: dict[str, Any] = {}
        field_names: list[str] = []

        for name, param in sig.parameters.items():
            ann = hints.get(name, str)
            desc = arg_descs.get(name, name)
            if param.default is not inspect.Parameter.empty:
                field_defs[name] = (ann, Field(default=param.default, description=desc))
            else:
                field_defs[name] = (ann, Field(description=desc))
            field_names.append(name)

        # create_model sets __module__ and __qualname__ properly
        DynAction = create_model(
            action_cls_name,
            __base__=Action,
            __module__=_THIS_MODULE.__name__,
            **field_defs,
        )
        DynAction.__qualname__ = action_cls_name
        setattr(_THIS_MODULE, action_cls_name, DynAction)

        # ── Dynamic Observation (module-level) ────────────────────────
        DynObservation = _make_module_level_class(
            obs_cls_name, (Observation,), {}
        )

        # ── Dynamic Executor (closure OK — not a DiscriminatedUnionMixin) ──
        _fn = func  # capture in closure
        _fields = list(field_names)
        _obs_cls = DynObservation

        DynExecutor = _make_module_level_class(
            executor_cls_name,
            (ToolExecutor,),
            {
                "__call__": lambda self, action, conversation=None: (
                    _exec_tool(_fn, _fields, _obs_cls, action)
                ),
            },
        )

        # ── Dynamic ToolDefinition (module-level) ─────────────────────
        _idempotent = idempotent if idempotent is not None else read_only

        # Capture all closure vars for the factory function
        _desc = description
        _action = DynAction
        _obs = DynObservation
        _annotations = ToolAnnotations(
            title=tool_title,
            readOnlyHint=read_only,
            destructiveHint=destructive,
            idempotentHint=_idempotent,
        )

        def _tool_factory(conv_state=None, **kw):
            """Callable factory for register_tool."""
            return [
                ToolDefinition.__pydantic_init_subclass_complete_init__(
                    description=_desc,
                    action_type=_action,
                    observation_type=_obs,
                    annotations=_annotations,
                    executor=DynExecutor(),
                )
            ]

        # We use register_tool with a callable factory instead of a
        # ToolDefinition subclass.  This avoids DiscriminatedUnionMixin
        # serialization issues with dynamically created subclasses.
        register_tool(reg_name, _create_tool_class(
            tool_cls_name=tool_cls_name,
            func_name=func.__name__,
            description=_desc,
            action_cls=_action,
            obs_cls=_obs,
            annotations=_annotations,
            executor_cls=DynExecutor,
        ))

        # Attach metadata so callers can reference the tool
        func._tool_name = reg_name
        func._tool_title = tool_title
        return func

    # Support both @simple_tool and @simple_tool(...)
    if fn is not None:
        return _decorator(fn)
    return _decorator


def _exec_tool(fn, fields, obs_cls, action):
    """Execute a tool function and return an Observation."""
    kwargs = {n: getattr(action, n) for n in fields}
    try:
        result = fn(**kwargs)
        text = str(result) if result is not None else "Done."
        return obs_cls.from_text(text)
    except Exception as e:
        return obs_cls.from_text(f"Error: {e}", is_error=True)


def _create_tool_class(
    tool_cls_name: str,
    func_name: str,
    description: str,
    action_cls: type,
    obs_cls: type,
    annotations: ToolAnnotations,
    executor_cls: type,
) -> type:
    """Create a ToolDefinition subclass at module level.

    Uses types.new_class so that __init_subclass__ fires with the
    correct __name__ (no '<locals>' in __qualname__).
    """
    # Capture closure vars
    _desc = description
    _action = action_cls
    _obs = obs_cls
    _ann = annotations
    _exec_cls = executor_cls

    def _exec_body(ns):
        # Set 'name' in class dict so __init_subclass__ skips auto-naming
        ns["name"] = func_name

        @classmethod
        def create(cls, conv_state=None, **kw) -> Sequence[ToolDefinition]:
            return [
                cls(
                    description=_desc,
                    action_type=_action,
                    observation_type=_obs,
                    annotations=_ann,
                    executor=_exec_cls(),
                )
            ]

        ns["create"] = create

    DynTool = types.new_class(tool_cls_name, (ToolDefinition,), {}, _exec_body)
    DynTool.__module__ = _THIS_MODULE.__name__
    DynTool.__qualname__ = tool_cls_name
    setattr(_THIS_MODULE, tool_cls_name, DynTool)
    return DynTool


def tool_spec(func) -> Tool:
    """Get an OpenHands ``Tool`` spec for a ``@simple_tool``-decorated function.

    Usage:
        agent = Agent(llm=llm, tools=[tool_spec(get_schema), tool_spec(run_sql)])
    """
    name = getattr(func, "_tool_name", None)
    if name is None:
        raise ValueError(
            f"{func.__name__} is not decorated with @simple_tool"
        )
    return Tool(name=name)
