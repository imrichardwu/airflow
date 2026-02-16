# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Serialized Dag and BaseOperator."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

from airflow._shared.module_loading import qualname
from airflow._shared.secrets_masker import redact
from airflow.configuration import conf
from airflow.settings import json

if TYPE_CHECKING:
    from airflow.partition_mapper.base import PartitionMapper
    from airflow.timetables.base import Timetable as CoreTimetable


def _truncate_rendered_value(rendered: str, max_length: int) -> str:
    MIN_CONTENT_LENGTH = 7

    if max_length <= 0:
        return ""

    prefix = "Truncated. You can change this behaviour in [core]max_templated_field_length. "
    suffix = "..."
    value = rendered

    # Always prioritize showing the truncation message first
    trunc_only = f"{prefix}{suffix}"

    # If max_length is too small to even show the message, return it anyway
    # (message takes priority over the constraint)
    if max_length < len(trunc_only):
        return trunc_only

    # Check if value already has outer quotes - if so, preserve them and don't add extra quotes
    has_outer_quotes = (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    )

    if has_outer_quotes:
        # Preserve existing quote character and strip outer quotes to get inner content
        quote_char = value[0]
        content = value[1:-1]
    else:
        # Choose quote character: use double quotes if value contains single quotes,
        # otherwise use single quotes
        if "'" in value and '"' not in value:
            quote_char = '"'
        else:
            quote_char = "'"
        content = value

    # Calculate overhead: prefix + opening quote + closing quote + suffix
    overhead = len(prefix) + 2 + len(suffix)
    available = max_length - overhead

    # Only show content if there's meaningful space for it
    if available < MIN_CONTENT_LENGTH:
        return trunc_only

    # Get content and trim trailing spaces
    content = content[:available].rstrip()

    # Build the result and ensure it doesn't exceed max_length
    result = f"{prefix}{quote_char}{content}{quote_char}{suffix}"

    # Trim content to ensure result < max_length, with a small buffer when possible
    target_length = max_length - 1
    while len(result) > target_length and len(content) > 0:
        content = content[:-1].rstrip()
        result = f"{prefix}{quote_char}{content}{quote_char}{suffix}"

    return result


def serialize_template_field(template_field: Any, name: str) -> str | dict | list | int | float:
    """
    Return a serializable representation of the templated field.

    If ``templated_field`` is provided via a callable then
    return the following serialized value: ``<callable full_qualified_name>``

    If ``templated_field`` contains a class or instance that requires recursive
    templating, store them as strings. Otherwise simply return the field as-is.
    """

    def is_jsonable(x):
        try:
            json.dumps(x)
        except (TypeError, OverflowError):
            return False
        else:
            return True

    def translate_tuples_to_lists(obj: Any):
        """Recursively convert tuples to lists."""
        if isinstance(obj, tuple):
            return [translate_tuples_to_lists(item) for item in obj]
        if isinstance(obj, list):
            return [translate_tuples_to_lists(item) for item in obj]
        if isinstance(obj, dict):
            return {key: translate_tuples_to_lists(value) for key, value in obj.items()}
        return obj

    def sort_dict_recursively(obj: Any) -> Any:
        """Recursively sort dictionaries to ensure consistent ordering."""
        if isinstance(obj, dict):
            return {k: sort_dict_recursively(v) for k, v in sorted(obj.items())}
        if isinstance(obj, list):
            return [sort_dict_recursively(item) for item in obj]
        if isinstance(obj, tuple):
            return tuple(sort_dict_recursively(item) for item in obj)
        return obj

    max_length = conf.getint("core", "max_templated_field_length")

    if not is_jsonable(template_field):
        try:
            serialized = template_field.serialize()
        except AttributeError:
            if callable(template_field):
                full_qualified_name = qualname(template_field, True)
                serialized = f"<callable {full_qualified_name}>"
            else:
                serialized = str(template_field)
        if len(serialized) > max_length:
            rendered = redact(serialized, name)
            return _truncate_rendered_value(str(rendered), max_length)
        return serialized
    if not template_field and not isinstance(template_field, tuple):
        # Avoid unnecessary serialization steps for empty fields unless they are tuples
        # and need to be converted to lists
        return template_field
    template_field = translate_tuples_to_lists(template_field)
    # Sort dictionaries recursively to ensure consistent string representation
    # This prevents hash inconsistencies when dict ordering varies
    if isinstance(template_field, dict):
        template_field = sort_dict_recursively(template_field)
    serialized = str(template_field)
    if len(serialized) > max_length:
        rendered = redact(serialized, name)
        return _truncate_rendered_value(str(rendered), max_length)
    return template_field


class TimetableNotRegistered(ValueError):
    """When an unregistered timetable is being accessed."""

    def __init__(self, type_string: str) -> None:
        self.type_string = type_string

    def __str__(self) -> str:
        return (
            f"Timetable class {self.type_string!r} is not registered or "
            "you have a top level database access that disrupted the session. "
            "Please check the airflow best practices documentation."
        )


def find_registered_custom_timetable(importable_string: str) -> type[CoreTimetable]:
    """Find a user-defined custom timetable class registered via a plugin."""
    from airflow import plugins_manager

    timetable_classes = plugins_manager.get_timetables_plugins()
    with contextlib.suppress(KeyError):
        return timetable_classes[importable_string]
    raise TimetableNotRegistered(importable_string)


def find_registered_custom_partition_mapper(importable_string: str) -> type[PartitionMapper]:
    """Find a user-defined custom partition mapper class registered via a plugin."""
    from airflow import plugins_manager

    partition_mapper_cls = plugins_manager.get_partition_mapper_plugins()
    with contextlib.suppress(KeyError):
        return partition_mapper_cls[importable_string]
    raise PartitionMapperNotFound(importable_string)


def is_core_timetable_import_path(importable_string: str) -> bool:
    """Whether an importable string points to a core timetable class."""
    return importable_string.startswith("airflow.timetables.")


class PartitionMapperNotFound(ValueError):
    """Raise when a PartitionMapper cannot be found."""

    def __init__(self, type_string: str) -> None:
        self.type_string = type_string

    def __str__(self) -> str:
        return (
            f"PartitionMapper class {self.type_string!r} could not be imported or "
            "you have a top level database access that disrupted the session. "
            "Please check the airflow best practices documentation."
        )


def is_core_partition_mapper_import_path(importable_string: str) -> bool:
    """Whether an importable string points to a core partition mapper class."""
    return importable_string.startswith("airflow.partition_mapper.")
