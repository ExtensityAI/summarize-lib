import json
import re
from typing import get_args, get_origin

import numpy as np
from loguru import logger
from pydantic import ValidationError
from symai.components import Function
from symai.models import LLMDataModel

############################################################################################################
# ValidatedFunction
############################################################################################################


class ValidatedFunction(Function):
    def __init__(
        self,
        data_model: LLMDataModel = None,
        retry_count=5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.retry_count = retry_count
        self.data_model = data_model
        self._last_quote_repair_replacements = 0

    def prepare_seeds(self, num_seeds: int, **kwargs):
        # get list of seeds for remedy (to avoid same remedy for same input)
        if "seed" in kwargs:
            seed = kwargs["seed"]
        elif hasattr(self, "seed"):
            seed = self.seed
        else:
            seed = 42

        rnd = np.random.RandomState(seed=seed)
        seeds = rnd.randint(
            0, np.iinfo(np.int16).max, size=num_seeds, dtype=np.int16
        ).tolist()
        return seeds

    def simplify_validation_errors(self, error: ValidationError) -> str:
        """
        Simplifies Pydantic validation errors into a concise, LLM-friendly format, including lists and nested elements.

        Args:
            error (ValidationError): The Pydantic ValidationError instance.

        Returns:
            str: A simplified and actionable error message.
        """
        simplified_errors = []
        for err in error.errors():
            # Build a human-readable field path
            field_path = " -> ".join(
                [str(element) for element in err["loc"]]
            )  # Includes indices for lists, keys, etc.
            message = err["msg"]  # Error message
            expected_type = err.get("type", "unknown")  # Expected type (if available)
            provided_value = err.get("ctx", {}).get(
                "given", "unknown"
            )  # Provided value (if available)

            # Create a concise, actionable error message
            error_message = (
                f"Field '{field_path}': {message}. "
                f"Expected type: {expected_type}. Provided value: {provided_value}."
            )
            simplified_errors.append(error_message)

        # Combine all errors into a single message
        return "\n".join(simplified_errors)

    def forward(self, *args, **kwargs):
        # force JSON mode
        kwargs["response_format"] = {"type": "json_object"}
        if "JSON" not in self.static_context:
            raise Exception("The static context must contain the string 'JSON'")

        # forward the function
        maybe_json = super().forward(*args, **kwargs)
        maybe_json = maybe_json.value

        # get list of seeds for remedy (to avoid same remedy for same input)
        remedy_seeds = self.prepare_seeds(self.retry_count, **kwargs)

        # prepare remedy function
        remedy_function = Function(
            """
            [Task]
            Fix the provided JSON string to ensure it is valid according to the schema and resolves all listed validation errors.

            [Important Guidelines]
            1. Only address the specific issues described in the validation errors.
            2. Preserve the meaning and values of the original JSON as much as possible unless changes are necessary for schema compliance.
            3. Ensure that the corrected JSON is both well-formatted and valid for the given schema.
            4. Return the corrected JSON string as the output.
            """,
            static_context="""
            You are tasked with fixing a string that is intended to be in **JSON format** but contains errors.
            Your goal is to correct the errors and ensure the JSON string is valid according to a given JSON schema.
            Follow these rules:

            1. Parse the provided string and use the list of validation errors to identify what needs to be fixed.
            2. Correct the identified errors to produce a properly formatted JSON string.
            3. Ensure the corrected JSON complies fully with the provided JSON schema.
            4. Preserve all original keys and values as much as possible. Only modify keys or values if they do not comply with the schema.
            5. Only modify the structure or values if necessary to meet the schema's requirements.
            6. Return the corrected JSON string as the output.

            [Requirements]
            - The output must be a valid, well-formatted JSON string.
            - Do not introduce new data or alter the intent of the original content unless required for schema compliance.
            - Ensure all changes are minimal and strictly necessary to fix the listed errors.
            """,
            response_format={"type": "json_object"},
        )

        # Ensure valid JSON is returned
        result = None
        last_error = ""
        for i in range(self.retry_count):
            try:
                # try to validate against provided data model
                result = self.data_model.model_validate_json(maybe_json, strict=True)
                break
            except ValidationError as e:
                quote_repaired_json = self._try_repair_unescaped_inner_quotes(
                    maybe_json, error=e
                )
                if quote_repaired_json is not None:
                    try:
                        result = self.data_model.model_validate_json(
                            quote_repaired_json, strict=True
                        )
                        logger.debug(
                            "Recovered invalid JSON via quote-escape repair; "
                            f"escaped_quotes={self._last_quote_repair_replacements}"
                        )
                        break
                    except ValidationError as quote_repair_error:
                        maybe_json = quote_repaired_json
                        e = quote_repair_error

                salvaged_json = self._try_salvage_single_string_json(maybe_json)
                if salvaged_json is not None:
                    try:
                        result = self.data_model.model_validate_json(
                            salvaged_json, strict=True
                        )
                        logger.debug(
                            "Recovered invalid JSON via single-field salvage."
                        )
                        break
                    except Exception:
                        result = None

                logger.debug(e)
                # collect and format error messages
                error_str = self.simplify_validation_errors(e)

                logger.debug(
                    f"[Retry {i + 1}/{self.retry_count}] ValidationError:\n{error_str}"
                )

                # adapt remedy function
                remedy_function.clear()
                remedy_function.adapt(f"[Original Input]\n```json\n{maybe_json}\n```\n")
                remedy_function.adapt(f"[Validation Errors]\n{error_str}\n")
                remedy_function.adapt(
                    f"[JSON Schema]\n{self.data_model.instruct_llm()}\n"
                )

                # apply remedy function
                maybe_json = remedy_function(seed=remedy_seeds[i])
                maybe_json = maybe_json.value

                # update last error for exception details
                last_error = error_str

        if result is None:
            quote_repaired_json = self._try_repair_unescaped_inner_quotes(maybe_json)
            if quote_repaired_json is not None:
                try:
                    result = self.data_model.model_validate_json(
                        quote_repaired_json, strict=True
                    )
                except Exception:
                    result = None

        if result is None:
            salvaged_json = self._try_salvage_single_string_json(maybe_json)
            if salvaged_json is not None:
                try:
                    result = self.data_model.model_validate_json(
                        salvaged_json, strict=True
                    )
                except Exception:
                    result = None

        if result is None:
            raise Exception(f"Failed to retrieve valid JSON: {last_error}")

        return result

    def _try_repair_unescaped_inner_quotes(
        self, maybe_json: str, error: ValidationError | None = None
    ) -> str | None:
        """Escape likely interior quotes inside JSON string content."""
        if not maybe_json:
            self._last_quote_repair_replacements = 0
            return None

        if error is not None:
            try:
                has_json_invalid = any(
                    err.get("type") == "json_invalid" for err in error.errors()
                )
            except Exception:
                has_json_invalid = False
            if not has_json_invalid:
                self._last_quote_repair_replacements = 0
                return None

        raw = maybe_json.strip()
        if not raw:
            self._last_quote_repair_replacements = 0
            return None

        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()

        open_brace = raw.find("{")
        close_brace = raw.rfind("}")
        if open_brace != -1 and close_brace != -1 and close_brace > open_brace:
            candidate = raw[open_brace : close_brace + 1]
        else:
            candidate = raw

        repaired, replacements = self._escape_interior_string_quotes(candidate)
        if replacements == 0:
            self._last_quote_repair_replacements = 0
            return None

        try:
            json.loads(repaired)
        except json.JSONDecodeError:
            self._last_quote_repair_replacements = 0
            return None

        self._last_quote_repair_replacements = replacements
        return repaired

    def _escape_interior_string_quotes(self, json_text: str) -> tuple[str, int]:
        """Escape quotes that are likely part of string content, not delimiters."""
        if not json_text:
            return json_text, 0

        out: list[str] = []
        in_string = False
        escape_active = False
        replacements = 0
        i = 0
        n = len(json_text)

        while i < n:
            ch = json_text[i]

            if not in_string:
                out.append(ch)
                if ch == '"':
                    in_string = True
                    escape_active = False
                i += 1
                continue

            if escape_active:
                out.append(ch)
                escape_active = False
                i += 1
                continue

            if ch == "\\":
                out.append(ch)
                escape_active = True
                i += 1
                continue

            if ch == '"':
                j = i + 1
                while j < n and json_text[j].isspace():
                    j += 1
                next_non_space = json_text[j] if j < n else ""

                if next_non_space in {",", "}", "]", ":", ""}:
                    out.append(ch)
                    in_string = False
                else:
                    out.append('\\"')
                    replacements += 1
                i += 1
                continue

            out.append(ch)
            i += 1

        return "".join(out), replacements

    def _try_salvage_single_string_json(self, maybe_json: str) -> str | None:
        """Coerce malformed JSON into a valid object for 1-field string schemas."""
        if self.data_model is None:
            return None

        try:
            model_fields = getattr(self.data_model, "model_fields", {})
            fields = [
                name
                for name, field in model_fields.items()
                if not getattr(field, "exclude", False)
            ]
        except Exception:
            return None

        if len(fields) != 1:
            return None

        field_name = fields[0]
        try:
            ann = self.data_model.model_fields[field_name].annotation
        except Exception:
            return None

        origin = get_origin(ann)
        if not (ann is str or (origin is not None and str in get_args(ann))):
            return None

        raw = (maybe_json or "").strip()
        if not raw:
            return None

        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()

        open_brace = raw.find("{")
        close_brace = raw.rfind("}")
        if open_brace != -1 and close_brace != -1 and close_brace > open_brace:
            candidate = raw[open_brace : close_brace + 1]
        else:
            candidate = raw

        value = None
        key_match = re.search(rf'"{re.escape(field_name)}"\s*:\s*"', candidate)
        if key_match:
            start = key_match.end()
            end_limit = candidate.rfind("}")
            if end_limit == -1:
                end_limit = len(candidate)

            end = candidate.rfind('"', start, end_limit)
            tail = candidate[end + 1 : end_limit] if end != -1 else ""
            has_terminal_quote = end > start and tail.strip() == ""
            if has_terminal_quote:
                value = candidate[start:end]
            else:
                value = candidate[start:end_limit]

        if value is None:
            value = candidate

        value = value.replace("\r\n", "\n").replace("\r", "\n")
        return json.dumps({field_name: value}, ensure_ascii=False)
