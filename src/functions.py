from .lazy_imports import (
    lazy_numpy, lazy_loguru, lazy_validation_error,
    lazy_function, lazy_llm_data_model, get_logger, get_random_state, get_int16_max
)

############################################################################################################
# ValidatedFunction
############################################################################################################


class ValidatedFunction:
    """ValidatedFunction class with lazy inheritance."""

    def __init__(
        self,
        data_model=None,
        retry_count=5,
        *args,
        **kwargs,
    ):
        # Lazy import of base class
        Function = lazy_function()
        LLMDataModel = lazy_llm_data_model()

        # Validate data_model type
        if data_model is not None and not issubclass(data_model, LLMDataModel):
            raise TypeError("data_model must be a subclass of LLMDataModel")

        # Create base instance for delegation
        self._base_instance = Function.__new__(Function)
        Function.__init__(self._base_instance, *args, **kwargs)

        # Copy base class attributes
        for attr_name in dir(self._base_instance):
            if not attr_name.startswith('_') and not hasattr(self, attr_name):
                setattr(self, attr_name, getattr(self._base_instance, attr_name))

        self.retry_count = retry_count
        self.data_model = data_model

    def __getattr__(self, name):
        """Delegate to base instance for missing attributes."""
        if name == '_base_instance':
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        if '_base_instance' in self.__dict__:
            return getattr(self.__dict__['_base_instance'], name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        """Delegate to base instance for certain attributes."""
        if name == '_base_instance':
            super().__setattr__(name, value)
            return
        if '_base_instance' in self.__dict__ and hasattr(self.__dict__['_base_instance'], name):
            setattr(self.__dict__['_base_instance'], name, value)
        else:
            super().__setattr__(name, value)

    def prepare_seeds(self, num_seeds: int, **kwargs):
        # get list of seeds for remedy (to avoid same remedy for same input)
        if "seed" in kwargs:
            seed = kwargs["seed"]
        elif hasattr(self, "seed"):
            seed = self.seed
        else:
            seed = 42

        rnd = get_random_state(seed)
        seeds = rnd.randint(
            0, get_int16_max(), size=num_seeds, dtype='int16'
        ).tolist()
        return seeds

    def simplify_validation_errors(self, error) -> str:
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
        Function = lazy_function()
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
        ValidationError = lazy_validation_error()
        for i in range(self.retry_count):
            try:
                # try to validate against provided data model
                result = self.data_model.model_validate_json(maybe_json, strict=True)
                break
            except ValidationError as e:
                logger = get_logger()
                logger.debug(e)
                # collect and format error messages
                error_str = self.simplify_validation_errors(e)

                logger.debug(
                    f"[Retry {i + 1}/{self.retry_count}] ValidationError:\n{error_str}"
                )

                # adapt remedy function
                remedy_function.clear()
                remedy_function.adapt(f"[Original Input]\n```json\n{maybe_json}\n´´´\n")
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
            raise Exception(f"Failed to retrieve valid JSON: {last_error}")

        return result

