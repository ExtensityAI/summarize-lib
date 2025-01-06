from typing import Any, List
from symai.components import ExceptionWithUsage, LengthConstrainedFunction
from loguru import logger
from symai import Symbol
from pydantic import BaseModel, Field


class ResultValidator(LengthConstrainedFunction):
    def __init__(
        self,
        validation_retry_count: int = 5,
        *args,
        **kwargs,
    ):
        super().__init__(character_constraints=[], *args, **kwargs)
        self.validation_retry_count = validation_retry_count

    def validate(self, result) -> List[str]:
        # validation_criteria = {
        #     "Interview": "Does this summary identify different speakers and their key discussion points?",
        #     "Keynote": "Does this summary include speaker details, their expertise, and key messages?",
        #     "Scientific Paper": "Does this summary include methodology details and research findings?",
        #     "Report": "Does this summary include specific numerical results and statistics?",
        #     "Book": "Does this summary include character descriptions and their relationships?",
        #     "Presentation Slides": "Does this summary include the core idea and value proposition?"
        # }
        
        # # Get the validation prompt for this content type
        # if result.type in validation_criteria:
        #     validation_prompt = validation_criteria[result.type]
        # else:
        #     validation_prompt = "Is this a content summary?"
        
        # validation = Symbol(f"{validation_prompt} Return yes or no.\n{result.summary}").interpret()
        # is_valid = "yes" in validation.lower() or "true" in validation.lower()

        # print(f"Validation: {validation}")
        
        # # Return a list of validation errors (empty list if valid)
        # return [] if is_valid else [f"Content type '{result.type}' validation failed"]

        return []

    def forward(self, *args, **kwargs):
        result, usage = super().forward(*args, **kwargs)

        if self.validation_retry_count > 0:
            # save original task
            original_task = args[0]

            # get list of seeds for remedy (to avoid same remedy for same input)
            remedy_seeds = self.prepare_seeds(self.validation_retry_count, **kwargs)

            # validate the result
            for i in range(self.validation_retry_count):
                validation_errors = self.validate(result)

                if len(validation_errors) > 0:
                    for violation in validation_errors:
                        logger.info(f"Validation error: {violation}")
                        
                    logger.debug(str(result))

                    # build remedy task
                    remedy_task = self.wrap_task(
                        original_task, result.model_dump_json(), validation_errors
                    )

                    # attempt to remedy the result
                    kwargs["seed"] = remedy_seeds[i]
                    result, remedy_usage = super().forward(remedy_task, *args[1:], **kwargs)

                    # update local usage
                    usage.prompt_tokens += remedy_usage.prompt_tokens
                    usage.completion_tokens += remedy_usage.completion_tokens
                    usage.total_tokens += remedy_usage.total_tokens
                else:
                    break

            validation_errors = self.check_constraints(result)
            if i == self.validation_retry_count and len(validation_errors) > 0:
                raise ExceptionWithUsage(
                    f"Failed to enforce constraints: {' | '.join(validation_errors)}",
                    usage,
                )

        return result, usage

    def wrap_task(self, task: str, result: str, validation_errors: List[str]):
        joined_validation_errors = "\n".join(validation_errors)

        remedy_task = f"""
            You had the following task:

            [Original Task]
            {task}

            [Original Output]
            {result}

            However, the output has the following validation errors:

            [Validation Errors]
            {joined_validation_errors}

            [Task]
            Follow the origianl task but fix the validation errors.
            """

        return remedy_task
    
    @property
    def static_context(self):
        return (
            "You are an agent for validating 'JSON' schemas and fixing errors."
        )

class LLMDataModel(BaseModel):
    """
    A base class for Pydantic models that provides nicely formatted string output,
    suitable for LLM prompts, with support for nested models, lists, and optional section headers.
    """

    section_header: str = Field(
        default=None, exclude=True, frozen=True
    )  # Optional section header for top-level models

    def format_field(self, key: str, value: Any, indent: int = 0) -> str:
        """
        Formats a single field for output. Handles nested models, lists, and dictionaries.
        """
        indent_str = " " * indent
        if isinstance(value, LLMDataModel):
            # Nested model
            nested_str = value.__str__(indent + 2).strip()
            return f"{indent_str}{key}:\n{nested_str}" if key else nested_str
        elif isinstance(value, list):
            # List of items (handle nested models inside lists)
            formatted_items = "\n".join(
                [
                    f"{indent_str}  - {self.format_field('', item, indent).strip()}"
                    for item in value
                ]
            )
            return f"{indent_str}{key}:\n{formatted_items}" if key else formatted_items
        elif isinstance(value, dict):
            # Dictionary of key-value pairs
            formatted_items = "\n".join(
                [
                    f"{indent_str}  {k}: {self.format_field('', v, indent + 4).strip()}"
                    for k, v in value.items()
                ]
            )
            return f"{indent_str}{key}:\n{formatted_items}" if key else formatted_items
        else:
            # Primitive types
            return f"{indent_str}{key}: {value}" if key else f"{indent_str}{value}"

    def __str__(self, indent: int = 0) -> str:
        """
        Converts the model into a formatted string for LLM prompts.
        Handles indentation for nested models and includes an optional section header.
        """
        indent_str = " " * indent
        fields = "\n".join(
            self.format_field(name, getattr(self, name), indent + 2)
            for name, field in self.model_fields.items()
            if (
                getattr(self, name, None) is not None
                and not getattr(field, "exclude", False)
                and not name == "section_header"
            )  # Exclude None values and "exclude" fields
        )
        fields += "\n"  # add line break at the end to separate from the next section

        if self.section_header and indent == 0:
            header = f"{indent_str}[[{self.section_header}]]\n"
            return f"{header}{fields}"
        return fields