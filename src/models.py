"""Data models for hierarchical summarization."""
from typing import List, Optional

from pydantic import Field
from symai.models import LLMDataModel


class Summary(LLMDataModel):
    """Summary data model with comprehensive document summary, facts, and quotes."""
    summary: str = Field(
        description="An extremely comprehensive summary of the document. Do not start with 'This document is about...' or similar phrases."
    )
    facts: List[str] = Field(
        description="Important facts and subjects extracted from the document."
    )
    quotes: Optional[List[str]] = Field(
        default=None,
        description="Significant quotes extracted from the document **verbatim** if there are any.",
    )
    type: Optional[str] = None

    def validate():
        # TODO: validate that quotes are verbatim from the document
        pass


def gather(chunks: List[LLMDataModel]):
    """Aggregate chunks by concatenating string fields and extending list fields.

    Args:
        chunks: List of LLMDataModel instances to aggregate

    Returns:
        Dictionary with aggregated field values
    """
    res_dict = {}
    type_dict = {
        list: {"default": list, "func": "append"},
        str: {"default": str, "func": "concatenate"},
    }
    for chunk in chunks:
        chunk_fields = chunk.model_fields
        for field_name, field_type in chunk_fields.items():
            field = getattr(chunk, field_name)
            if type(field) in type_dict and not field_type.exclude:
                _type = type_dict[type(field)]
                # setup field
                if field_name not in res_dict:
                    res_dict[field_name] = type(field)()

                # append or concatenate
                if _type["func"] == "append":
                    res_dict[field_name].extend(field)
                elif _type["func"] == "concatenate":
                    res_dict[field_name] += field + "\n"

    return res_dict

