import sys
from pathlib import Path

from pydantic import BaseModel, ValidationError

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.functions import ValidatedFunction


class _TextPayload(BaseModel):
    text: str


class _NumericPayload(BaseModel):
    num: int


class _OptionalTextPayload(BaseModel):
    text: str | None = None


def _vf_for(data_model: type[BaseModel]) -> ValidatedFunction:
    vf = ValidatedFunction.__new__(ValidatedFunction)
    vf.data_model = data_model
    vf.retry_count = 5
    vf._last_quote_repair_replacements = 0
    return vf


def test_quote_repair_handles_unescaped_dialogue_quote_in_field_text():
    vf = _vf_for(_TextPayload)
    broken = '{"text":"... Frage: „Wer entscheidet hier final?" Plötzlich ..."}'

    repaired = vf._try_repair_unescaped_inner_quotes(broken)

    assert repaired is not None
    parsed = _TextPayload.model_validate_json(repaired, strict=True)
    assert parsed.text == '... Frage: „Wer entscheidet hier final?" Plötzlich ...'


def test_quote_repair_handles_unescaped_quote_before_following_sentence():
    vf = _vf_for(_TextPayload)
    broken = '{"text":"... aussprechen - „Sagen Sie mehr dazu." Diese ..."}'

    repaired = vf._try_repair_unescaped_inner_quotes(broken)

    assert repaired is not None
    parsed = _TextPayload.model_validate_json(repaired, strict=True)
    assert parsed.text == '... aussprechen - „Sagen Sie mehr dazu." Diese ...'


def test_quote_repair_skips_valid_json_with_already_escaped_quotes():
    vf = _vf_for(_TextPayload)
    valid = '{"text":"He said \\"hello\\" and left."}'

    repaired = vf._try_repair_unescaped_inner_quotes(valid)

    assert repaired is None


def test_quote_repair_is_gated_to_json_invalid_errors():
    vf = _vf_for(_NumericPayload)
    candidate = '{"num":"abc"}'

    try:
        _NumericPayload.model_validate_json(candidate, strict=True)
    except ValidationError as err:
        repaired = vf._try_repair_unescaped_inner_quotes(candidate, error=err)
    else:  # pragma: no cover
        raise AssertionError("Expected ValidationError for invalid strict integer JSON")

    assert repaired is None


def test_single_field_salvage_handles_optional_string_schema():
    vf = _vf_for(_OptionalTextPayload)
    broken = '{"text":"He said "hello" and left.'

    salvaged = vf._try_salvage_single_string_json(broken)

    assert salvaged is not None
    parsed = _OptionalTextPayload.model_validate_json(salvaged, strict=True)
    assert parsed.text == 'He said "hello" and left.'
