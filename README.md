# Summarize ‑ Hierarchical document summarization for SymbolicAI

`ExtensityAI/summarize-lib` is a SymbolicAI plug-in that turns **any long form content**
— books, papers, interviews, reports, slide decks, … — into a structured JSON
summary.  It is built around the `HierarchicalSummary` class which automatically
splits large documents into manageable chunks, summarises each chunk in
parallel and finally combines the partial results into a comprehensive summary
object.

This README explains

* how to install the plug-in,
* how to load it with SymbolicAI’s dynamic importer, and
* how to configure the `HierarchicalSummary` class.

---

## Installation

The plug-in is distributed through SymbolicAI’s package manager `sympkg`.

```bash
# make sure symbolicai is available in the current (virtual-)environment
pip install symbolicai   # or conda / poetry etc.

# install the plug-in
sympkg i ExtensityAI/summarize-lib
```

`sympkg` clones the repository into
`<env>/.symai/packages/ExtensityAI/summarize-lib`, so nothing touches your
global `site-packages`.

---

## Quick start

```python
from symai.extended import Import

# Load the class dynamically (no classical `pip install` needed)
HierarchicalSummary = Import.load_expression(
    "ExtensityAI/summarize-lib",
    "HierarchicalSummary",
)

# Either pass a path / http(s) link …
summarizer = HierarchicalSummary(
    file_link="./my_long_report.pdf",
    # optional parameters ↓
    document_lang="EN",           # force target language
    min_num_chunks=5,
)

# …or raw text that you already have in memory
# summarizer = HierarchicalSummary(content=big_text_str, document_name="report.txt")

result = summarizer()  # executes the full summarization pipeline

print(result.summary)  # the full summary text
print(result.facts)    # → list[str]
print(result.quotes)   # → list[str] | None
```

The returned object is a `pydantic` `LLMDataModel`; you can therefore treat it
like any other data-class and call `result.model_dump()` to obtain the raw
Python dictionary.

---

## `HierarchicalSummary` constructor

Below is a concise description of all public arguments.  Only **one** of
`file_link` **or** `content` must be supplied.

| argument | type | default | description |
|----------|------|---------|-------------|
| `file_link` | `str \| None` | `None` | Local path or `http(s)` URL to a file (PDF, DOCX, TXT, …). Mutually exclusive with `content`. |
| `content` | `str \| None` | `None` | Raw text that is already available in memory. Mutually exclusive with `file_link`. |
| `document_name` | `str \| None` | `None` | Human readable identifier that will be embedded in the prompt. Required when using `content=`. If omitted with `file_link`, the file name is used. |
| `document_lang` | `str \| None` | `None` | ISO-639 based language string (e.g. `"EN"`, `"DE"`). Forces the output language. When omitted the language is auto-detected from the first chunk. |
| `asset_name` | `str \| None` | `None` | Convenience alias that is mapped to `document_name` if the latter is not supplied. |
| `data_model` | `type[LLMDataModel]` | `Summary` | Custom Pydantic model that defines the JSON schema you expect from the LLM. Advanced use-cases only. |
| `min_num_chunks` | `int` | `5` | Minimum number of chunks the document will be split into. Controls the trade-off between context size per chunk and amount of parallel calls. |
| `min_chunk_size` | `int` | `250` | Lower bound (in tokens) for an individual chunk. |
| `max_chunk_size` | `int` | `1000` | Upper bound (in tokens) for an individual chunk. |
| `max_output_tokens` | `int` | `10000` | Hard upper limit for the total size of the produced summary. If the combined summary is bigger a second compression pass is executed automatically. |
| `user_prompt` | `str \| None` | `None` | Extra instructions targeted at your specific use-case. They are appended verbatim to the system prompt under the section *Goal-specific Instructions*. |
| `include_quotes` | `bool` | `False` | When `True` the model will try to populate the `quotes` field with verbatim quotations from the source text. |
| `tokenizer_name` | `str` | `"gpt2"` | Name of the HuggingFace tokenizer that is used for accurate token counting. Any model that is compatible with `tokenizers` works (e.g. `"Xenova/gpt-4o"`). |
| `chunker_name` | `str` | `"RecursiveChunker"` | Chunking strategy; the default recursively splits the text along `\n\n`, `.`, ` `, … boundaries until the target size is reached. |
| `seed` | `int` | `42` | Seed used for sampling as well as for the internal JSON-repair routine. |

### Document type specific prompts

`HierarchicalSummary` automatically classifies the input into one of the
`DocumentType` enum values (book, scientific_paper, interview, …).  Depending
on the detected type an additional prompt snippet is injected to steer the LLM
towards type-aware behaviour (e.g. *extract title, authors, methodology* for
papers, or *highlight key statistics* for reports).  You do **not** have to do
anything; the classification is fully automatic but you can still override the
detection afterwards by editing `result.type` if required.

---

## Advanced: Custom output schema

Sometimes the default `Summary` schema is not enough.  Because
`HierarchicalSummary` accepts any `pydantic.BaseModel` subclass you can pass
your own structure and the plug-in will ensure (via an internal
`ValidatedFunction`) that the LLM’s response always conforms to it.

```python
from pydantic import BaseModel, Field

class MySchema(BaseModel):
    overview: str = Field(..., description="Short, high-level summary")
    bullets: list[str] = Field(..., description="Key points for slide decks")

summarizer = HierarchicalSummary(
    file_link="talk.pdf",
    data_model=MySchema,
)

result = summarizer()  # -> MySchema instance
```

---

## Under the hood (very short)

1. The document is **downloaded or read from disk** and wrapped in a
   `[[DOCUMENT::NAME]]` tag so that file artefacts do not pollute the prompt.
2. Token length is measured with the selected tokenizer – if the text is too
   large it is split into chunks (respecting the min/max knobs).
3. Each chunk is summarised **in parallel** (async + `tenacity` retries) and
   validated against the `data_model`.
4. The partial JSONs are merged (`gather()` helper) and, if necessary, passed
   through additional compression rounds until `max_output_tokens` is met.

All of that is hidden behind the single call `HierarchicalSummary(...)()` so
that you can focus on what really matters: the content.

---

## License

See `LICENSE` for full details.  In short: the code is MIT; documents you feed
into the summarizer stay **yours**.
