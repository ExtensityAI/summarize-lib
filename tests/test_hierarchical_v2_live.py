import ast
import json
import os
import sys
from pathlib import Path
from typing import Any, List, Optional

import pytest
from pydantic import Field
from symai.models import LLMDataModel

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.hierarchical_v2 import HierarchicalSummaryV2


RUN_LIVE = os.getenv("RUN_LIVE_SUMMARIZE_TESTS") == "1"


class LiveExtractionSchema(LLMDataModel):
    overview: str = Field(description="Coherent overview of the source content.")
    facts: List[str] = Field(description="Detailed factual statements from the source.")
    anecdotes: List[str] = Field(description="Stories or narrative examples from the source, if available.")
    insights: List[str] = Field(description="Insights or interpretations supported by the source.")
    quotes: Optional[List[str]] = Field(
        default=None,
        description="Verbatim quotes from the source, only if clearly present.",
    )


def _default_storyone_assets_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "storyone" / "assets" / "unit_tests" / "files"


def _default_storyone_repo_root() -> Path:
    return Path(__file__).resolve().parents[2] / "storyone"


def _resolve_test_file(relative_path: str) -> Path:
    root = Path(os.getenv("STORYONE_TEST_ASSET_ROOT", str(_default_storyone_assets_dir())))
    path = root / relative_path
    if not path.exists():
        pytest.skip(f"Test file not found: {path}")
    return path


def _load_storyone_assetsummary_class() -> type[LLMDataModel]:
    storyone_root = Path(os.getenv("STORYONE_REPO_ROOT", str(_default_storyone_repo_root())))
    types_path = storyone_root / "src" / "book" / "components" / "types.py"
    if not types_path.exists():
        pytest.skip(f"storyone types.py not found: {types_path}")

    source = types_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(types_path))

    class_defs: dict[str, ast.ClassDef] = {
        node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
    }
    if "AssetSummary" not in class_defs:
        pytest.skip("AssetSummary class not found in storyone types.py")

    def _collect_ann_names(annotation: ast.AST) -> set[str]:
        names: set[str] = set()
        for node in ast.walk(annotation):
            if isinstance(node, ast.Name):
                names.add(node.id)
        return names

    needed: set[str] = {"AssetSummary"}
    queue = ["AssetSummary"]
    while queue:
        class_name = queue.pop(0)
        class_node = class_defs[class_name]
        for stmt in class_node.body:
            if isinstance(stmt, ast.AnnAssign) and stmt.annotation is not None:
                for dep in _collect_ann_names(stmt.annotation):
                    if dep in class_defs and dep not in needed:
                        needed.add(dep)
                        queue.append(dep)

    selected_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in needed
    ]

    isolated_module = ast.Module(body=selected_nodes, type_ignores=[])
    ast.fix_missing_locations(isolated_module)
    code = compile(isolated_module, filename=str(types_path), mode="exec")

    namespace: dict[str, Any] = {
        "LLMDataModel": LLMDataModel,
        "Field": Field,
        "List": List,
        "Optional": Optional,
    }
    exec(code, namespace, namespace)
    return namespace["AssetSummary"]


def _item_to_text(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if hasattr(item, "model_dump"):
        payload = item.model_dump(mode="json")
    elif isinstance(item, dict):
        payload = item
    else:
        return str(item).strip()

    for key in ("fact", "quote", "summary", "text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return json.dumps(payload, ensure_ascii=False, sort_keys=True).strip()


@pytest.mark.live
@pytest.mark.skipif(not RUN_LIVE, reason="Set RUN_LIVE_SUMMARIZE_TESTS=1 to run live LLM tests")
def test_hierarchical_v2_live_storyone_assetsummary():
    AssetSummary = _load_storyone_assetsummary_class()
    file_path = _resolve_test_file(
        "239ec949-f02f-4c4d-a86f-078ee94ea215/Interview_Robert_HADZETOVIC_auf_Medianet.docx"
    )

    summarizer = HierarchicalSummaryV2(
        file_link=str(file_path),
        data_model=AssetSummary,
        min_num_chunks=4,
        min_chunk_size=700,
        max_chunk_size=2200,
        max_output_tokens=1200,
        user_prompt="Extract rich material for book writing with concrete details.",
        include_quotes=True,
        tokenizer_name="gpt2",
        chunker_name="SemanticHybridChunker",
    )

    result = summarizer()

    assert isinstance(result.summary, str) and result.summary.strip()
    assert isinstance(result.facts, list) and len(result.facts) >= 2
    assert all(_item_to_text(x) for x in result.facts)

    usage = summarizer.usage_report
    assert usage["runtime"]["total_seconds"] > 0
    assert usage["llm"]["calls_total"] > 0
    assert usage["chunking"]["chunks"] >= 1


@pytest.mark.live
@pytest.mark.skipif(not RUN_LIVE, reason="Set RUN_LIVE_SUMMARIZE_TESTS=1 to run live LLM tests")
def test_hierarchical_v2_live_semantic_usage_tracking():
    file_path = _resolve_test_file(
        "7c7b407b-d67d-4e57-bc37-39c9f86f7738/ChocolateCognitiveFunction_Chocolate.pdf"
    )

    summarizer = HierarchicalSummaryV2(
        file_link=str(file_path),
        data_model=LiveExtractionSchema,
        min_num_chunks=4,
        min_chunk_size=700,
        max_chunk_size=2200,
        max_output_tokens=1100,
        user_prompt="Extract detailed source-grounded insights and facts.",
        include_quotes=True,
        tokenizer_name="gpt2",
        chunker_name="SemanticHybridChunker",
    )

    result = summarizer()
    assert isinstance(result.overview, str) and result.overview.strip()
    assert isinstance(result.facts, list) and len(result.facts) >= 1

    usage = summarizer.usage_report
    assert usage["chunking"]["strategy"] in {"semantic_hybrid", "structural", "recursive_fallback"}
    assert usage["embedding"]["enabled"] is True
    assert usage["embedding"]["requests"] >= 1
    assert usage["embedding"]["items"] >= 1
    assert usage["embedding"]["input_tokens"] >= 1
    assert usage["embedding"]["failed_requests"] >= 0
