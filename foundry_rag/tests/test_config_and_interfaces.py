"""
tests/test_config_and_interfaces.py — Tests for config and interface contracts.

Run with:
    pytest foundry_rag/tests/test_config_and_interfaces.py -v
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone

from foundry_rag.config import FoundryConfig, ConfigError
from foundry_rag.interfaces.embedder import BaseEmbedder, EmbeddingError
from foundry_rag.interfaces.vector_store import (
    BaseVectorStore, SearchRequest, SearchResult, VectorStoreError
)
from foundry_rag.interfaces.pii_and_connector import (
    BasePIIScrubber, ScrubResult, PIIScrubError,
    BaseConnector, RawDocument, ConnectorError
)


# ===========================================================================
# Config tests
# ===========================================================================

REQUIRED_ENV = {
    "AZURE_TENANT_ID": "test-tenant",
    "AZURE_CLIENT_ID": "test-client",
    "AZURE_CLIENT_SECRET": "test-secret",
    "AZURE_SUBSCRIPTION_ID": "test-sub",
    "AZURE_RESOURCE_GROUP": "test-rg",
    "AZURE_FOUNDRY_ENDPOINT": "https://test.services.ai.azure.com",
    "AZURE_FOUNDRY_PROJECT": "test-project",
    "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "text-embedding-3-large",
    "AZURE_SEARCH_ENDPOINT": "https://test.search.windows.net",
    "AZURE_SEARCH_KEY": "test-search-key",
    "AZURE_SEARCH_INDEX_NAME": "test-index",
}


class TestFoundryConfig:
    def test_loads_from_env(self, monkeypatch):
        for k, v in REQUIRED_ENV.items():
            monkeypatch.setenv(k, v)
        config = FoundryConfig.from_env()
        assert config.tenant_id == "test-tenant"
        assert config.search_index_name == "test-index"
        assert config.openai_embedding_deployment == "text-embedding-3-large"

    def test_raises_on_missing_required(self, monkeypatch):
        for k in REQUIRED_ENV:
            monkeypatch.delenv(k, raising=False)
        with pytest.raises(ConfigError) as exc:
            FoundryConfig.from_env()
        error_msg = str(exc.value)
        # All missing vars reported at once
        assert "AZURE_TENANT_ID" in error_msg
        assert "AZURE_SEARCH_KEY" in error_msg
        assert "AZURE_FOUNDRY_ENDPOINT" in error_msg

    def test_raises_on_single_missing_required(self, monkeypatch):
        for k, v in REQUIRED_ENV.items():
            monkeypatch.setenv(k, v)
        monkeypatch.delenv("AZURE_SEARCH_KEY")
        with pytest.raises(ConfigError) as exc:
            FoundryConfig.from_env()
        assert "AZURE_SEARCH_KEY" in str(exc.value)

    def test_optional_fields_have_defaults(self, monkeypatch):
        for k, v in REQUIRED_ENV.items():
            monkeypatch.setenv(k, v)
        config = FoundryConfig.from_env()
        assert config.pii_enabled is True
        assert config.max_chunk_tokens == 512
        assert config.overlap_ratio == 0.10
        assert config.embedding_batch_size == 100
        assert config.language_endpoint == ""

    def test_optional_fields_load_from_env(self, monkeypatch):
        for k, v in REQUIRED_ENV.items():
            monkeypatch.setenv(k, v)
        monkeypatch.setenv("CONFLUENCE_URL", "https://acme.atlassian.net")
        monkeypatch.setenv("CONFLUENCE_TOKEN", "token-abc")
        monkeypatch.setenv("FOUNDRY_RAG_MAX_CHUNK_TOKENS", "1024")
        monkeypatch.setenv("FOUNDRY_RAG_PII_ENABLED", "false")
        config = FoundryConfig.from_env()
        assert config.confluence_url == "https://acme.atlassian.net"
        assert config.max_chunk_tokens == 1024
        assert config.pii_enabled is False

    def test_has_pii_scrubbing_false_without_language(self, monkeypatch):
        for k, v in REQUIRED_ENV.items():
            monkeypatch.setenv(k, v)
        config = FoundryConfig.from_env()
        assert config.has_pii_scrubbing() is False

    def test_has_pii_scrubbing_true_with_language(self, monkeypatch):
        for k, v in REQUIRED_ENV.items():
            monkeypatch.setenv(k, v)
        monkeypatch.setenv("AZURE_LANGUAGE_ENDPOINT", "https://test.cognitiveservices.azure.com")
        monkeypatch.setenv("AZURE_LANGUAGE_KEY", "test-lang-key")
        config = FoundryConfig.from_env()
        assert config.has_pii_scrubbing() is True

    def test_enabled_connectors_empty_by_default(self, monkeypatch):
        for k, v in REQUIRED_ENV.items():
            monkeypatch.setenv(k, v)
        config = FoundryConfig.from_env()
        assert config.enabled_connectors() == []

    def test_enabled_connectors_detects_configured(self, monkeypatch):
        for k, v in REQUIRED_ENV.items():
            monkeypatch.setenv(k, v)
        monkeypatch.setenv("CONFLUENCE_URL", "https://acme.atlassian.net")
        monkeypatch.setenv("CONFLUENCE_TOKEN", "tok")
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
        config = FoundryConfig.from_env()
        connectors = config.enabled_connectors()
        assert "confluence" in connectors
        assert "slack" in connectors

    def test_summary_never_exposes_secrets(self, monkeypatch):
        for k, v in REQUIRED_ENV.items():
            monkeypatch.setenv(k, v)
        config = FoundryConfig.from_env()
        summary = config.summary()
        assert "test-secret" not in summary
        assert "test-search-key" not in summary
        assert "test-project" in summary
        assert "test-index" in summary

    def test_invalid_int_raises_config_error(self, monkeypatch):
        for k, v in REQUIRED_ENV.items():
            monkeypatch.setenv(k, v)
        monkeypatch.setenv("FOUNDRY_RAG_MAX_CHUNK_TOKENS", "not-a-number")
        with pytest.raises(ConfigError, match="must be an integer"):
            FoundryConfig.from_env()


# ===========================================================================
# Interface contract tests — verify implementations will satisfy contracts
# ===========================================================================

class ConcreteEmbedder(BaseEmbedder):
    """Minimal concrete implementation for contract testing."""
    async def embed(self, texts):
        return [[0.1] * 3072 for _ in texts]
    async def embed_one(self, text):
        return [0.1] * 3072
    @property
    def dimensions(self):
        return 3072
    @property
    def model_name(self):
        return "text-embedding-3-large"


class TestEmbedderInterface:
    def test_concrete_satisfies_interface(self):
        embedder = ConcreteEmbedder()
        assert isinstance(embedder, BaseEmbedder)

    @pytest.mark.asyncio
    async def test_embed_returns_correct_shape(self):
        embedder = ConcreteEmbedder()
        texts = ["hello world", "foo bar", "baz qux"]
        result = await embedder.embed(texts)
        assert len(result) == 3
        assert all(len(v) == 3072 for v in result)

    @pytest.mark.asyncio
    async def test_embed_one_returns_vector(self):
        embedder = ConcreteEmbedder()
        result = await embedder.embed_one("test text")
        assert len(result) == 3072

    def test_dimensions_property(self):
        assert ConcreteEmbedder().dimensions == 3072

    def test_model_name_property(self):
        assert ConcreteEmbedder().model_name == "text-embedding-3-large"

    def test_embedding_error_retryable_by_default(self):
        err = EmbeddingError("rate limit exceeded")
        assert err.retryable is True

    def test_embedding_error_not_retryable(self):
        err = EmbeddingError("invalid input", retryable=False)
        assert err.retryable is False


class ConcreteVectorStore(BaseVectorStore):
    """Minimal concrete implementation for contract testing."""
    def __init__(self):
        self._docs = {}
        self._exists = False

    async def upsert(self, documents):
        for doc in documents:
            self._docs[doc["id"]] = doc
        return len(documents)

    async def search(self, request):
        return [SearchResult(
            chunk_id="c1", text="test", score=0.95,
            doc_type="pdf", source_id="doc-1"
        )]

    async def delete(self, chunk_ids):
        count = sum(1 for cid in chunk_ids if cid in self._docs)
        for cid in chunk_ids:
            self._docs.pop(cid, None)
        return count

    async def create_index(self):
        self._exists = True

    async def index_exists(self):
        return self._exists

    async def document_count(self):
        return len(self._docs)


class TestVectorStoreInterface:
    @pytest.mark.asyncio
    async def test_upsert_returns_count(self):
        store = ConcreteVectorStore()
        docs = [{"id": f"c{i}", "text": f"doc {i}"} for i in range(5)]
        count = await store.upsert(docs)
        assert count == 5

    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        store = ConcreteVectorStore()
        request = SearchRequest(query_vector=[0.1] * 3072, top_k=5)
        results = await store.search(request)
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    @pytest.mark.asyncio
    async def test_create_index_idempotent(self):
        store = ConcreteVectorStore()
        await store.create_index()
        await store.create_index()  # second call must not raise
        assert await store.index_exists() is True

    @pytest.mark.asyncio
    async def test_delete_returns_count(self):
        store = ConcreteVectorStore()
        await store.upsert([{"id": "c1", "text": "hello"}, {"id": "c2", "text": "world"}])
        deleted = await store.delete(["c1"])
        assert deleted == 1
        assert await store.document_count() == 1

    def test_search_request_defaults(self):
        req = SearchRequest(query_vector=[0.1] * 10)
        assert req.top_k == 5
        assert req.min_score == 0.0
        assert req.filters == {}
        assert req.acl_groups == []


class TestPIIScrubberInterface:
    def test_scrub_result_fields(self):
        result = ScrubResult(
            text="Hello [PERSON]",
            entities_found=["PERSON"],
            was_modified=True,
        )
        assert result.was_modified is True
        assert "PERSON" in result.entities_found

    def test_pii_scrub_error_message(self):
        err = PIIScrubError("service unavailable")
        assert "service unavailable" in str(err)


class TestConnectorInterface:
    def test_raw_document_defaults(self):
        doc = RawDocument(
            source_id="test::doc::001",
            content=b"PDF bytes",
            content_type="application/pdf",
            filename="test.pdf",
        )
        assert doc.acl_groups == []
        assert doc.extra_metadata == {}
        assert isinstance(doc.created_at, datetime)

    def test_connector_error_message(self):
        err = ConnectorError("Confluence", "401 Unauthorized")
        assert "Confluence" in str(err)
        assert "401" in str(err)
