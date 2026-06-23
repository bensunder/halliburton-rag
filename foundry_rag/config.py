"""
config.py — Central configuration for foundry-rag.

Single source of truth for all environment variables.
Every Azure service reads its endpoint from here — never from os.environ directly.

Usage:
    from foundry_rag.config import FoundryConfig
    config = FoundryConfig.from_env()

    # All fields validated on construction — fail fast, not mid-pipeline
    print(config.search_endpoint)
    print(config.embedding_deployment)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


class ConfigError(Exception):
    """Raised when required environment variables are missing or invalid."""
    pass


@dataclass
class FoundryConfig:
    """
    Complete configuration for a foundry-rag deployment.

    Required fields must be set — construction fails with a clear error
    listing every missing variable rather than failing on the first one.

    Optional fields have sensible defaults that work for most Azure deployments.
    """

    # ------------------------------------------------------------------
    # Azure identity — required
    # ------------------------------------------------------------------
    tenant_id: str
    client_id: str
    client_secret: str
    subscription_id: str
    resource_group: str

    # ------------------------------------------------------------------
    # Azure AI Foundry project — required
    # ------------------------------------------------------------------
    foundry_endpoint: str
    foundry_project: str

    # ------------------------------------------------------------------
    # Azure OpenAI — required for embedding
    # ------------------------------------------------------------------
    openai_endpoint: str
    openai_embedding_deployment: str

    # ------------------------------------------------------------------
    # Azure AI Search — required for vector store
    # ------------------------------------------------------------------
    search_endpoint: str
    search_key: str
    search_index_name: str

    # ------------------------------------------------------------------
    # Optional — Azure AI Language (PII scrubbing)
    # ------------------------------------------------------------------
    language_endpoint: str = ""
    language_key: str = ""

    # ------------------------------------------------------------------
    # Optional — Azure Blob Storage (file connector)
    # ------------------------------------------------------------------
    storage_connection_string: str = ""
    storage_container_name: str = "foundry-rag-documents"

    # ------------------------------------------------------------------
    # Optional — Azure Key Vault
    # ------------------------------------------------------------------
    keyvault_url: str = ""

    # ------------------------------------------------------------------
    # Optional — Connector credentials
    # ------------------------------------------------------------------
    confluence_url: str = ""
    confluence_token: str = ""
    slack_bot_token: str = ""
    sql_connection_string: str = ""
    git_repo_url: str = ""
    sap_export_container: str = ""

    # ------------------------------------------------------------------
    # Optional — Pipeline tuning
    # ------------------------------------------------------------------
    openai_api_version: str = "2024-02-01"
    embedding_batch_size: int = 100
    max_chunk_tokens: int = 512
    overlap_ratio: float = 0.10
    min_chunk_chars: int = 50
    pii_enabled: bool = True
    dead_letter_container: str = "foundry-rag-dlq"

    @classmethod
    def from_env(cls) -> "FoundryConfig":
        """
        Construct config from environment variables.
        Collects ALL missing required variables before raising —
        operators see every problem at once, not one at a time.
        """
        required = {
            "tenant_id":                    "AZURE_TENANT_ID",
            "client_id":                    "AZURE_CLIENT_ID",
            "client_secret":                "AZURE_CLIENT_SECRET",
            "subscription_id":              "AZURE_SUBSCRIPTION_ID",
            "resource_group":               "AZURE_RESOURCE_GROUP",
            "foundry_endpoint":             "AZURE_FOUNDRY_ENDPOINT",
            "foundry_project":              "AZURE_FOUNDRY_PROJECT",
            "openai_endpoint":              "AZURE_OPENAI_ENDPOINT",
            "openai_embedding_deployment":  "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
            "search_endpoint":              "AZURE_SEARCH_ENDPOINT",
            "search_key":                   "AZURE_SEARCH_KEY",
            "search_index_name":            "AZURE_SEARCH_INDEX_NAME",
        }

        optional = {
            "language_endpoint":            "AZURE_LANGUAGE_ENDPOINT",
            "language_key":                 "AZURE_LANGUAGE_KEY",
            "storage_connection_string":    "AZURE_STORAGE_CONNECTION_STRING",
            "storage_container_name":       "AZURE_STORAGE_CONTAINER_NAME",
            "keyvault_url":                 "AZURE_KEYVAULT_URL",
            "confluence_url":               "CONFLUENCE_URL",
            "confluence_token":             "CONFLUENCE_TOKEN",
            "slack_bot_token":              "SLACK_BOT_TOKEN",
            "sql_connection_string":        "SQL_CONNECTION_STRING",
            "git_repo_url":                 "GIT_REPO_URL",
            "sap_export_container":         "SAP_EXPORT_CONTAINER",
            "openai_api_version":           "AZURE_OPENAI_API_VERSION",
        }

        bool_optional = {
            "pii_enabled":                  "FOUNDRY_RAG_PII_ENABLED",
        }

        int_optional = {
            "embedding_batch_size":         "FOUNDRY_RAG_EMBEDDING_BATCH_SIZE",
            "max_chunk_tokens":             "FOUNDRY_RAG_MAX_CHUNK_TOKENS",
            "min_chunk_chars":              "FOUNDRY_RAG_MIN_CHUNK_CHARS",
        }

        float_optional = {
            "overlap_ratio":                "FOUNDRY_RAG_OVERLAP_RATIO",
        }

        # Collect all missing required vars before raising
        missing: list[str] = []
        values: dict[str, Any] = {}

        for field_name, env_var in required.items():
            val = os.environ.get(env_var, "").strip()
            if not val:
                missing.append(env_var)
            else:
                values[field_name] = val

        if missing:
            raise ConfigError(
                f"Missing required environment variables:\n"
                + "\n".join(f"  {v}" for v in missing)
                + "\n\nCopy .env.example to .env and fill in all required values."
            )

        # Optional string fields
        for field_name, env_var in optional.items():
            val = os.environ.get(env_var, "").strip()
            if val:
                values[field_name] = val

        # Optional bool fields
        for field_name, env_var in bool_optional.items():
            val = os.environ.get(env_var, "").strip().lower()
            if val in ("false", "0", "no"):
                values[field_name] = False
            elif val in ("true", "1", "yes"):
                values[field_name] = True

        # Optional int fields
        for field_name, env_var in int_optional.items():
            val = os.environ.get(env_var, "").strip()
            if val:
                try:
                    values[field_name] = int(val)
                except ValueError:
                    raise ConfigError(f"{env_var} must be an integer, got: {val!r}")

        # Optional float fields
        for field_name, env_var in float_optional.items():
            val = os.environ.get(env_var, "").strip()
            if val:
                try:
                    values[field_name] = float(val)
                except ValueError:
                    raise ConfigError(f"{env_var} must be a float, got: {val!r}")

        return cls(**values)

    def has_pii_scrubbing(self) -> bool:
        """True if Azure AI Language is configured and PII scrubbing is enabled."""
        return bool(self.language_endpoint and self.language_key and self.pii_enabled)

    def has_blob_storage(self) -> bool:
        """True if Azure Blob Storage is configured."""
        return bool(self.storage_connection_string)

    def has_keyvault(self) -> bool:
        """True if Azure Key Vault is configured."""
        return bool(self.keyvault_url)

    def enabled_connectors(self) -> list[str]:
        """Return list of connector names that have credentials configured."""
        connectors = []
        if self.storage_connection_string:
            connectors.append("blob_storage")
        if self.confluence_url and self.confluence_token:
            connectors.append("confluence")
        if self.slack_bot_token:
            connectors.append("slack")
        if self.sql_connection_string:
            connectors.append("sql")
        if self.git_repo_url:
            connectors.append("git")
        if self.sap_export_container:
            connectors.append("sap_csv")
        return connectors

    def summary(self) -> str:
        """Human-readable config summary for startup logging. Never logs secrets."""
        lines = [
            "foundry-rag configuration",
            f"  Foundry project:    {self.foundry_project}",
            f"  Foundry endpoint:   {self.foundry_endpoint}",
            f"  Search index:       {self.search_index_name}",
            f"  Embedding model:    {self.openai_embedding_deployment}",
            f"  PII scrubbing:      {'enabled' if self.has_pii_scrubbing() else 'disabled'}",
            f"  Blob storage:       {'configured' if self.has_blob_storage() else 'not configured'}",
            f"  Key Vault:          {'configured' if self.has_keyvault() else 'not configured'}",
            f"  Enabled connectors: {', '.join(self.enabled_connectors()) or 'none'}",
            f"  Max chunk tokens:   {self.max_chunk_tokens}",
            f"  Overlap ratio:      {self.overlap_ratio}",
        ]
        return "\n".join(lines)
