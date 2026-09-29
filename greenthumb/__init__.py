"""GreenThumb Marketplace customer-care agent (ReAct + RAG, LangChain + LiteLLM)."""

import os

# LiteLLM downloads its model cost map at import time; the bundled copy is enough
# and avoids network/SSL issues behind proxies. Must be set before importing litellm.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # Chroma telemetry

try:
    # Use the operating system's certificate store for HTTPS. Needed on machines where an
    # antivirus or corporate proxy inspects TLS with its own root certificate, which the
    # certifi bundle used by Python does not know; harmless elsewhere.
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

__version__ = "1.0.0"
