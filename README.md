# GreenThumb Customer Care Agent

Assistente AI per il servizio clienti di **GreenThumb Marketplace** (e-commerce fittizio di giardinaggio).
Gestisce in autonomia le richieste di primo livello (stato ordini, prodotti, consigli agronomici, resi e
policy) e riconosce quando passare la conversazione a un operatore umano.

- **Agente ReAct** scritto esplicitamente con LangChain; tutte le chiamate al modello passano da **LiteLLM**
  (`langchain-litellm`), quindi il provider si cambia con una variabile d'ambiente.
- **4 tool**: `search_catalog` e `search_knowledge_base` (RAG su Chroma), `get_order_status`, `escalate_to_human`.
- **Memoria a breve termine**: *summarization* (default) oppure *trimming*, con soglie configurabili.
- **Memoria a lungo termine semantica**: knowledge base di 41 documenti indicizzata su Chroma.
- **Structured output** validato con Pydantic: `answer`, `confidence`, `sources`, `needs_human`.
  `confidence` e `needs_human` sono calcolati con regole deterministiche; `sources` contiene solo
  documenti realmente recuperati.
- **Valutazione** su 23 conversazioni con RAGAS (faithfulness, answer relevancy, context recall) e
  DeepEval (GEval correctness, tool correctness), più controlli deterministici.
- **API FastAPI** in locale (`POST /chat`).

La relazione tecnica completa è in [REPORT.md](REPORT.md); il notebook di consegna è
[notebooks/GreenThumb_Customer_Care_Agent.ipynb](notebooks/GreenThumb_Customer_Care_Agent.ipynb).

## Struttura

```
greenthumb/            codice dell'agente
  config.py            impostazioni da variabili d'ambiente
  llm.py               chat model ed embeddings via LiteLLM
  knowledge_base.py    caricamento, chunking, indicizzazione e retrieval (memoria a lungo termine)
  orders.py            lettura degli ordini
  escalation.py        ticket per gli operatori
  tools.py             i 4 tool LangChain con schemi Pydantic
  memory.py            memoria a breve termine (summarization / trimming)
  prompts.py           prompt di sistema, di output finale e di riassunto
  policy.py            regole deterministiche per sources, confidence, needs_human
  schemas.py           modelli Pydantic (contratto API, output strutturato, trace)
  agent.py             loop ReAct
  api.py               applicazione FastAPI
data/
  knowledge_base/      41 documenti Markdown (products/, guides/, policies/, faq/)
  orders.json          24 ordini fittizi
evaluation/
  eval_dataset.json    23 conversazioni di test (anche multi-turno)
  run_evaluation.py    pipeline RAGAS + DeepEval
  results/             output e punteggi
scripts/               generazione ordini, indicizzazione, chat da terminale
tests/                 test offline (modello simulato, nessuna chiave richiesta)
notebooks/             notebook Colab di consegna
```

## Avvio rapido (locale)

Requisiti: Python 3.11+ e una chiave OpenAI (o di un altro provider supportato da LiteLLM).

```bash
python -m venv .venv
```

```bash
.venv/Scripts/activate      # Windows (Git Bash) - su Linux/macOS: source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

```bash
cp .env.example .env        # poi inserisci OPENAI_API_KEY nel file .env
```

```bash
python scripts/build_index.py
```

```bash
uvicorn greenthumb.api:app --port 8000
```

La documentazione interattiva è su http://localhost:8000/docs.

### Esempio di chiamata

```bash
curl -s -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"message\": \"Quando arriva l'ordine 1042? Posso piantare adesso i bulbi di tulipano?\"}"
```

Risposta (esempio):

```json
{
  "answer": "L'ordine 1042 è stato spedito con BRT e arriverà domani, 17/06/2026. I tulipani non vanno piantati adesso: si piantano in autunno, da metà ottobre a fine novembre. Fino ad allora conserva i bulbi in un luogo fresco, buio e asciutto.",
  "confidence": "high",
  "sources": ["guides/bulbi_autunnali.md"],
  "needs_human": false
}
```

Per continuare la stessa conversazione (memoria a breve termine) rimanda l'header `X-Session-Id`
ricevuto come campo `session_id` del body. `POST /chat/debug` restituisce anche la trace ReAct completa.

### Altri comandi

```bash
python -m pytest tests
```

```bash
python scripts/chat_cli.py
```

```bash
python evaluation/run_evaluation.py
```

## Configurazione

Tutte le impostazioni sono variabili d'ambiente (vedi [.env.example](.env.example)); nessuna chiave è
scritta nel codice. Le principali:

| Variabile | Default | Significato |
|---|---|---|
| `GREENTHUMB_LLM_MODEL` | `openai/gpt-4o-mini` | modello dell'agente (sintassi LiteLLM `provider/modello`) |
| `GREENTHUMB_EMBEDDING_MODEL` | `openai/text-embedding-3-small` | modello di embedding |
| `GREENTHUMB_EVAL_MODEL` | `openai/gpt-4o-mini` | modello giudice di RAGAS e DeepEval |
| `GREENTHUMB_CURRENT_DATE` | `2026-06-16` | "oggi" simulato, coerente con `orders.json` |
| `GREENTHUMB_MEMORY_STRATEGY` | `summarization` | `summarization` oppure `trimming` |
| `GREENTHUMB_MEMORY_MAX_TOKENS` | `1200` | soglia oltre cui la memoria viene compattata |
| `GREENTHUMB_RETRIEVAL_TOP_K` | `4` | chunk restituiti per ricerca |
| `GREENTHUMB_HIGH_CONFIDENCE_SCORE` | `0.50` | score minimo delle fonti per `confidence = high` |

I dati sono fittizi: `orders.json` si rigenera con `python scripts/generate_orders.py --reference-date AAAA-MM-GG`.
