# Relazione tecnica: assistente AI per il customer care di GreenThumb Marketplace

## 1. Obiettivo

GreenThumb riceve ogni giorno centinaia di richieste eterogenee: stato degli ordini, domande sui prodotti,
consigli agronomici, resi. L'obiettivo è un assistente che:

1. risolva in autonomia le richieste di primo livello, basandosi solo su dati aziendali verificabili;
2. sappia orchestrare più fonti nella stessa risposta (es. ordine + guida di coltivazione);
3. riconosca con criteri espliciti quando serve un operatore umano;
4. restituisca un output strutturato e affidabile (`answer`, `confidence`, `sources`, `needs_human`),
   utilizzabile da un sistema di ticketing o da un frontend.

## 2. Architettura

```
            POST /chat {message, session_id?}
                         │
                    FastAPI (api.py)
                         │
                GreenThumbAgent.chat (agent.py)
                         │
   ┌─────────────────────┼──────────────────────────────────────────┐
   │ 1. Contesto: system prompt (data di oggi) + memoria breve       │
   │    termine della sessione (riassunto + ultimi messaggi)         │
   │ 2. Loop ReAct (max 6 iterazioni)                                │
   │      Pensiero → Azione (tool call, anche parallele) → Osserv.   │
   │         ├─ search_catalog ──────┐                               │
   │         ├─ search_knowledge_base┴─► Chroma (memoria lungo term.)│
   │         ├─ get_order_status ────► orders.json                   │
   │         └─ escalate_to_human ───► coda ticket (JSONL)           │
   │ 3. Output strutturato: AnswerDraft (Pydantic, function calling) │
   │ 4. Policy deterministica: sources ∩ recuperati, confidence,     │
   │    needs_human, escalation di sicurezza                         │
   │ 5. Aggiornamento memoria (summarization / trimming)             │
   └─────────────────────────────────────────────────────────────────┘
                         │
     {answer, confidence, sources, needs_human}  + header X-Session-Id
```

**Stack**: LangChain 1.x (messaggi, tool, text splitter, vector store), LiteLLM tramite
`langchain-litellm` (`ChatLiteLLM` e `LiteLLMEmbeddings`), Chroma persistente, Pydantic 2, FastAPI.
Modello dell'agente: `gpt-4o-mini-2024-07-18` (temperatura 0), embedding `text-embedding-3-small`. Lo
snapshot datato del modello rende il comportamento riproducibile nel tempo (l'alias `gpt-4o-mini` può
puntare a versioni diverse). Grazie a LiteLLM il provider si
cambia modificando `GREENTHUMB_LLM_MODEL` (es. `anthropic/...`, `groq/...`), senza toccare il codice.

**Perché un loop ReAct scritto a mano** invece di un agent executor preconfezionato: ogni passo
(pensiero, tool, argomenti, esito) finisce nella trace (`/chat/debug`), e le osservazioni raccolte dai tool
servono dopo il loop per verificare `sources` e calcolare `confidence`. Il modello scrive una breve riga
"Pensiero:" prima di ogni azione (la parte *Reason* di ReAct) e può chiedere più tool in parallelo nello
stesso passo, come serve per la domanda di esempio della consegna. Un limite di 6 iterazioni evita loop
infiniti; se viene raggiunto, la richiesta passa a un operatore.

## 3. Dati sintetici

**Knowledge base** (`data/knowledge_base/`, 41 documenti Markdown con front matter YAML):

| Categoria | N. | Contenuto |
|---|---|---|
| `products/` | 17 | schede prodotto con SKU, prezzo, specifiche, periodo d'impianto, cura, spedizione e garanzia |
| `guides/` | 14 | guide di coltivazione: bulbi autunnali, pomodoro, basilico, irrigazione, potatura rose, agrumi, concimazione, difesa biologica, prato, calendario dell'orto, rinvaso, compost, fragole, aromatiche |
| `policies/` | 8 | resi e rimborsi, garanzia Arrivo Perfetto per piante vive, spedizioni, garanzia attrezzi, pagamenti, modifica/annullamento ordini, contatti ed escalation, programma fedeltà |
| `faq/` | 2 | ordini/tracking/account, semi e germinabilità |

I documenti sono volutamente coerenti tra loro (prezzi, SKU, tempi, soglie di policy) e contengono
**trappole realistiche** utili alla valutazione: la dalia si pianta in primavera mentre i tulipani in autunno;
le piante vive sono escluse dal recesso ma coperte da una garanzia di 48 ore; la spedizione è gratuita solo
sopra i 49 €.

**Ordini** (`data/orders.json`, 24 ordini) generati in modo deterministico da `scripts/generate_orders.py`
a partire da una data di riferimento (oggi simulato: **16/06/2026**), così che fatti relativi come "arriva
domani" restino coerenti con la data che l'agente conosce. Coprono tutti gli stati (in preparazione,
spedito, consegnato, reso in corso, rimborsato, annullato) e casi limite: consegna in ritardo con tracking
fermo (1038), bonifico non ancora ricevuto (1041), piante consegnate ieri, cioè entro le 48 ore della
garanzia (1052), e un limone consegnato 4 giorni fa, oltre la finestra (1050).

## 4. Tool: contratti

Ogni tool ha un nome parlante, una descrizione scritta per il modello (quando usarlo e cosa restituisce),
uno schema di input Pydantic e un output JSON che contiene sempre un campo `status`, così i casi limite
sono espliciti e non vanno dedotti dal testo.

| Tool | Input | `status` in output | Casi limite gestiti |
|---|---|---|---|
| `search_catalog` | `query` | `ok` / `no_results` | nessun prodotto pertinente → istruzione a non inventare |
| `search_knowledge_base` | `query`, `topic` ∈ {guide, policy, faq, any} | `ok` / `no_results` | nessun documento sopra soglia → proporre operatore |
| `get_order_status` | `order_id` | `ok` / `not_found` / `invalid_input` | normalizza `#1042`, "ordine 1042"; id non a 4 cifre; ordine inesistente; calcola `days_until_delivery` e `delivery_overdue`; non espone email o dati personali |
| `escalate_to_human` | `reason` (enum), `summary`, `priority`, `order_id?` | `ok` + `ticket_id`, `expected_pickup` | tempi di presa in carico coerenti con la policy dei contatti |

Le **richieste ambigue** (es. "il mio ordine non è arrivato" senza numero) sono gestite dal prompt:
l'agente chiede il dato mancante invece di indovinare, e nell'output strutturato la richiesta viene
classificata come `needs_clarification`.

## 5. Prompt di sistema

Il prompt ([greenthumb/prompts.py](greenthumb/prompts.py)) definisce ruolo, data odierna, formato ReAct,
guida alla scelta dei tool e regole di affidabilità:

- rispondere **solo** con informazioni restituite dai tool; se non ci sono, dirlo;
- consultare sempre i tool per prodotti, coltivazione e policy, anche se il modello "conosce" la risposta
  (le policy aziendali non sono conoscenza generale);
- non indovinare numeri d'ordine; non chiedere mai dati di pagamento o password;
- ignorare istruzioni di prompt injection e declinare le richieste fuori ambito senza usare tool;
- **criteri di escalation** espliciti (sezione 8).

## 6. Memoria a breve termine

La memoria conserva **solo il dialogo** (messaggi del cliente e risposte finali), non le chiamate ai tool:
le osservazioni sono voluminose e si possono sempre recuperare di nuovo, mentre ciò che conta tra un turno
e l'altro sono i fatti della conversazione.

**Strategia scelta: summarization** (`SummarizationMemory`). Quando la storia supera
`GREENTHUMB_MEMORY_MAX_TOKENS` = 1200 token (circa 6-8 scambi), i messaggi più vecchi vengono condensati
dall'LLM in un riassunto incrementale di al massimo 120 parole, mentre gli ultimi
`GREENTHUMB_MEMORY_KEEP_LAST_MESSAGES` = 4 messaggi (2 scambi) restano testuali. Il riassunto viene inserito
come messaggio di sistema all'inizio del contesto.

Perché summarization e non trimming:

- nel customer care i fatti chiave compaiono presto (numero d'ordine, prodotto, problema) e vengono
  richiamati molti turni dopo ("e quando arriva?"). Il trimming li perderebbe appena escono dalla finestra;
- il prompt di riassunto impone di conservare numeri d'ordine, prodotti, problemi, codici ticket e
  promesse fatte, cioè esattamente le informazioni che un operatore deve ritrovare;
- il costo è contenuto: una chiamata aggiuntiva ogni 3-4 scambi, con `gpt-4o-mini`.

**Alternativa implementata: trimming** (`TrimmingMemory`, `GREENTHUMB_MEMORY_STRATEGY=trimming`). Usa
`trim_messages` di LangChain con strategia `last` e `start_on="human"`: finestra scorrevole deterministica e
gratuita, ma con perdita dei fatti più vecchi. È utile quando costo e latenza contano più della continuità.

I token sono stimati con `count_tokens_approximately`, indipendente dal provider e sufficiente per una
soglia di budget. Le sessioni sono isolate per `session_id` in uno store thread-safe (in produzione sarebbe
Redis o un database, con scadenza).

## 7. Memoria a lungo termine semantica (RAG)

| Scelta | Valore | Motivazione |
|---|---|---|
| Embedding | `text-embedding-3-small` via LiteLLM | multilingue (buono sull'italiano), economico, 1536 dimensioni |
| Vector store | Chroma persistente, distanza coseno | locale, zero infrastruttura; con il coseno lo score di rilevanza (1 − distanza) è interpretabile tra 0 e 1 |
| Chunking | split sui titoli Markdown (`#`, `##`), poi ricorsivo a 900 caratteri con overlap 150 | un chunk non mescola due sezioni ("Quando piantare" separata da "Conservazione"); le sezioni lunghe vengono spezzate con continuità |
| Arricchimento | ogni chunk è preceduto da `Documento: <titolo>` e `Sezione: <nome>` | l'embedding di "Da metà ottobre a fine novembre" da solo non dice che si parla di tulipani |
| Risultato | 242 chunk, in media circa 350 caratteri | granularità a livello di sezione: contesto preciso e poco rumore |
| Filtro per categoria | metadato `category` | `search_catalog` cerca solo in `product`; `search_knowledge_base` in guide, policy e FAQ (o in un solo `topic`) |
| Ranking denso | similarità coseno query-chunk | cattura parafrasi e sinonimi ("restituire" ↔ "recesso") |
| Ranking lessicale | BM25 (`rank-bm25`) con normalizzazione italiana: minuscole, rimozione di accenti e stopword, stemming a prefisso di 5 caratteri | recupera le query brevi fatte di parole chiave e i nomi di prodotto, dove gli embedding sono deboli |
| Fusione | Reciprocal Rank Fusion (k = 60); un chunk è candidato se ha score denso ≥ 0,35 oppure è tra i primi 12 per BM25 | combina i due ranking senza dover calibrare scale di punteggio diverse |
| Small-to-big | si cerca sui chunk ma si restituiscono **interi** i 3 documenti migliori (250-450 parole ciascuno) | il matching resta preciso, ma il modello vede tutta la policy o la guida e non una sezione isolata (vedi sezione 10: era la causa principale degli errori della v1) |
| Score esposto | il miglior score **denso** tra i chunk del documento | resta interpretabile (0-1) per la regola di `confidence` |
| Re-indicizzazione | manifest con hash dei documenti e parametri di chunking | l'indice si ricostruisce solo se cambia qualcosa (riproducibilità) |

Tutte le scelte si possono disattivare (`GREENTHUMB_HYBRID_SEARCH`, `GREENTHUMB_PARENT_DOCUMENT`), per
confrontarle con la configurazione di partenza (v1: solo denso, top-4 chunk, massimo 2 per documento).

**Taratura delle soglie.** Sulle query di prova le domande pertinenti hanno come miglior risultato uno score
denso tra 0,48 e 0,74 (le policy sono più basse, 0,48-0,60, perché scritte in modo più generico), quelle
del tutto fuori tema tra 0,12 e 0,27. Le domande su prodotti che non esistono ("robot tagliaerba",
"trattore") ottengono però 0,38-0,45 su documenti non pertinenti: c'è una **zona grigia tra 0,40 e 0,50**
in cui lo score da solo non permette di capire che un'informazione manca. Per questo la soglia minima
(0,35) serve solo a eliminare il rumore evidente, e il giudizio di pertinenza resta al modello, che legge
i documenti. La soglia per `confidence = high` è 0,45.

Non è stato introdotto un cross-encoder per il re-ranking perché aggiunge dipendenze pesanti (torch) e
latenza. Con una KB di questa dimensione, ricerca ibrida ed espansione al documento risolvono i problemi
osservati; con una KB molto più grande il re-ranker diventerebbe la prossima scelta, e il calcolo dello
score denso su tutti i chunk andrebbe sostituito da una ricerca top-k.

## 8. Structured output e criteri riproducibili

Alla fine del loop il modello produce un `AnswerDraft` Pydantic tramite function calling
(`with_structured_output`). Il draft contiene solo ciò che solo l'LLM può sapere: il testo della risposta,
i documenti usati, la classificazione della richiesta (`in_scope` / `out_of_scope` / `needs_clarification`)
e se ha risposto a tutto (`fully_answered`). Il resto è deciso **dal codice**
([greenthumb/policy.py](greenthumb/policy.py)), con regole testate in `tests/test_components.py`:

**`sources`**: intersezione tra i documenti citati dal modello e quelli effettivamente restituiti dai tool
nel turno corrente. Le citazioni inventate vengono scartate e registrate nella trace (`dropped_sources`),
quindi `sources` non può contenere file inesistenti o non recuperati.

**`needs_human`** è `true` se:
1. l'agente ha aperto un ticket con `escalate_to_human`, oppure
2. il loop ha raggiunto il limite di iterazioni, oppure
3. la richiesta è pertinente ma il draft dichiara di non averle risposto del tutto. In questo caso scatta
   una **escalation di sicurezza**: il codice apre il ticket e aggiunge alla risposta codice e tempi.

**`confidence`** (regole valutate in ordine):

| Condizione | confidence |
|---|---|
| `needs_human` | `low`: la risposta automatica non risolve la richiesta |
| richiesta fuori ambito | `high`: il rifiuto è sempre conforme alla policy |
| richiesta di chiarimento | `medium` |
| nessun tool usato (convenevoli) | `medium` |
| tool usati ma nessuna evidenza utilizzabile | `low` |
| documenti recuperati ma nessuno citato (conoscenza non attribuita): si aggiunge uno score 0 all'evidenza | di fatto `medium` |
| minimo tra gli score delle fonti citate (1,0 per ogni ordine letto con successo) ≥ 0,45 | `high` |
| altrimenti | `medium` |

Chiedere all'LLM di autovalutare la propria confidenza produce valori poco calibrati e non riproducibili;
con queste regole la stessa evidenza produce sempre la stessa etichetta, e ogni etichetta è spiegabile
("la fonte più debole ha score 0,42"). La soglia 0,45 è configurabile (`GREENTHUMB_HIGH_CONFIDENCE_SCORE`)
ed è stata tarata sugli score osservati (sezione 7).

**Criteri di escalation** (nel prompt di sistema, allineati alla policy
`policies/servizio_clienti_contatti.md`): richiesta esplicita di un operatore; azioni che l'agente non può
eseguire (annullare o modificare ordini, rimborsi, eccezioni, garanzie, sostituzioni); prodotti danneggiati,
difettosi o errati; problemi di pagamento; consegne in ritardo (`delivery_overdue`); reclami; informazioni
non disponibili dopo la ricerca. La priorità (high / medium / low) determina i tempi di presa in carico
comunicati al cliente.

## 9. Deployment

`uvicorn greenthumb.api:app --port 8000`:

- `POST /chat` restituisce esattamente i 4 campi richiesti. Il `session_id` viaggia nell'header
  `X-Session-Id`, così il body resta identico al contratto della consegna;
- `POST /chat/debug` restituisce anche la trace ReAct completa;
- `DELETE /sessions/{id}` cancella la memoria di una sessione;
- `GET /health` mostra la configurazione attiva.

Gli input sono validati da Pydantic (messaggio non vuoto, massimo 2000 caratteri). Gli errori del modello
diventano 502 con un messaggio chiaro, e i fallimenti dei tool vengono restituiti al modello come
osservazione (`status: error`) invece di interrompere il turno.

## 10. Valutazione

### Dataset
`evaluation/eval_dataset.json`: **23 conversazioni** (3 multi-turno), ciascuna con risposta di riferimento,
tool attesi, fonti attese e `needs_human` atteso:

| Categoria | N. | Cosa verifica |
|---|---|---|
| multi_tool | 1 | l'esempio della consegna (ordine + guida) |
| order_status | 2 | lettura dello stato e dei rimborsi |
| product_info / cultivation_guide / policy | 7 | RAG su catalogo, guide e policy, compresa la trappola dalia/tulipani |
| escalation | 6 | richiesta esplicita, danni, ritardo, annullamento, doppio addebito, reclamo fuori finestra |
| edge_case | 3 | ordine inesistente, richiesta ambigua, prodotto fuori catalogo |
| multi_turn_memory | 2 | risoluzione di riferimenti ai turni precedenti ("e quando arriva?") |
| out_of_scope | 2 | domanda estranea e prompt injection |

Le metriche sono calcolate sull'ultimo turno di ogni conversazione, dopo aver giocato tutti i turni
precedenti nella stessa sessione.

### Metriche

| Metrica | Libreria | Cosa misura | Perché |
|---|---|---|---|
| Faithfulness | RAGAS | quota di affermazioni della risposta supportate dai contesti recuperati (chunk + record d'ordine) | il rischio principale di un assistente clienti è inventare date, prezzi o policy |
| Answer relevancy | RAGAS | quanto la risposta è pertinente alla domanda | risposte vaghe o fuori tema non risolvono il ticket |
| Context recall | RAGAS | quanto i contesti recuperati coprono i fatti della risposta di riferimento | qualità del retrieval, indipendente dalla generazione |
| Correctness (GEval) | DeepEval | correttezza rispetto al riferimento: fatti, date, importi, comportamento (escalation, rifiuto, chiarimento) | misura end-to-end dell'utilità della risposta |
| Tool correctness | DeepEval | quota dei tool attesi effettivamente chiamati | verifica l'orchestrazione ReAct |
| needs_human accuracy, source hit, retrieval hit, sources grounded, schema valid | deterministiche | escalation corretta; fonte attesa citata o recuperata; nessuna fonte non recuperata; JSON conforme | coprono i requisiti dello structured output |

Il giudice, per entrambe le librerie, è configurato con `GREENTHUMB_EVAL_MODEL` e passa da LiteLLM. Nella
configurazione consegnata è `gpt-4o-2024-11-20`: un modello più capace e **diverso** da quello
dell'agente, per ridurre il *self-preference bias* (un modello tende a valutare meglio i testi simili a
quelli che produce).

**Nota sui numeri riportati qui sotto**: le esecuzioni v1 e v2 sono state fatte con `gpt-4o-mini` come
giudice, prima del passaggio a `gpt-4o-2024-11-20`. Il confronto v1/v2 resta valido perché usa lo stesso
giudice. Rieseguendo `python evaluation/run_evaluation.py` con la configurazione attuale i punteggi
assoluti possono cambiare: un giudice più severo di solito abbassa faithfulness e correctness di
qualche punto.

### Risultati

<!-- RESULTS:START -->
La valutazione è stata eseguita prima sulla **v1** (baseline: solo ricerca densa con i 4 chunk migliori).
L'analisi dei suoi errori ha portato alla **v2**, che è la versione consegnata:
- ricerca ibrida ed espansione al documento intero;
- tre regole di prompt: "se la ricerca non contiene l'informazione prova l'altro strumento", "nessun fatto
  non presente nei risultati", "cita sempre i documenti da cui prendi informazioni";
- una regola di confidence per la conoscenza non attribuita;
- la correzione di un errore di misura: in v1 l'output di `escalate_to_human` non era incluso nei contesti
  passati a RAGAS, quindi codice ticket e tempi risultavano "non supportati".

I risultati completi sono in `evaluation/results/` (v2) e `evaluation/results_v1_baseline/` (v1).

| Metrica | N casi | v1 | **v2** | Δ |
|---|---|---|---|---|
| RAGAS faithfulness | 17 | 0,49 | **0,67** | +0,18 |
| RAGAS answer relevancy | 21 | 0,59 | **0,56** | −0,03 |
| RAGAS context recall | 12 | 0,55 | **0,88** | +0,33 |
| DeepEval correctness (GEval) | 23 | 0,79 | **0,83** | +0,04 |
| DeepEval tool correctness | 23 | 0,84 | **0,83** | −0,01 |
| needs_human accuracy | 23 | 0,96 | **1,00** | +0,04 |
| source hit (fonte attesa citata) | 13 | 0,69 | **0,69** | = |
| sources grounded (fonti ⊆ recuperate) | 23 | 1,00 | **1,00** | = |
| schema valid | 23 | 1,00 | **1,00** | = |

Latenza media: 6,2 s per conversazione (i casi multi-turno contano 3 turni). Distribuzione della
confidence in v2: 10 high, 6 medium, 7 low. Il numero di casi per le metriche RAGAS varia da un'esecuzione
all'altra, perché dipende dai tool che l'agente ha effettivamente chiamato (senza contesti la faithfulness
non è applicabile).

**Medie v2 per categoria**

| Categoria | Faithfulness | Answer relevancy | Context recall | Correctness | Tool correctness |
|---|---|---|---|---|---|
| multi_tool | 1,00 | 0,66 | 1,00 | 0,86 | 1,00 |
| order_status | 0,81 | 0,56 | n/a | 0,88 | 1,00 |
| product_info | 0,90 | 0,72 | 1,00 | 0,98 | 1,00 |
| policy | 0,83 | 0,58 | 1,00 | 0,82 | 1,00 |
| cultivation_guide | 0,78 | 0,66 | 0,83 | 0,88 | 0,67 |
| escalation | 0,46 | 0,52 | 0,75 | 0,72 | 0,86 |
| edge_case | 0,20 | 0,44 | n/a | 0,88 | 1,00 |
| multi_turn_memory | n/a | 0,50 | n/a | 0,81 | 0,00 |
| out_of_scope | n/a | n/a | n/a | 0,87 | 1,00 |

(n/a: metrica non applicabile, ad esempio faithfulness senza contesti o context recall senza documenti attesi.)

### Commento

**Cosa funziona.**
- **Structured output affidabile**: tutte le risposte rispettano lo schema, e `sources` contiene sempre e
  solo documenti effettivamente recuperati (100%, garantito per costruzione dalla policy).
- **Escalation**: `needs_human` è corretto in 23 casi su 23. L'agente riconosce richieste esplicite,
  danni, ritardi, annullamenti, doppi addebiti e reclami fuori finestra, e non passa all'operatore le
  domande informative, i fuori ambito o la prompt injection. Nel caso del prodotto inesistente (c19),
  l'escalation di sicurezza apre il ticket anche se il modello si era limitato a proporlo.
- **Orchestrazione multi-tool**: l'esempio della consegna (c01) usa `get_order_status` e
  `search_knowledge_base` in parallelo e produce la risposta attesa, con fonte `guides/bulbi_autunnali.md`,
  confidence high, faithfulness 1,00 e correctness 0,86.
- **Ordini, prodotti, guide e policy**: correctness tra 0,82 e 0,98, faithfulness tra 0,78 e 0,90,
  context recall tra 0,83 e 1,00. I casi limite (ordine inesistente, richiesta senza numero, fuori ambito,
  injection) hanno correctness tra 0,84 e 0,90: l'agente chiede il dato mancante o declina, senza inventare.

**Cosa ha insegnato la v1.** Gli errori più gravi della baseline venivano dal retrieval, non dal modello.
- Per la query breve "cesoie" la scheda giusta aveva score denso 0,15, sotto il concime, e l'agente
  concludeva che il prodotto non esisteva.
- Per "restituzione piantina" venivano recuperate sezioni della garanzia ma non "Prodotti esclusi dal
  recesso", e l'agente rispondeva, sbagliando, che la pianta si può rendere (correctness 0,22).
- Per i GreenPoints mancava la sezione con i "14 giorni dopo la consegna" e il modello inventava "dopo la
  spedizione".

La ricerca ibrida ha risolto il primo caso (correctness 0,98), l'espansione al documento intero gli altri
due (0,90 e 0,88). Il context recall è salito da 0,55 a 0,88 e la faithfulness da 0,49 a 0,67.

**Punti deboli.**
1. **Escalation poco informativa** (categoria escalation: correctness 0,72, faithfulness 0,46). In c08
   (piante danneggiate) il modello verifica l'ordine ma non consulta la policy; in c15 (annullamento) apre
   subito il ticket senza controllare né l'ordine né la policy (tool correctness 0,67 e 0,50). Così non
   ricorda al cliente di preparare le foto, la finestra di 48 ore o che l'ordine è annullabile perché
   ancora in preparazione. Per lo stesso motivo le escalation non citano la policy e la *source hit* resta
   ferma a 0,69. Inoltre il modello aggiunge frasi di cortesia con tempi o orari non presenti nei risultati,
   e questo abbassa la faithfulness (c09: 0,50, c15: 0,25).
2. **Conoscenza parametrica non ancorata** (c22, faithfulness 0,33). La risposta sulla dalia è corretta
   ma viene dalla conoscenza generale del modello: ha cercato solo nelle guide e non nel catalogo, dove si
   trova la scheda della dalia. La regola "prova l'altro strumento" non è stata applicata, perché il
   modello "sapeva" già la risposta.
3. **Risposte corrette ma incomplete** (c07, correctness 0,68, faithfulness 0,50): costo e tempi della
   spedizione standard sono giusti, ma mancano le alternative express e punto di ritiro, presenti nel
   documento recuperato.
4. **Tool correctness sui multi-turno** (0,00): all'ultimo turno di c16 e c17 l'agente risponde usando la
   memoria, perché data di consegna e periodo di trapianto erano già nella risposta precedente. Il
   comportamento è corretto (correctness 0,74 e 0,89) e più efficiente; è la metrica, che si aspettava una
   nuova chiamata, a penalizzarlo. Conferma però che la memoria a breve termine funziona.
5. **Answer relevancy bassa in generale** (0,56) e pari a 0 per il prodotto fuori catalogo (c19). RAGAS
   genera domande a partire dalla risposta e ne misura la similarità coseno con la domanda originale. Con
   embedding su testo italiano, risposte in più parti e formule di cortesia, la scala risulta compressa
   verso 0,5-0,7. Le risposte "non trovo il prodotto" vengono classificate come *non committal* e ricevono
   0 per costruzione, anche quando sono il comportamento desiderato. Va quindi letta insieme alla
   correctness, che confronta con il riferimento.
6. **Calibrazione della confidence**: la correctness media è 0,86 per le risposte *high*, 0,88 per le
   *medium* e 0,74 per le *low*. Le *low* coincidono con le escalation, come atteso. High e medium invece
   non si distinguono, perché *medium* raccoglie soprattutto chiarimenti e risposte da memoria, che sono
   facili. Con 23 casi la soglia di 0,45 non si può validare statisticamente.
7. **Testo della risposta dopo l'escalation di sicurezza** (c19): il modello scrive "se desideri posso
   aprire un ticket" e subito dopo il codice aggiunge "ho inoltrato la tua richiesta". È un'incoerenza di
   tono, da risolvere rigenerando la frase finale quando scatta l'escalation automatica.

**Affidabilità dei giudici e variabilità dell'agente.** Nelle esecuzioni riportate RAGAS e GEval usano
`gpt-4o-mini` come giudice, lo stesso modello dell'agente (da qui il passaggio a `gpt-4o-2024-11-20`), e
anche l'agente (a temperatura 0) non è perfettamente deterministico. Tra due esecuzioni della v2 le medie
sono cambiate di qualche centesimo (ad esempio la correctness è rimasta a 0,83, la faithfulness è passata
da 0,65 a 0,67). I punteggi dei singoli casi vanno letti insieme alle motivazioni salvate in `scores.csv`
(`deepeval_correctness_reason`). Le metriche deterministiche (needs_human, sources grounded, schema valid,
tool correctness) non hanno la varianza del giudice.

### Miglioramenti prioritari
1. **Escalation guidata dal codice**: prima di `escalate_to_human`, se nel messaggio c'è un numero
   d'ordine, eseguire automaticamente `get_order_status` e la ricerca della policy collegata al `reason`,
   invece di affidarsi al prompt. È il punto debole principale e si risolve in modo deterministico.
2. **Verifica di grounding pre-risposta**: un controllo (NLI o LLM) che confronti le affermazioni del
   draft con i contesti e rigeneri la risposta se trova fatti non supportati, come fa la faithfulness ma
   a runtime.
3. **Modello più capace per il passo di decisione** (es. `gpt-4o-2024-11-20` o un modello Claude via
   LiteLLM, cambiando solo `GREENTHUMB_LLM_MODEL`), tenendo `gpt-4o-mini` per riassunti e output
   strutturato; va valutato il
   rapporto tra costo e aumento della correctness.
4. **Tool correctness consapevole della memoria**: nei test multi-turno accettare come corretta anche la
   risposta senza tool, quando l'informazione è già nella memoria.
5. **Dataset più ampio** (almeno 100 conversazioni, idealmente da ticket reali anonimizzati), per ridurre
   la varianza e poter calibrare la soglia di confidence.
<!-- RESULTS:END -->

## 11. Limiti e possibili miglioramenti

- **Autenticazione**: chiunque conosca un numero d'ordine ne vede lo stato. In produzione servirebbe una
  verifica (email o login) prima di `get_order_status`.
- **Persistenza delle sessioni**: la memoria è in RAM, quindi si perde al riavvio e non scala su più
  istanze. Servirebbe Redis con TTL.
- **Re-ranking**: un cross-encoder multilingue o un re-ranker via API migliorerebbe la precisione sulle
  domande con termini simili tra più documenti.
- **Calibrazione della confidenza**: la soglia 0,45 va ricalibrata su un dataset più ampio, verificando
  che le risposte `high` abbiano davvero correttezza più alta (sezione 10, punto 6).
- **Costo e latenza**: ogni turno richiede almeno 3 chiamate LLM (decisione, tool, output strutturato). Si
  potrebbe far generare direttamente l'output strutturato all'ultimo passo del loop.
- **Valutazione**: 23 conversazioni sono poche per conclusioni statistiche; i giudici LLM hanno varianza
  propria. Servirebbe un dataset più ampio, estratto da conversazioni reali anonimizzate.
