# 04 — Solución

## Implementación concreta

Esta solución materializa las 3 iniciativas en **slices funcionales** — unidades mínimas de trabajo que se pueden implementar, probar y poner en producción de forma independiente.

---

## Slice 1: Pipeline de Ingesta de Corpus (Iniciativa 1)
**Duración:** ~3 días | **Estado:** ✅ Completado

### ¿Qué se construyó?
Un pipeline automatizado que toma un video de YouTube y lo convierte en un registro lingüísticamente anotado en PostgreSQL.

### Flujo de extremo a extremo
```
YouTube URL
   │
   ▼
┌─────────────────┐
│  yt-dlp          │  → Descarga audio MP3
│  (downloader.py) │
└────────┬────────┘
         ▼
┌─────────────────┐
│  pyannote        │  → Identifica al candidato por embedding de voz
│  (diarizer.py)   │  → Extrae solo sus segmentos de audio
└────────┬────────┘
         ▼
┌─────────────────┐
│  Whisper large-v3│  → Transcribe audio → texto con timestamps
│  (transcriber.py)│  → CPU (~7.5 min / 20 min de audio)
└────────┬────────┘
         ▼
┌─────────────────┐
│  Limpieza        │  → Muletillas, falsos comienzos, normalización
│  (cleaner.py)    │  → Regex + reglas lingüísticas
└────────┬────────┘
         ▼
┌─────────────────┐
│  NLP Pipeline    │  → spaCy: tokens, POS, lemas, dep parse
│  (nlp_pipeline)  │  → BETO NER + gazetteer DANE (1,099 ubicaciones)
└────────┬────────┘
         ▼
┌─────────────────┐
│  DB Loader       │  → PostgreSQL: speeches, entities, annotations
│  (db_loader.py)  │
└─────────────────┘
```

### Componentes (UI → Lógica → BD)
| Capa | Componente | Archivo |
|------|-----------|---------|
| **Entrada** | URL de YouTube + metadata | `src/corpus/downloader.py` |
| **Lógica** | Diarización con embedding de referencia | `src/corpus/diarizer.py` |
| **Lógica** | Transcripción Whisper (CPU, large-v3) | `src/corpus/transcriber.py` |
| **Lógica** | Limpieza de lenguaje oral | `src/corpus/cleaner.py` |
| **Lógica** | Pipeline NLP (spaCy + BETO NER) | `src/pipeline/` |
| **Lógica** | Orquestador del pipeline completo | `src/corpus/pipeline_runner.py` |
| **BD** | Esquema PostgreSQL + pgvector | `schema.sql` |
| **BD** | Carga de datos | `src/corpus/db_loader.py` |

### Decisiones técnicas clave
- **Whisper en CPU** (no MPS) — 1.9x más rápido en M4 Mac Mini
- **BETO NER** reemplaza spaCy NER para mayor precisión en español
- **Gazetteer DANE** con 1,099 ubicaciones colombianas para NER geográfico
- **Diarización** con embedding de referencia — umbral de similitud coseno 0.25
- **Pipeline con `nohup`** — sobrevive al cierre de sesión de terminal

### Resultado
14 discursos ingresados (de 44 disponibles), ~30,963 palabras, anotaciones NLP completas en BD.

---

## Slice 2: Sistema RAG con Citación (Iniciativa 2)
**Duración:** ~2 días | **Estado:** ✅ Completado

### ¿Qué se construyó?
Un sistema de preguntas y respuestas que recupera fragmentos reales de discursos y genera respuestas citadas.

### Flujo de extremo a extremo
```
"¿Qué propone sobre reforma agraria?"
   │
   ▼
┌──────────────────┐
│  Embedder         │  → Convierte la pregunta en vector 768d
│  (embedder.py)    │  → paraphrase-multilingual-mpnet-base-v2
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Retriever        │  → pgvector: cosine similarity, top-5
│  (retriever.py)   │  → JOIN con speeches para metadata
│                    │  → Umbral mínimo: 0.3
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Generator        │  → Claude API (Haiku para dev, Sonnet/Opus para prod)
│  (generator.py)   │  → Prompt de sistema: neutral, español, citas obligatorias
│                    │  → Solo genera desde fragmentos recuperados
└────────┬─────────┘
         ▼
Respuesta con citas:
  "En su discurso 'Título' del 15/03/2026, Cepeda propuso..."
  (Fuente: [ver video](https://youtube.com/watch?v=abc&t=120))
```

### Componentes (UI → Lógica → BD)
| Capa | Componente | Archivo |
|------|-----------|---------|
| **Entrada** | Pregunta en lenguaje natural | `src/rag/query.py` |
| **Lógica** | Chunking semántico (~200 palabras, overlap 1 oración) | `src/rag/chunker.py` |
| **Lógica** | Embedding multilingual (768d) | `src/rag/embedder.py` |
| **Lógica** | Retrieval pgvector + citaciones | `src/rag/retriever.py` |
| **Lógica** | Generación Claude con restricciones | `src/rag/generator.py` |
| **BD** | `speech_chunks` con HNSW index | `schema.sql` |
| **BD** | Backfill para discursos pre-existentes | `src/rag/backfill.py` |

### Decisiones técnicas clave
- **Chunking por oraciones** (no por caracteres ni por ROOT del dependency parse) — respeta unidades temáticas del discurso
- **Overlap de 1 oración** entre chunks consecutivos — evita pérdida de información en fronteras
- **Timestamps fuzzy** — matching aproximado entre segmentos Whisper y oraciones spaCy para links con timestamp al video
- **Auto-chunk en ingesta** — cada nuevo discurso queda RAG-ready automáticamente
- **HF Inference API** como alternativa para embedding de queries en deploy (evita cargar ~868 MB del modelo local)

### Resultado
131 chunks indexados, 20/20 tests pasando, respuestas con citas en <5 segundos.

---

## Slice 3: Servidor MCP + Herramientas de Datos (Iniciativa 3)
**Duración:** ~2 días | **Estado:** ✅ Completado

### ¿Qué se construyó?
Un servidor MCP (Model Context Protocol) con 8 herramientas que exponen todas las capacidades del sistema como funciones invocables por un LLM.

### Herramientas MCP
| # | Herramienta | Tipo | Descripción |
|---|-------------|------|-------------|
| 1 | `retrieve_chunks` | Lectura | Búsqueda semántica pgvector + citas |
| 2 | `list_speeches` | Lectura | Listado de discursos con metadata |
| 3 | `get_speech_detail` | Lectura | Detalle de un discurso + conteos |
| 4 | `search_entities` | Lectura | Búsqueda ILIKE de entidades (con `limit`) |
| 5 | `get_speech_entities` | Lectura | Entidades de un discurso agrupadas por tipo NER |
| 6 | `get_corpus_stats` | Lectura | Estadísticas globales del corpus |
| 7 | `submit_opinion` | Escritura | Guardar opinión ciudadana (texto + voto) |
| 8 | `get_opinions` | Lectura | Consultar opiniones con estadísticas |

### Componentes
| Capa | Componente | Archivo |
|------|-----------|---------|
| **API** | Servidor MCP (FastMCP 3.0) | `src/mcp/server.py` |
| **Lógica** | Conexión BD ligera (sin imports pesados) | `src/mcp/db.py` |
| **Seguridad** | Validación de entrada + filtro de inyección | `src/mcp/server.py` |
| **Tests** | 29 tests (herramientas + seguridad) | `tests/mcp/` |

### Resultado
8 herramientas funcionando, 29 tests pasando, conectable a Claude Desktop como bonus.

---

## Slice 4: Chat Streamlit con Visualizaciones (Iniciativa 3)
**Duración:** ~2 días | **Estado:** ✅ Completado

### ¿Qué se construyó?
Una interfaz de chat donde un ciudadano hace preguntas y recibe respuestas citadas con visualizaciones automáticas.

### Flujo de extremo a extremo
```
Usuario escribe pregunta
   │
   ▼
┌──────────────────┐
│  Streamlit UI     │  → st.chat_input + st.chat_message
│  (app.py)         │  → Historial de conversación multi-turno
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Claude Haiku     │  → Analiza la pregunta
│  (Orquestador)    │  → Decide qué herramienta(s) MCP invocar
│                    │  → tool_use: retrieve_chunks, search_entities, etc.
└────────┬─────────┘
         ▼
┌──────────────────┐
│  MCP Tools        │  → Ejecuta la herramienta seleccionada
│  (server.py)      │  → Devuelve datos crudos (JSON)
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Claude Haiku     │  → Sintetiza respuesta en español
│  (Generador)      │  → Incluye citas con links a YouTube
│                    │  → Tono neutral e informativo
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Visualizaciones  │  → Barras de entidades (Plotly)
│  (viz.py)         │  → Mapa de Colombia (bubble plot geográfico)
│                    │  → Tablas interactivas
│                    │  → 6 tipos de gráficas según herramienta
└────────┬─────────┘
         ▼
Respuesta renderizada en el chat
   con texto + citas + gráficas + mapa
```

### Componentes (UI → Lógica → BD)
| Capa | Componente | Archivo |
|------|-----------|---------|
| **UI** | Chat Streamlit + sidebar | `src/frontend/app.py` |
| **UI** | Visualizaciones Plotly (6 tipos + mapa) | `src/frontend/viz.py` |
| **Lógica** | Prompts de sistema + definición de tools | `src/frontend/prompts.py` |
| **Lógica** | Dispatch de herramientas MCP | `src/frontend/app.py` |
| **BD** | Todas las tablas via MCP tools | `src/mcp/db.py` |

### Decisiones técnicas clave
- **Streamlit = backend** — Python server-side, API key segura, MCP calls locales
- **Claude Haiku como orquestador** — ~$0.0047/query, suficiente para producción ligera
- **Rate limiting** — 30 mensajes por sesión
- **UUIDs en gráficas** — `key=str(uuid4())` para evitar crashes por IDs duplicados en Streamlit
- **Regla anti-alucinación** — Haiku fabricó un resultado de herramienta; prompt explícito: "NUNCA finjas haber ejecutado una herramienta"

### Resultado
Chat funcional con visualizaciones, 24 tests de visualización pasando, costo ~$0.23 para demo de 50 queries.

---

## Slice 5: Deploy en la Nube (Iniciativa 3)
**Duración:** ~1 día | **Estado:** 🔲 Pendiente

### ¿Qué se va a construir?
Despliegue en Streamlit Community Cloud apuntando a PostgreSQL remoto.

### Arquitectura de despliegue
```
Local (Mac Mini M4)                Cloud
┌────────────────────┐             ┌────────────────────────┐
│ Pipeline pesado:    │   sync     │ Streamlit Community     │
│ yt-dlp, pyannote,  │  ───────►  │   Cloud (free tier)     │
│ Whisper, spaCy,    │  (DB_HOST) │                          │
│ BETO, embedder     │            │ PostgreSQL remoto        │
│                     │            │   + pgvector (RDS/Neon) │
│ Costo: $0           │            │                          │
│ (hardware propio)   │            │ Claude API (Haiku)      │
└────────────────────┘             │   ~$0.0047/query        │
                                    └────────────────────────┘
```

### Componentes
| Capa | Componente | Detalle |
|------|-----------|---------|
| **Infra** | BD remota | RDS PostgreSQL o Neon (free tier) |
| **Infra** | App hosting | Streamlit Community Cloud |
| **Seguridad** | SSL obligatorio | `DB_SSLMODE=require` |
| **Lógica** | Embedding via API | `EMBEDDING_PROVIDER=hf_api` (evita cargar 868 MB en cloud) |

### Costo estimado
- **Demo (50 queries/mes):** ~$0.23 en API + $0 hosting = **~$0.23/mes**
- **Producción ligera:** ~$10-35/mes (RDS + EC2 si no se usa free tier)

---

## Resumen de Slices

| Slice | Iniciativa | Duración | Estado | Entrega |
|-------|-----------|----------|--------|---------|
| 1. Pipeline de Ingesta | Corpus Digital | ~3 días | ✅ | 14 discursos anotados en BD |
| 2. Sistema RAG | RAG + Citación | ~2 días | ✅ | Preguntas/respuestas con citas |
| 3. Servidor MCP | Interfaz Chat | ~2 días | ✅ | 8 herramientas, 29 tests |
| 4. Chat Streamlit | Interfaz Chat | ~2 días | ✅ | Chat con visualizaciones |
| 5. Deploy Cloud | Interfaz Chat | ~1 día | 🔲 | App pública accesible |

**Total implementado:** ~9 días de trabajo → sistema funcional de extremo a extremo.

### Slices futuros (diferidos)

| Slice | Descripción |
|-------|-------------|
| 6. Análisis n-gramas + BERTopic | Descubrimiento automático de temas, frecuencias léxicas |
| 7. Sentimiento y retórica | Tono emocional por discurso/tema, dispositivos retóricos |
| 8. Gramática formal + Ontología | CFG para español, ontología de dominio político |
| 9. Clasificador de intenciones | BETO fine-tuned para intenciones del usuario |
| 10. Diálogo inteligente | Gestor de estado, flujos pragmáticos, seguimiento multi-turno |
