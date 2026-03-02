# 03 — Iniciativa

## Frentes estratégicos de acción

Para resolver el problema de inaccesibilidad del discurso político, se definen **3 iniciativas** (líneas de acción estratégicas). Cada una es un frente amplio que agrupa trabajo técnico concreto.

---

## Iniciativa 1: Construcción del Corpus Digital Anotado

**Objetivo:** Transformar ~50 discursos en audio/video en un corpus lingüístico digital, estructurado, almacenado en base de datos, con anotaciones NLP completas.

**Alcance:**
- Pipeline automatizado de extremo a extremo: descarga → diarización → transcripción → limpieza → análisis NLP → carga en BD
- Identificación automática del candidato entre múltiples locutores (embedding de voz de referencia)
- Anotaciones: tokenización, lematización, POS tagging, NER (BETO + gazetteer DANE), parsing de dependencias
- Almacenamiento en PostgreSQL con metadatos por discurso (fecha, evento, duración, URL)
- Procesamiento incremental: nuevos discursos se ingresan ejecutando el pipeline con `--new=N`

**No incluye (pero prepara para):** análisis lingüístico avanzado (n-gramas, BERTopic, sentimiento), que consumirá las anotaciones ya almacenadas.

**Decisión clave:** El pipeline NLP corre durante la ingesta — las anotaciones se acumulan "gratis" para futuros análisis.

---

## Iniciativa 2: Sistema RAG con Citación Verificable

**Objetivo:** Construir un sistema de Retrieval-Augmented Generation que responda preguntas sobre las propuestas del candidato con respuestas fundamentadas, citadas y verificables.

**Alcance:**
- Chunking semántico lingüístico (agrupación por oraciones, ~200 palabras, no por conteo de caracteres)
- Embeddings multilingües (`paraphrase-multilingual-mpnet-base-v2`, 768 dimensiones)
- Almacenamiento vectorial en pgvector con índice HNSW
- Retrieval por similitud coseno (top-5, umbral 0.3)
- Generación con Claude API restringida por prompt de sistema (neutral, español, con citas obligatorias)
- Citación: título del discurso, fecha, enlace a YouTube con timestamp exacto
- Transparencia: cuando el tema no fue abordado, el sistema lo dice explícitamente

**Principio rector:** "Claude es la boca, BERT es el cerebro." El LLM solo genera a partir de fragmentos pre-recuperados. Nunca ve el corpus completo.

**Decisión clave:** MCP-only architecture — sin API REST. Claude (Haiku) en Streamlit orquesta herramientas MCP via `tool_use`.

---

## Iniciativa 3: Interfaz Conversacional Accesible

**Objetivo:** Hacer que el sistema sea usable por un ciudadano no técnico, mediante una interfaz de chat con visualizaciones y exploración del corpus.

**Alcance:**
- Chat Streamlit con Claude Haiku como orquestador
- 8 herramientas MCP (6 consulta + 2 escritura): búsqueda semántica, listado de discursos, detalle de discurso, búsqueda de entidades, estadísticas del corpus, opiniones ciudadanas
- Visualizaciones inline: barras de entidades, mapa de Colombia con bubble plot geográfico, tablas interactivas
- Rate limiting por sesión (30 mensajes/sesión)
- Despliegue en Streamlit Community Cloud (free tier)
- Pipeline de procesamiento local (Mac Mini) apuntando a BD remota — sin costo de GPU en la nube

**Decisión clave:** Streamlit ES el backend — Python corre server-side, la API key está segura, las llamadas MCP son locales. El navegador solo ve HTML renderizado.

---

## Relación entre iniciativas

```
Iniciativa 1              Iniciativa 2              Iniciativa 3
(Corpus Digital)    ──►    (RAG + Citación)    ──►    (Interfaz Chat)
                                                          │
Audio/Video                Fragmentos                    │
   → BD anotada              → Respuestas citadas        ▼
                                                     Usuario final
```

Las iniciativas son secuenciales: sin corpus no hay retrieval, sin retrieval no hay chat. Pero cada una entrega valor por sí sola:
- **Iniciativa 1** = corpus consultable por SQL
- **Iniciativa 2** = preguntas/respuestas por terminal
- **Iniciativa 3** = producto usable por cualquier persona

## Iniciativas futuras (diferidas)

| Iniciativa | Descripción | Estado |
|------------|-------------|--------|
| Análisis lingüístico | N-gramas, BERTopic, sentimiento, perfil sintáctico | Diferida — datos ya acumulados |
| Lingüística formal | CFG (NLTK), ontología de dominio político | Diferida |
| Diálogo inteligente | Clasificador de intenciones BETO, gestor de diálogo, flujos pragmáticos | Diferida |
