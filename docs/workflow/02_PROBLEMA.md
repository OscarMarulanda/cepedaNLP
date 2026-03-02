# 02 — Problema

## Obstáculo concreto

**Los ciudadanos no pueden consultar de forma precisa y verificable qué dijo un candidato presidencial sobre un tema específico.**

## El dolor — específico, claro y medible

### 1. El contenido político es inaccesible como dato

Un candidato como Iván Cepeda ha producido **~50 discursos** (~25 horas de audio, ~200k+ palabras). Este contenido existe como videos de YouTube de 30-90 minutos cada uno. Para responder una pregunta como *"¿Qué propone Cepeda sobre reforma agraria?"*, un ciudadano tendría que:

- Buscar entre ~50 videos cuáles mencionan el tema
- Ver horas de contenido (o saltar aleatoriamente)
- Tomar notas y cruzar información entre discursos
- Distinguir al candidato de otros panelistas en debates

**Costo estimado:** 3-8 horas para responder una sola pregunta con rigor. En la práctica, nadie lo hace.

### 2. Las alternativas actuales son deficientes

| Alternativa | Problema |
|-------------|----------|
| Resúmenes periodísticos | Sesgo editorial, selección parcial de citas |
| Clips en redes sociales | Descontextualizados, editados para viralidad |
| Preguntar a un LLM directamente | Fabrica respuestas ("alucinación"), no cita fuentes reales |
| Buscar en Google | Devuelve páginas web, no fragmentos de discursos con citas |
| Ver los videos | Impracticable — nadie ve 25+ horas de contenido |

### 3. No existe infraestructura para el discurso político como dato estructurado

- Los discursos no están transcritos (son solo audio/video)
- Cuando hay transcripción (subtítulos automáticos de YouTube), es de baja calidad y no identifica quién habla
- No hay segmentación temática ni indexación semántica
- No hay vinculación de entidades mencionadas (personas, lugares, organizaciones) con los fragmentos donde aparecen
- No hay sistema de citación que permita verificar una afirmación contra la fuente original

### 4. Problema técnico subyacente

Convertir ~25 horas de audio político multilocutor en un sistema consultable requiere resolver una cadena de problemas de NLP/ML:

| Paso | Desafío |
|------|---------|
| Diarización | Identificar al candidato entre múltiples locutores (paneles, debates, entrevistas) |
| Transcripción | Audio con ruido de fondo, aplausos, interrupciones, acentos regionales |
| Limpieza | Muletillas ("o sea", "digamos"), falsos comienzos, repeticiones propias del habla |
| Análisis lingüístico | Tokenización, POS, NER, parsing en español colombiano |
| Chunking semántico | Dividir en unidades temáticas coherentes (no por caracteres) |
| Embedding | Representar fragmentos como vectores para búsqueda semántica |
| Retrieval | Encontrar los fragmentos más relevantes para una pregunta |
| Generación | Sintetizar una respuesta coherente con citas, sin inventar nada |

**Ningún producto existente resuelve esta cadena completa para discurso político en español.**

## Métrica de éxito

El problema está resuelto cuando un usuario puede escribir una pregunta en lenguaje natural (ej: *"¿Qué ha dicho sobre el racismo?"*) y recibir en <5 segundos una respuesta fundamentada con:
- Fragmentos reales de discursos citados
- Título y fecha de cada discurso fuente
- Enlace directo al momento exacto del video en YouTube
- Transparencia cuando el tema no fue abordado
