# 01 — Idea de Negocio

## ¿Para qué existe el proyecto?

**Visión:** Democratizar el acceso informado al discurso político colombiano mediante inteligencia artificial, permitiendo que cualquier ciudadano consulte, explore y verifique las propuestas reales de un candidato presidencial — basándose exclusivamente en lo que dijo, con fuentes y citas verificables.

## Oportunidad

Colombia enfrenta elecciones presidenciales en 2026. Los candidatos producen decenas de horas de discursos públicos — en plazas, debates, entrevistas, foros — pero este contenido es efímero: se ve una vez en YouTube y se olvida. No existe una forma estructurada de consultarlo. Los votantes dependen de resúmenes periodísticos (con sesgo editorial), clips virales en redes sociales (descontextualizados) o la memoria colectiva (imprecisa).

Al mismo tiempo, los modelos de lenguaje (LLMs) han demostrado ser capaces de sintetizar información y responder preguntas de forma natural. Pero si se usan sin datos reales como fuente, fabrican respuestas — el problema conocido como "alucinación". La tecnología RAG (Retrieval-Augmented Generation) resuelve esto: primero recupera fragmentos reales, luego genera una respuesta fundamentada.

## La oportunidad específica

- **~50 discursos** del candidato Iván Cepeda Castro, disponibles públicamente en YouTube (~25 horas de audio, ~200k+ palabras)
- Cero herramientas existentes que permitan consultar este corpus de forma estructurada
- La tecnología RAG + NLP está madura y accesible (open source en su mayoría)
- El proyecto sirve como caso de uso real dentro de la iniciativa de IA de la empresa, demostrando cómo construir sistemas RAG end-to-end sobre datos no estructurados en español

## Propuesta de valor

> Un asistente conversacional que responde preguntas sobre las propuestas de un candidato presidencial colombiano, citando discursos específicos con título, fecha y enlace con timestamp al video original. Neutral, verificable, transparente. **El LLM es la boca; BERT es el cerebro.**

## Stakeholders

| Rol | Quién | Interés |
|-----|-------|---------|
| Usuario final | Ciudadano colombiano, periodista, analista | Consultar propuestas con fuentes verificables |
| Desarrollador | Oscar Marulanda | Diseñar, construir y documentar el sistema |
| Empresa | ST&T Colombia — Iniciativa de IA | Caso de uso replicable: RAG + NLP sobre datos no estructurados en español |
| Candidato (indirecto) | Iván Cepeda Castro | Visibilidad y transparencia de sus propuestas |
