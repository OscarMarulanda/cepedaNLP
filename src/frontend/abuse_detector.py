"""Detect obvious tech-person attacks before they reach the LLM.

Catches blatant SQL injection, prompt injection/jailbreak, XSS, and
system-prompt extraction attempts.  Regular user questions — even
off-topic ones — pass through unchanged.

The MATRIX_RAIN_HTML constant provides a self-contained canvas
animation that renders inside a Streamlit chat bubble via
``st.components.v1.html()``.
"""

import re

# ---------------------------------------------------------------------------
# 1. SQL Injection
# ---------------------------------------------------------------------------
_SQL_PATTERNS = re.compile(
    r"""
    (?:                          # structural SQL injection fragments
        '\s*OR\s+.+=.+--         # ' OR 1=1 --
      | '\s*;\s*(?:DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE)\b
      | UNION\s+(?:ALL\s+)?SELECT\b
      | (?:DROP|DELETE\s+FROM|TRUNCATE)\s+TABLE\b
      | OR\s+1\s*=\s*1           # OR 1=1 (without leading quote)
      | '\s*OR\s+'[^']*'\s*=\s*' # ' OR 'a'='a'
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ---------------------------------------------------------------------------
# 2. Prompt Injection / Jailbreak
# ---------------------------------------------------------------------------
_PROMPT_INJECTION_PATTERNS = re.compile(
    r"""
    (?:
        ignor(?:e|a)\s+(?:all\s+)?(?:previous|prior|above|all|anteriores|previas)\s+
            (?:instructions?|instrucciones|prompts?|rules?)
      | ignor(?:e|a)\s+(?:todas?\s+)?(?:las?\s+)?(?:instructions?|instrucciones|prompts?|rules?)\s+
            (?:previous|prior|anteriores|previas)
      | ignor(?:e|a)\s+(?:your|tus?)\s+(?:instructions?|instrucciones|prompts?|rules?)
      | (?:you\s+are|eres|act(?:\s+as|\s+like|[úu]a\s+como))\s+(?:now\s+)?
            (?:DAN|evil|unfiltered|uncensored|un\s+(?:asistente\s+)?(?:sin\s+filtro|malvado))
      | jailbreak
      | \[?\s*SYSTEM\s*\]?:?\s+
      | developer\s+mode
      | modo\s+(?:desarrollador|dios|god)
      | do\s+anything\s+now
      | simulate\s+(?:a\s+)?(?:unrestricted|unfiltered|evil)
      | bypass\s+(?:your\s+)?(?:filters?|safety|rules?|restrictions?)
      | override\s+(?:your\s+)?(?:instructions?|programming|system\s+prompt)
      | pretend\s+(?:you\s+(?:are|have)\s+)?no\s+(?:restrictions?|rules?|filters?)
      | finge\s+que\s+no\s+tienes\s+(?:restricciones|reglas|filtros)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ---------------------------------------------------------------------------
# 3. XSS
# ---------------------------------------------------------------------------
_XSS_PATTERNS = re.compile(
    r"""
    (?:
        <\s*script\b
      | <\s*iframe\b
      | <\s*img\b[^>]+\bon\w+\s*=   # <img onerror=...>
      | \bon(?:error|load|click|mouseover)\s*=
      | javascript\s*:
      | <\s*svg\b[^>]+\bon\w+\s*=
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ---------------------------------------------------------------------------
# 4. System Prompt Extraction
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT_PATTERNS = re.compile(
    r"""
    (?:
        (?:show|display|reveal|print|output|give)\s+(?:me\s+)?
            (?:your|the)\s+(?:system\s+)?(?:prompt|instructions?)
      | repeat\s+(?:everything|all(?:\s+the\s+text)?)\s+(?:above|before|prior)
      | (?:cu[aá]les?\s+son|dime|mu[eé]strame|revela|imprime)\s+
            tus?\s+(?:instrucciones|reglas|prompt)
      | what\s+(?:is|are)\s+your\s+(?:system\s+)?(?:prompt|instructions?|rules?)
      | dump\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions?)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_ALL_PATTERNS = [
    _SQL_PATTERNS,
    _PROMPT_INJECTION_PATTERNS,
    _XSS_PATTERNS,
    _SYSTEM_PROMPT_PATTERNS,
]


def detect_abuse(message: str) -> bool:
    """Return True if the message is an obvious tech-person attack."""
    if not message or not message.strip():
        return False
    for pattern in _ALL_PATTERNS:
        if pattern.search(message):
            return True
    return False


# ---------------------------------------------------------------------------
# Matrix Rain HTML — self-contained canvas animation
# ---------------------------------------------------------------------------
MATRIX_RAIN_HEIGHT = 420

MATRIX_RAIN_HTML = """
<!DOCTYPE html>
<html>
<head>
<style>
  * { margin: 0; padding: 0; }
  body { background: #000; overflow: hidden; }
  canvas { display: block; }
</style>
</head>
<body>
<canvas id="matrix"></canvas>
<script>
(function() {
  const canvas = document.getElementById('matrix');
  const ctx = canvas.getContext('2d');

  canvas.width = window.innerWidth || 600;
  canvas.height = 400;

  const chars = 'アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン'
              + 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
              + '0123456789ñ¿¡{}[]<>@#$%&';
  const fontSize = 14;
  const columns = Math.floor(canvas.width / fontSize);
  const drops = new Array(columns).fill(1);

  const startTime = Date.now();
  const duration = 3000;
  const fadeStart = 2500;

  function draw() {
    const elapsed = Date.now() - startTime;

    if (elapsed >= duration) {
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      return;
    }

    // Trail effect
    ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Fade out in last 500ms
    let alpha = 1;
    if (elapsed > fadeStart) {
      alpha = 1 - (elapsed - fadeStart) / (duration - fadeStart);
    }

    ctx.fillStyle = `rgba(0, 255, 0, ${alpha})`;
    ctx.font = fontSize + 'px monospace';

    for (let i = 0; i < drops.length; i++) {
      const char = chars[Math.floor(Math.random() * chars.length)];
      ctx.fillText(char, i * fontSize, drops[i] * fontSize);

      if (drops[i] * fontSize > canvas.height && Math.random() > 0.975) {
        drops[i] = 0;
      }
      drops[i]++;
    }

    requestAnimationFrame(draw);
  }

  draw();
})();
</script>
</body>
</html>
""".strip()
