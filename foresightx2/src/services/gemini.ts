// Shared Google Gemini client for the dashboard's language features
// (Query Urban Intelligence + Explainable AI plain-language summary).
// The key is injected at build time from GEMINI_API_KEY via vite.config.ts.

const GEMINI_MODEL = 'gemini-2.5-flash';
const GEMINI_API_URL = `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent`;

interface GeminiOptions {
  system: string;
  prompt: string;
  temperature?: number;
  maxOutputTokens?: number;
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Sends one prompt to Gemini and returns the response text.
 * Retries once on rate-limit (429) or transient server errors (5xx).
 */
export async function generateWithGemini({
  system,
  prompt,
  temperature = 0.5,
  maxOutputTokens = 1200,
}: GeminiOptions): Promise<string> {
  const apiKey = process.env.GEMINI_API_KEY || '';
  if (!apiKey) throw new Error('GEMINI_API_KEY is not configured');

  const body = JSON.stringify({
    systemInstruction: { parts: [{ text: system }] },
    contents: [{ role: 'user', parts: [{ text: prompt }] }],
    generationConfig: {
      temperature,
      maxOutputTokens,
      // Disable "thinking" so the whole token budget goes to the visible answer.
      thinkingConfig: { thinkingBudget: 0 },
    },
  });

  for (let attempt = 0; attempt < 2; attempt++) {
    const response = await fetch(GEMINI_API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'x-goog-api-key': apiKey },
      body,
    });

    if (response.ok) {
      const data = await response.json();
      const parts = data?.candidates?.[0]?.content?.parts || [];
      return parts.map((p: { text?: string }) => p.text || '').join('');
    }

    const retryable = response.status === 429 || response.status >= 500;
    if (retryable && attempt === 0) {
      await sleep(2000);
      continue;
    }
    const detail = await response.text();
    throw new Error(`Gemini request failed (${response.status}): ${detail.slice(0, 200)}`);
  }

  throw new Error('Gemini request failed after retry');
}
