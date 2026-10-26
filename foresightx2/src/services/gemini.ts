// Language features (Query Urban Intelligence, plain-language summaries) go
// through the backend's rate-limited Gemini proxy. No API key ships in the
// browser bundle any more.

import { backendApi } from './dataService';

interface GeminiOptions {
  system: string;
  prompt: string;
  temperature?: number;
  maxOutputTokens?: number;
}

export async function generateWithGemini({
  system,
  prompt,
  temperature = 0.5,
  maxOutputTokens = 1200,
}: GeminiOptions): Promise<string> {
  const response = await backendApi.ai({ system, prompt, temperature, maxOutputTokens });
  return response.data?.text || '';
}
