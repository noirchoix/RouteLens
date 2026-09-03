import { asCapabilitiesResponse, asJsonObject, asJsonValue, asModelsResponse, asReadyResponse, isRecord } from './contracts';
import type { BatchRequest, CapabilitiesResponse, ContextualRequest, JsonObject, JsonValue, ModelsResponse, ReadyResponse } from './types';

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly payload: JsonValue | null
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

type RequestOptions = {
  method?: 'GET' | 'POST';
  body?: JsonObject;
  experimental?: boolean;
};

function errorMessage(status: number, payload: JsonValue | null): string {
  if (isRecord(payload)) {
    const detail = payload.detail;
    if (typeof detail === 'string') return detail;
    if (isRecord(detail)) {
      if (typeof detail.message === 'string') return detail.message;
      if (typeof detail.code === 'string') return detail.code.replaceAll('_', ' ');
    }
  }
  return `Request failed with HTTP ${status}.`;
}

export class RouteLensClient {
  constructor(private readonly getApiKey: () => string) {}

  private async request(path: string, options: RequestOptions = {}): Promise<unknown> {
    const headers = new Headers({ Accept: 'application/json' });
    if (options.body) headers.set('Content-Type', 'application/json');
    const apiKey = this.getApiKey().trim();
    if (apiKey) headers.set('X-API-Key', apiKey);
    if (options.experimental) headers.set('X-REACTS-Allow-Experimental', 'true');

    let response: Response;
    try {
      response = await fetch(path, {
        method: options.method ?? 'GET',
        headers,
        body: options.body ? JSON.stringify(options.body) : undefined
      });
    } catch (error) {
      throw new ApiError(error instanceof Error ? error.message : 'Network request failed.', 0, null);
    }

    const text = await response.text();
    let payload: JsonValue | null = null;
    if (text) {
      try {
        payload = asJsonValue(JSON.parse(text));
      } catch {
        payload = text;
      }
    }
    if (!response.ok) throw new ApiError(errorMessage(response.status, payload), response.status, payload);
    return payload;
  }

  async health(): Promise<JsonObject> {
    return asJsonObject(await this.request('/health'), 'health response');
  }

  async ready(): Promise<ReadyResponse> {
    return asReadyResponse(await this.request('/ready'));
  }

  async artifacts(): Promise<JsonObject> {
    return asJsonObject(await this.request('/api/v2/artifacts'), 'artifacts response');
  }

  async models(): Promise<ModelsResponse> {
    return asModelsResponse(await this.request('/api/v2/models'));
  }

  async datasets(): Promise<JsonObject> {
    return asJsonObject(await this.request('/api/v2/datasets'), 'datasets response');
  }


  async capabilities(): Promise<CapabilitiesResponse> {
    return asCapabilitiesResponse(await this.request('/api/v2/capabilities'));
  }

  async contextual(body: ContextualRequest): Promise<JsonValue> {
    const requestBody = asJsonObject(body, 'contextual request');
    return asJsonValue(
      await this.request('/api/v2/inference/contextual', {
        method: 'POST',
        body: requestBody,
        experimental: body.allow_experimental
      })
    );
  }

  async batch(body: BatchRequest): Promise<JsonValue> {
    const requestBody = asJsonObject(body, 'batch request');
    return asJsonValue(
      await this.request('/api/v2/inference/batch', {
        method: 'POST',
        body: requestBody,
        experimental: body.allow_experimental
      })
    );
  }

  async post(path: string, body: unknown): Promise<JsonValue> {
    const requestBody = asJsonObject(body, `${path} request`);
    return asJsonValue(await this.request(path, { method: 'POST', body: requestBody }));
  }

  async route(routeId: string): Promise<JsonValue> {
    return asJsonValue(await this.request(`/api/v2/routes/${encodeURIComponent(routeId)}`));
  }
}
