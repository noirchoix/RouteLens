(function attachRouteLensApi(global) {
  "use strict";

  class ApiError extends Error {
    constructor(message, status, payload) {
      super(message);
      this.name = "ApiError";
      this.status = status;
      this.payload = payload;
    }
  }

  function isRecord(value) {
    return Boolean(value) && typeof value === "object" && !Array.isArray(value);
  }

  function parseBody(text) {
    if (!text) return null;
    try {
      return JSON.parse(text);
    } catch (_error) {
      return text;
    }
  }

  function errorMessage(status, payload) {
    if (isRecord(payload)) {
      const detail = payload.detail;
      if (typeof detail === "string") return detail;
      if (isRecord(detail)) {
        if (typeof detail.message === "string") return detail.message;
        if (typeof detail.code === "string") return detail.code.replaceAll("_", " ");
      }
    }
    if (typeof payload === "string" && payload.trim()) return payload.trim();
    return `Request failed with HTTP ${status}`;
  }

  function createClient(getApiKey) {
    async function request(path, options = {}) {
      const headers = new Headers(options.headers || {});
      headers.set("Accept", "application/json");
      if (options.body !== undefined) headers.set("Content-Type", "application/json");
      const apiKey = String(getApiKey() || "").trim();
      if (apiKey) headers.set("X-API-Key", apiKey);
      if (options.experimental) headers.set("X-REACTS-Allow-Experimental", "true");

      let response;
      try {
        response = await fetch(path, {
          method: options.method || "GET",
          headers,
          body: options.body === undefined ? undefined : JSON.stringify(options.body),
          signal: options.signal,
        });
      } catch (error) {
        throw new ApiError(error instanceof Error ? error.message : "Network request failed", 0, null);
      }

      const payload = parseBody(await response.text());
      if (!response.ok) throw new ApiError(errorMessage(response.status, payload), response.status, payload);
      return payload;
    }

    return {
      request,
      health: () => request("/health"),
      ready: () => request("/ready"),
      artifacts: () => request("/api/v2/artifacts"),
      models: () => request("/api/v2/models"),
      datasets: () => request("/api/v2/datasets"),
      contextual: (body) => request("/api/v2/inference/contextual", { method: "POST", body, experimental: body.allow_experimental }),
      batch: (body) => request("/api/v2/inference/batch", { method: "POST", body, experimental: body.allow_experimental }),
      repair: (body) => request("/api/v2/inference/repair", { method: "POST", body }),
      anomaly: (body) => request("/api/v2/inference/anomaly", { method: "POST", body }),
      routeQuality: (body) => request("/api/v2/inference/route-quality", { method: "POST", body }),
      retrieveReactions: (body) => request("/api/v2/retrieval/reactions", { method: "POST", body }),
      retrieveRoutes: (body) => request("/api/v2/retrieval/routes", { method: "POST", body }),
      route: (routeId) => request(`/api/v2/routes/${encodeURIComponent(routeId)}`),
    };
  }

  global.RouteLensApi = { ApiError, createClient, isRecord };
})(window);
