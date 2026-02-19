# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- **BREAKING**: Authentication is now enabled by default (`AUTH_ENABLED=true`).
  Previously, authentication was disabled by default, meaning all API endpoints
  were accessible without credentials. This change enforces authentication on
  all endpoints except health probes and OGC discovery paths.

  **Migration guide**: If you rely on the previous behavior (no auth), set
  `AUTH_ENABLED=false` in your environment or `.env` file. For production
  deployments, generate API keys using the `APIKeyStore` and provide them
  via the `X-API-Key` header.

### Added

- `customer_id` field on `UserContext` and `APIKey` dataclasses for multi-tenant
  isolation. Defaults to `"legacy"` for backward compatibility.
- `TenantMiddleware` that extracts `customer_id` from the authenticated API key
  and attaches it to `request.state.customer_id`.
- Auth exemption allowlist for health probes (`/health`, `/ready`) and OGC
  discovery paths (`/oapi/`, `/oapi/conformance`, `/oapi/processes`).
- Database migration `000_add_customer_id.sql` for persistent API key storage.
- `.env.example` with documented configuration variables.
