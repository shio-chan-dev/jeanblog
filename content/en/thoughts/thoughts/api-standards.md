---
title: "API Standards"
date: 2025-11-06
draft: false
---

# Title

**How to Write a Qualified API Document: From Swagger to Modern OpenAPI**

---

## Subtitle / Abstract

Want developers to actually enjoy using your API? This article covers the structure, examples, and best practices of high-quality API documentation based on Swagger/OpenAPI (originally by Tony Tam).

---

## Target readers

- Beginners who want a standard API doc structure
- Mid-level developers improving maintainability
- Architects and leads defining API standards

---

## Background / Motivation

Common problems in API docs:

- inconsistent format
- out-of-date content
- not usable for automation or testing

Tony Tam introduced **Swagger** (renamed to OpenAPI) in 2010 to solve this. It is now the de facto standard for REST API docs, used by Google, Amazon, Stripe, and more.

---

## Core concepts

| Concept | Description |
| --- | --- |
| **API doc** | Technical specification of how to call an API and interpret requests/responses |
| **Swagger/OpenAPI** | Standard to define, generate, and test REST APIs |
| **Endpoint** | A concrete path like `/users/{id}` |
| **Schema** | Field structure for requests and responses |

---

## Practical steps

1. **Define a clear structure**
   - Overview
   - Authentication
   - Endpoints
   - Schemas
   - Errors and examples

2. **Use OpenAPI (YAML is recommended)**

3. **Recommended tools**
   - Editors: Swagger Editor, Stoplight Studio, VS Code + YAML
   - Docs: Swagger UI / ReDoc
   - Auto-gen: Springdoc, FastAPI, NestJS

---

## Runnable example (OpenAPI)

```yaml
openapi: 3.0.0
info:
  title: User Management API
  version: 1.0.0
  description: APIs for user management.
servers:
  - url: https://api.example.com/v1
paths:
  /users/{id}:
    get:
      summary: Get user by ID
      parameters:
        - name: id
          in: path
          required: true
          description: User ID
          schema:
            type: string
      responses:
        '200':
          description: OK
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/User'
        '404':
          description: User not found
components:
  schemas:
    User:
      type: object
      properties:
        id:
          type: string
          description: Unique user id
        name:
          type: string
          description: User name
        email:
          type: string
          description: Email address
```

You can import this into [Swagger Editor](https://editor.swagger.io/) for visualization and testing.

---

## Explanation

**Why OpenAPI?**

- Standardized: avoid custom formats
- Automated: generate SDKs, tests, mocks
- Interactive: Swagger UI allows live testing

**Alternatives:**

- RAML (MuleSoft)
- API Blueprint (documentation-focused)

OpenAPI wins because of its tooling ecosystem.

---

## Common pitfalls

| Problem | Cause | Fix |
| --- | --- | --- |
| Docs and code drift | manual updates | auto-generate from code (FastAPI, Springdoc) |
| Schema too complex | deep nesting | split models with `$ref` |
| Missing fields in examples | no mock testing | use mock server to validate |

---

## Best practices

1. Version your API paths (e.g., `/v1/`)
2. Standardize error format (`{code, message, data}`)
3. Keep docs in sync with code
4. Add real examples
5. Validate OpenAPI in CI

---

## Summary

Great API docs are not just documentation but a collaboration bridge. Swagger/OpenAPI is about making APIs **machine-readable and human-usable**. With the right structure and tools, your APIs become easier to maintain and test.

---

## References

- [OpenAPI Specification](https://swagger.io/specification/)
- [Swagger Editor](https://editor.swagger.io/)
- [ReDoc](https://redocly.com/)
- [Microsoft API Design Guidelines](https://learn.microsoft.com/en-us/azure/architecture/best-practices/api-design)

---

## Meta

- Reading time: 7 minutes
- Tags: API docs, Swagger, OpenAPI, standards, Tony Tam
- SEO keywords: API documentation standard, Swagger tutorial, OpenAPI example, RESTful design
- Meta description: Based on Swagger/OpenAPI, this guide explains API doc structure, examples, and best practices.

---

## Call to Action (CTA)

Try it now:

- Open [Swagger Editor](https://editor.swagger.io/) and paste the YAML above
- Or follow this series on API design; next post: auto-generate SDKs with OpenAPI
