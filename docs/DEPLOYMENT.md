# Deployment

Three targets, same code and images:

| Target | When | State lives in |
|---|---|---|
| **Vercel** (2 container services) | Public demo link for the interview | in-container (default) or Qdrant Cloud / hosted Redis / Postgres |
| **Docker Compose** | Local full stack, monitoring demo | Docker volumes |
| **Kubernetes** (EKS/GKE/AKS) | "How would you run this in production?" | PVC (Qdrant), Postgres, Redis |

---

## 1. Vercel — API and UI as Docker containers

Vercel runs OCI images built from a `Dockerfile.vercel` as Vercel Functions (Container Runtime, beta),
and `services` in `vercel.json` put several of them in one project behind path rewrites:

```json
"services": {
  "api": { "root": "./",  "entrypoint": "docker/Dockerfile.api.vercel" },
  "ui":  { "root": "ui/", "entrypoint": "Dockerfile.vercel" }
},
"rewrites": [
  { "source": "/api/(.*)", "destination": { "service": "api" } },
  { "source": "/(.*)",     "destination": { "service": "ui" } }
]
```

So `https://<project>.vercel.app/` is the Streamlit UI and `https://<project>.vercel.app/api/docs` is the
API. The API serves every route both at `/` and under `/api`, so it works whether or not the prefix is
forwarded.

### Steps

1. Push the repo to GitHub.
2. In Vercel: **Add New → Project → import the repo**. Vercel reads `vercel.json` and builds both images.
   (Or from your machine: `npm i -g vercel && vercel deploy --prod`.)
3. Project **Settings → Environment Variables**:

   | Variable | Value |
   |---|---|
   | `API_URL` | `https://<your-project>.vercel.app/api` (the UI calls the API through the public rewrite) |
   | `ANTHROPIC_API_KEY` | optional — turns on LLM understanding + answers |
   | `API_KEY` | optional — if set, also set it for the UI (the UI sends it as `X-API-Key`) |
   | `QDRANT_URL`, `QDRANT_API_KEY` | optional — Qdrant Cloud; default is in-container `:memory:` |
   | `REDIS_URL` | optional — e.g. Upstash / Redis from the Vercel Marketplace |
   | `DATABASE_URL` | optional — hosted Postgres (`postgresql+psycopg://…`, add `psycopg[binary]` to requirements) |

4. Redeploy so the variables take effect. Open `/api/ready` to check the index loaded.

### Things to know (and to mention if asked)

- **Port:** containers listen on `$PORT`, default 80 — Vercel routes to port 80 unless you set `PORT`.
- **Stateless:** containers scale to zero after ~5 minutes idle. With the default `:memory:` Qdrant each
  cold start re-indexes (~1–2 s with the lite backend). Feedback in SQLite under `/tmp` is lost on scale
  down — use hosted Postgres if you want to keep it.
- **Streamlit needs WebSockets.** Vercel Functions support WebSockets (public beta, Python included),
  but a connection closes at the function's max duration (300 s on Hobby, up to 800 s on Pro). Streamlit
  reconnects automatically; if that's disruptive in a demo, host the UI on Hugging Face Spaces (Docker
  Space, `docker/ui.Dockerfile`) or Streamlit Community Cloud and keep the API on Vercel.
- **Image size:** keep `EMBEDDING_BACKEND=lite` on Vercel. The `clip` backend means a multi-GB image
  (PyTorch + model weights) and slow cold starts; run that variant on Kubernetes or a GPU host.
- **Limitations of container functions today:** Secure Compute and static IPs aren't supported yet.

Vercel references: [Container Runtime](https://vercel.com/docs/functions/runtimes/container) ·
[Docker on Vercel](https://vercel.com/kb/guide/does-vercel-support-docker-deployments) ·
[WebSockets](https://vercel.com/kb/guide/do-vercel-serverless-functions-support-websocket-connections) ·
[Function limits](https://vercel.com/docs/functions/limitations)

---

## 2. Docker Compose (local full stack)

```bash
cp .env.example .env               # optional: add ANTHROPIC_API_KEY
docker compose up --build
# with monitoring:
docker compose --profile monitoring up --build
```

Order: `qdrant` (healthy) → `ingest` (one-shot ETL + indexing, exits 0) → `api` (waits for ingest and
redis) → `ui`. Grafana at `:3000` (admin/admin) has the "Boot Search Agent" dashboard pre-provisioned.

CLIP backend: `EMBEDDING_BACKEND=clip docker compose up --build` (the image downloads the models at
build time so containers start without internet access).

---

## 3. Kubernetes (EKS / GKE / AKS)

```bash
# images are pushed to GHCR by CI on every push to main
sed -i 's/YOUR_GITHUB_USER/<you>/' k8s/*.yaml
kubectl apply -f k8s/00-namespace-config.yaml -f k8s/10-qdrant-redis.yaml
kubectl apply -f k8s/20-ingest-job.yaml && kubectl -n boot-search wait --for=condition=complete job/boot-ingest
kubectl apply -f k8s/30-api.yaml -f k8s/40-ui-ingress.yaml
```

What's in there: Qdrant **StatefulSet** with a PVC, Redis, ingestion **Job**, API **Deployment** with
liveness/readiness/startup probes, rolling updates with `maxUnavailable: 0`, an **HPA** on CPU, Prometheus
scrape annotations, and an NGINX **Ingress** with a long read timeout for Streamlit's WebSocket.

On AWS specifically: EKS for the cluster, ECR or GHCR for images, Secrets Manager + External Secrets for
`ANTHROPIC_API_KEY`, RDS Postgres for `DATABASE_URL`, ElastiCache for Redis, Qdrant Cloud on AWS or the
StatefulSet on gp3 volumes. A lighter AWS option is App Runner or ECS Fargate running the same API image.

---

## 4. CI/CD

`.github/workflows/ci.yml`:

1. **test** — ruff, pytest (includes the retrieval quality gate), eval table in the job summary,
   `report.json` uploaded as an artifact.
2. **docker** — builds both images with layer caching; pushes `:sha` and `:latest` to GHCR on `main`.
3. **deploy-vercel** — optional; enable by setting the repo variable `DEPLOY_WITH_VERCEL_CLI=true` and
   secrets `VERCEL_TOKEN`, `VERCEL_ORG_ID`, `VERCEL_PROJECT_ID`. Skip it if you use Vercel's Git
   integration (it deploys on push by itself).
