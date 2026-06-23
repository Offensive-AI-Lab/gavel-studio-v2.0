# GAVEL Central Server

The one shared server every GAVEL backend connects to for **login, ratings,
bookmarks, and publishing to HuggingFace**. You deploy it **once**; everyone
points their backend at it.

It holds the user accounts, the token-signing key, and the HuggingFace write
token. Models and training data never touch it — those stay on each user's own
machine.

---

## Deploy it (Docker — the easy way)

On the server (a Linux VM is ideal) you need **Docker** and **git**.
On a fresh Linux box, installing Docker is a one-liner:
`curl -fsSL https://get.docker.com | sh`

**1. Get the code onto the server.** SSH in, then clone the repo and enter this
folder:

```bash
git clone https://github.com/OfekAvi/gavel-cloud-platform.git
cd gavel-cloud-platform/central-server
```

**2. Configure.** Copy the env file, then open `.env` and set two values:

```bash
cp .env.example .env
```

```bash
# a signing secret — generate one with:
#   python -c "import secrets; print(secrets.token_urlsafe(48))"
JWT_SECRET_KEY=...

# a HuggingFace WRITE token (huggingface.co → Settings → Access Tokens):
HF_TOKEN=hf_...
```

Both are required — the server signs logins with `JWT_SECRET_KEY` and uses
`HF_TOKEN` for all HuggingFace library access (publishing and sync).

> **Want instant library updates?** Also set `REGISTRY_WEBHOOK_SECRET` and add
> an HF webhook — see [Real-time updates](#real-time-updates-instant-updates-available)
> below. It's optional: without it, updates still arrive within a few minutes.

**3. Start it:**

```bash
docker compose up -d --build
```

This runs the server **and its own database** and serves on
**http://localhost:8001**. Verify:

```bash
curl http://localhost:8001/health      # → {"status":"ok"}
```

Stop it with `docker compose down` (add `-v` to also wipe its database).

> **Going public?** Put it behind a domain with HTTPS (a reverse proxy like
> nginx/Caddy, or a cloud load balancer) and set `ALLOWED_ORIGINS` in `.env`.

### Keep it running 24/7

**Nothing extra to do — `docker compose up -d` IS the 24/7 setup.** Run it once
and it stays up forever:

- `-d` keeps it running after you log out / disconnect (no `screen`/`tmux`/`nohup`).
- `restart: unless-stopped` (already in the compose) restarts it after a crash
  or a reboot.

On **Linux** that's the whole story — Docker starts on boot, so it comes back by
itself. (On **Windows**, just set Docker Desktop to start when you sign in.)

Check it: `docker compose ps` · live logs: `docker compose logs -f` · update after
a code change: `docker compose up -d --build`.

---

## Point the backends at it

In every user's `backend/.env`, set the URL — nothing else:

```bash
CENTRAL_SERVER_URL=https://your-central-server     # or http://localhost:8001 to test locally
```

The backend has **no** signing secret; it verifies logins by calling this
server. To move the central server later, change just this one URL.

---

## Real-time updates (instant "Updates available")

The central server runs a **control plane**: it watches the HF library and pushes
a `version_update` to every connected backend the moment something is published,
so users see an **"Updates available"** badge with no polling. It's on by default
— nothing to enable. The only question is **how fast** an update is detected:

- **Out of the box:** within a few minutes — a low-frequency safety poll
  (`REGISTRY_SAFETY_POLL_S`, default 300s) checks HuggingFace and broadcasts.
- **Instant:** add a HuggingFace **webhook** so HF tells the server the moment a
  publish lands. Two steps:

**1. Set a webhook secret** in `central-server/.env`, then restart the server:

```bash
# generate one:
python -c "import secrets; print(secrets.token_urlsafe(32))"
```
```bash
REGISTRY_WEBHOOK_SECRET=<that-value>
```

**2. Create the webhook** at <https://huggingface.co/settings/webhooks> →
*Add a new webhook*:

| Field | Value |
|---|---|
| **Target repository** | `GavelPublicData/public-library` (type **Dataset**) |
| **Webhook URL** | `https://<your-central-server>/api/v1/webhook` |
| **Secret** | the same `REGISTRY_WEBHOOK_SECRET` value |
| **Triggers** | **Repo update** (leave *Community* off) |

Save, then use HF's **Test** button (or just publish something) — a successful
delivery shows up as `POST /api/v1/webhook ... 200` in the server log, and from
then on every publish flips users' badges **instantly**.

> HF delivers webhooks over **HTTPS**, so the server must be reachable on a real
> `https://` URL (put it behind a reverse proxy / load balancer with TLS — the
> Docker "Going public?" note covers this). A plain-`http` server will accept the
> request if reachable, but HF may refuse to deliver to a non-HTTPS target.
>
> Verify the endpoint + secret yourself any time:
> ```bash
> curl -i -X POST https://<your-central-server>/api/v1/webhook \
>   -H "X-Webhook-Secret: <secret>" -H "Content-Type: application/json" -d "{}"
> # → 200 {"ok":true}  (a wrong/missing secret returns 401)
> ```

---

## Config (`central-server/.env`)

| Variable | Required? | What it's for |
|---|---|---|
| `JWT_SECRET_KEY` | **yes** | Signs login tokens. Any long random string; keep it secret. |
| `HF_TOKEN` | **yes** | HuggingFace **write** token. Used for all HF library access — publishing AND the sync check (`/hf/*` returns 503 without it). |
| `ALLOWED_ORIGINS` | optional | Comma-separated browser origins allowed to call it (CORS). |
| `REGISTRY_WEBHOOK_SECRET` | optional | Shared secret for the HF webhook (see [Real-time updates](#real-time-updates-instant-updates-available)). Set it for **instant** updates; without it, the safety poll still keeps everyone converged within a few minutes. |
| `DATABASE_URL` | ignored with Docker | The Docker setup runs its own Postgres. Only set this for the manual run below. |

> Advanced control-plane tuning (`ENABLE_CONTROL_PLANE`, `REGISTRY_SAFETY_POLL_S`,
> `REGISTRY_DEBOUNCE_S`, `REGISTRY_HF_TIMEOUT_S`, `REGISTRY_WS_MAX_CONN`) is
> documented inline in `.env.example`; the defaults are fine for most deployments.

---

## Alternatives

**Without Docker** — only if the machine can't run Docker. You bring your own
**Python 3.12** and **Postgres**, then:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # set DATABASE_URL + JWT_SECRET_KEY + HF_TOKEN
python _create_db.py          # one-time: creates the DB named in DATABASE_URL
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

That runs in the foreground (stops when you close the terminal). To keep it up
24/7 you'd run it under a service manager — **systemd** on Linux, **NSSM** or
Task Scheduler on Windows. (This is exactly the plumbing Docker does for you, so
unless you truly can't use Docker, use the Docker steps above instead.)

**Render** (managed hosting): `render.yaml` provisions the web service + a
Postgres automatically. New → Blueprint → connect this repo, then set
`JWT_SECRET_KEY` and `HF_TOKEN` (and optionally `ALLOWED_ORIGINS`) in the
service's Environment tab.
