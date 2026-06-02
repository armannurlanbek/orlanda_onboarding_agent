root@karashn:/opt/platform# docker logs --tail=50 platform
INFO  [alembic.runtime.migration] Context impl PostgresqlImpl.
INFO  [alembic.runtime.migration] Will assume transactional DDL.
/usr/local/lib/python3.11/site-packages/langgraph/checkpoint/serde/encrypted.py:5: LangChainPendingDeprecationWarning: The default value of `allowed_objects` will change in a future version. Pass an explicit value (e.g., allowed_objects='messages' or allowed_objects='core') to suppress this warning.
  from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
INFO:     Started server process [71]
INFO:     Waiting for application startup.
{"time": "2026-05-08T05:45:57", "level": "INFO", "logger": "rag_agent.api", "message": "Active chat model: anthropic:claude-sonnet-4-6"}
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
root@karashn:/opt/platform# curl -sv http://127.0.0.1:8001/ 2>&1 | head -30
*   Trying 127.0.0.1:8001...
* Connected to 127.0.0.1 (127.0.0.1) port 8001
> GET / HTTP/1.1
> Host: 127.0.0.1:8001
> User-Agent: curl/8.5.0
> Accept: */*
>
< HTTP/1.1 200 OK
< date: Fri, 08 May 2026 05:49:22 GMT
< server: uvicorn
< content-length: 1286
< content-type: text/html; charset=utf-8
< x-frame-options: DENY
< x-content-type-options: nosniff
< referrer-policy: strict-origin-when-cross-origin
< content-security-policy: default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'
< permissions-policy: geolocation=(), microphone=(), camera=()
<
{ [1286 bytes data]
* Connection #0 to host 127.0.0.1 left intact
<!doctype html>
<html lang="ru">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Orlanda — База знаний и AI-ассистент</title>
    <meta name="description" content="Внутренняя платформа Orlanda Engineering: AI-ассистент и база знаний." />
    <meta name="author" content="Orlanda Engineering" />

    <meta property="og:title" content="Orlanda — База знаний и AI-ассистент" />
root@karashn:/opt/platform# cat /etc/cloudflared/config.yml
tunnel: ca2ee98b-7451-4cdf-955c-c63a48398caf
credentials-file: /root/.cloudflared/ca2ee98b-7451-4cdf-955c-c63a48398caf.json

ingress:
  - hostname: agent.n8norlanda.com
    service: http://127.0.0.1:8080
  - hostname: n8n.n8norlanda.com
    service: http://127.0.0.1:5678
  - hostname: platform.n8norlanda.com
    service: http://127.0.0.1:8001
  - service: http_status:404
root@karashn:/opt/platform#