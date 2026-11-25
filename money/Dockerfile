# VORTEX Money: one container = the consumer web app + its API (accounts, sync, cached fund data).
FROM node:22-slim AS web
WORKDIR /web
COPY app/package.json app/package-lock.json ./
RUN npm ci
COPY app/ ./
RUN npx vite build

FROM python:3.12-slim
RUN pip install --no-cache-dir "fastapi>=0.110" "uvicorn[standard]" pydantic
RUN useradd -m -u 1000 user && mkdir -p /data && chown user /data
WORKDIR /home/user/srv
COPY --chown=user money_api/ money_api/
COPY --from=web --chown=user /web/dist/ app/dist/
COPY --chown=user app/public/money/engine.json app/public/money/engine.json
# MONEY_SECRET comes from a Space secret. The free tier has no persistent disk, so /data resets on restart.
ENV MONEY_DB=/data/money.sqlite3 MONEY_DIST=/home/user/srv/app/dist \
    MONEY_ENGINE=/home/user/srv/app/public/money/engine.json \
    MONEY_CORS=https://yash0238.github.io PYTHONUNBUFFERED=1
USER user
EXPOSE 7860
CMD uvicorn money_api.app:app --host 0.0.0.0 --port ${PORT:-7860} --proxy-headers --forwarded-allow-ips "*"
