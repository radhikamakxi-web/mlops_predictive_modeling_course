FROM python:3.12-slim

ARG TARGETARCH=amd64

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL "https://dl.min.io/server/minio/release/linux-${TARGETARCH}/minio" -o /usr/local/bin/minio \
    && chmod +x /usr/local/bin/minio \
    && mkdir -p /data

EXPOSE 9000 9001

ENTRYPOINT ["/usr/local/bin/minio"]
CMD ["server", "/data", "--console-address", ":9001"]
