FROM python:3.11-slim-bookworm

ENV MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/mpl

RUN apt-get update && \
    apt-get install -y --no-install-recommends git g++ zip unzip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN useradd -m -u 1000 mluser
USER mluser

WORKDIR /workspace
