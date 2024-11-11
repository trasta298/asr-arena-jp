FROM python:3.10-slim as build

RUN apt update && apt install -y build-essential curl git
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY uv.lock pyproject.toml ./
RUN uv sync --frozen

FROM python:3.10-slim

WORKDIR /workspace

COPY --from=build /.venv /opt/venv

RUN apt update

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH "${PYTHONPATH}:/workspace"

COPY . .

CMD ["python", "main.py"]
