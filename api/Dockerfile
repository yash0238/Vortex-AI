# VORTEX-AI live scenario API (CG-NSDE final model) for the research dashboard's "Generate fresh scenarios" button.
FROM python:3.12-slim
ENV OMP_NUM_THREADS=1 MALLOC_ARENA_MAX=2
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir numpy pandas scipy scikit-learn torchsde "fastapi>=0.110" "uvicorn[standard]" pydantic networkx yfinance torch_geometric
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
COPY --chown=user . .
USER user
ENV HOME=/home/user PYTHONUNBUFFERED=1
EXPOSE 7860
CMD uvicorn api.server:app --host 0.0.0.0 --port ${PORT:-7860}
