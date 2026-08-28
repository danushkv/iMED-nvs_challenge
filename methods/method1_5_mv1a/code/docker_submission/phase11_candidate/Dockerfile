# Build from the method1_rgbd_reprojection directory:
#   docker build -f docker_submission/phase11_candidate/Dockerfile \
#     -t method1-rgbd-reprojection:phase11-candidate .
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    IMED_NVS_DEVICE=cuda

WORKDIR /app

COPY docker_submission/phase11_candidate/requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir -r /app/requirements.txt

# Reuse only the validated Method-1 inference geometry and selected H2 fill.
# The Docker build context is the Method-1 root, not the Endo-4DGS root.
COPY camera.py reprojection.py soft_splatting.py hole_filling.py /app/
COPY docker_submission/phase11_candidate/imed_io.py /app/imed_io.py
COPY docker_submission/phase11_candidate/nvs_method.py /app/nvs_method.py
COPY docker_submission/phase11_candidate/entrypoint.py /app/entrypoint.py

ENTRYPOINT ["python", "/app/entrypoint.py"]
CMD ["--input", "/input", "--output", "/output"]
