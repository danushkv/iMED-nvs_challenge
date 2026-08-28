FROM docker.synapse.org/syn74277461/imed-nvs-baseline:v1

LABEL org.opencontainers.image.title="Method 2: Endo-4DGS plus metric L1 depth"
LABEL org.opencontainers.image.description="iMED-NVS Task-2 candidate: metric L1 depth supervision at weight 5e-5"

# Rebuild both CUDA extensions for a portable evaluator image. CUDA 11.8
# supports these native targets, and PTX at the newest target gives a forward-
# compatibility fallback. This is intentionally not an sm86-only image.
ENV CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX" \
    MAX_JOBS=4 \
    ENDO4DGS_REPO=/workspace/Endo-4DGS \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /workspace/Endo-4DGS

RUN cd submodules/simple-knn \
    && rm -rf build dist *.egg-info \
    && python -m pip install . --no-build-isolation --no-deps --force-reinstall \
    && cd ../diff-gaussian-rasterization-depth \
    && rm -rf build dist *.egg-info \
    && python -m pip install . --no-build-isolation --no-deps --force-reinstall

# Copy only the completed Method-2 files required by the accepted metric-depth
# configuration. The Dockerfile-specific ignore file makes this an allowlisted
# context even though the build runs from the repository root.
COPY train.py /workspace/Endo-4DGS/train.py
COPY arguments/__init__.py /workspace/Endo-4DGS/arguments/__init__.py
COPY scene/imed_loader.py /workspace/Endo-4DGS/scene/imed_loader.py
COPY imed_nvs_baseline.py /workspace/Endo-4DGS/imed_nvs_baseline.py
COPY docker_submission/method2_metric_l1/submission_entrypoint.py \
     /workspace/Endo-4DGS/docker_submission_entrypoint.py

ENTRYPOINT ["python", "/workspace/Endo-4DGS/docker_submission_entrypoint.py"]
CMD ["--input", "/input", "--output", "/output"]
