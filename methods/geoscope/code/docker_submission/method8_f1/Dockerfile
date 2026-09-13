ARG MV1A_BASE=docker.synapse.org/syn74277461/nct_tso-mv1a-nvs:latest
ARG METHOD2_BASE=docker.synapse.org/syn74277461/imed-nvs-baseline:v1

# Deliberately copy the exact utilities used by the frozen M3B image rather
# than reimplementing camera or calibration geometry in this submission.
FROM ${MV1A_BASE} AS mv1a-runtime

FROM ${METHOD2_BASE}

LABEL org.opencontainers.image.title="Method 8 F1: M3B plus Method 2 holes" \
      org.opencontainers.image.description="Frozen M3B surface rendering with strict hole-only Method-2 fallback" \
      org.opencontainers.image.base.method2="docker.synapse.org/syn74277461/imed-nvs-baseline:v1" \
      org.opencontainers.image.base.m3b-utils="docker.synapse.org/syn74277461/nct_tso-mv1a-nvs:latest" \
      imed.method8.variant="F1_strict_holes" \
      imed.method3.renderer="surface" \
      imed.method2.depth-loss="metric_l1" \
      imed.method2.depth-weight="5e-5"

# Preserve the accepted Method-2 multi-architecture CUDA rebuild, including
# native A100 sm_80 code and PTX fallback at the newest target.
ENV CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX" \
    MAX_JOBS=4 \
    ENDO4DGS_REPO=/workspace/Endo-4DGS \
    METHOD3_RENDERER=surface \
    IMED_NVS_DEVICE=cuda \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /workspace/Endo-4DGS

RUN cd submodules/simple-knn \
    && rm -rf build dist *.egg-info \
    && python -m pip install . --no-build-isolation --no-deps --force-reinstall \
    && cd ../diff-gaussian-rasterization-depth \
    && rm -rf build dist *.egg-info \
    && python -m pip install . --no-build-isolation --no-deps --force-reinstall

# Accepted Method-2 source delta only.
COPY method2/train.py /workspace/Endo-4DGS/train.py
COPY method2/arguments/__init__.py /workspace/Endo-4DGS/arguments/__init__.py
COPY method2/scene/imed_loader.py /workspace/Endo-4DGS/scene/imed_loader.py
COPY method2/imed_nvs_baseline.py /workspace/Endo-4DGS/imed_nvs_baseline.py

# Exact frozen MV1A utilities inherited by the production M3B image.
RUN mkdir -p /app/method8_m3b/configs
COPY --from=mv1a-runtime /app/camera.py /app/method8_m3b/camera.py
COPY --from=mv1a-runtime /app/reprojection.py /app/method8_m3b/reprojection.py
COPY --from=mv1a-runtime /app/soft_splatting.py /app/method8_m3b/soft_splatting.py
COPY --from=mv1a-runtime /app/hole_filling.py /app/method8_m3b/hole_filling.py
COPY --from=mv1a-runtime /app/imed_io.py /app/method8_m3b/imed_io.py

# Frozen M3B production modules. Confidence code remains an implementation
# dependency of surface geometry, but confidence-aware rendering is disabled.
COPY m3b/confidence.py /app/method8_m3b/confidence.py
COPY m3b/surface_geometry.py /app/method8_m3b/surface_geometry.py
COPY m3b/surface_splat.py /app/method8_m3b/surface_splat.py
COPY m3b/configs/m3_surface_default.json /app/method8_m3b/configs/m3_surface_default.json
COPY f1_fusion.py /app/f1_fusion.py
COPY combined_entrypoint.py /app/method8_entrypoint.py

# Fail the build if any copied frozen source differs from the audited files.
RUN echo "adcba1b40e0c76e6ae39cd3e81b92edfe36480e1874f4e42c67e6602979443c3  /workspace/Endo-4DGS/train.py" | sha256sum -c - \
    && echo "0049d6239f2c939b775ce27fc0f1738c3fa29f3190ea43a522827c4419a50cc2  /workspace/Endo-4DGS/arguments/__init__.py" | sha256sum -c - \
    && echo "76f64bf45fd77a041a0aee84fde97332861ac3f904063cf3c01bfb9673ed89c3  /workspace/Endo-4DGS/scene/imed_loader.py" | sha256sum -c - \
    && echo "d363c2984290b93f3daedd17538b174aa1cff03c96f81568c86be2032f835fde  /workspace/Endo-4DGS/imed_nvs_baseline.py" | sha256sum -c - \
    && echo "708ce410ee9c38f9b58a4951f2ef84abbaf07da2484fc64853faf15a8bbc6e8e  /app/method8_m3b/confidence.py" | sha256sum -c - \
    && echo "f678b71f5fd1606fa29b5fbfa673e83442d91229c4f40e4236a36ab841b3f8c1  /app/method8_m3b/surface_geometry.py" | sha256sum -c - \
    && echo "aae69eb981cf85d9bc697332920c8182332ff069c07d2acbec383ab2b309d2da  /app/method8_m3b/surface_splat.py" | sha256sum -c - \
    && echo "7bfd3b2195a69105184cad020a9f7788d6263b88dcfb46d981431143d9f7327c  /app/method8_m3b/configs/m3_surface_default.json" | sha256sum -c - \
    && PYTHONPATH=/app/method8_m3b python -c "import camera,hole_filling,imed_io,reprojection,soft_splatting,surface_splat; print('Frozen M3B modules import successfully')" \
    && sha256sum \
        /workspace/Endo-4DGS/train.py \
        /workspace/Endo-4DGS/arguments/__init__.py \
        /workspace/Endo-4DGS/scene/imed_loader.py \
        /workspace/Endo-4DGS/imed_nvs_baseline.py \
        /app/method8_m3b/*.py \
        /app/method8_m3b/configs/m3_surface_default.json \
        /app/f1_fusion.py \
        /app/method8_entrypoint.py \
        > /app/METHOD8_SOURCE_MANIFEST.sha256

WORKDIR /app
ENTRYPOINT ["python", "/app/method8_entrypoint.py"]
CMD ["--input", "/input", "--output", "/output"]
