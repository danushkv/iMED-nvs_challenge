ARG MV1A_BASE=docker.synapse.org/syn74277461/nct_tso-mv1a-nvs:latest
FROM ${MV1A_BASE}

ARG METHOD3_RENDERER=surface_confidence
RUN case "${METHOD3_RENDERER}" in \
        surface|surface_confidence) ;; \
        *) echo "METHOD3_RENDERER must be surface or surface_confidence" >&2; exit 2 ;; \
    esac

ENV METHOD3_RENDERER=${METHOD3_RENDERER} \
    IMED_NVS_DEVICE=cuda \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

LABEL org.opencontainers.image.title="Method 3 Surface Fusion" \
      org.opencontainers.image.description="iMED-NVS source-only bounded surface splatting" \
      org.opencontainers.image.base.name="docker.synapse.org/syn74277461/nct_tso-mv1a-nvs:latest" \
      org.opencontainers.image.base.expected-id="aaa952c110d1" \
      imed.method3.renderer=${METHOD3_RENDERER}

WORKDIR /app

# The confirmed MV1A parent supplies camera.py, reprojection.py,
# soft_splatting.py, hole_filling.py, and the inference-only imed_io.py.
# Copy only Method-3 production code and immutable configuration records.
COPY confidence.py surface_geometry.py surface_splat.py /app/
COPY configs/m3_surface_default.json /app/configs/m3_surface_default.json
COPY configs/m3_confidence_default.json /app/configs/m3_confidence_default.json
COPY docker_submission/method3_candidates/submission_adapter.py /app/submission_adapter.py

ENTRYPOINT ["python", "/app/submission_adapter.py"]
CMD ["--input", "/input", "--output", "/output"]
