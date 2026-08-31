"""ONNX Runtime 세션 생성·캐싱·언로드.

원본 PathoView processing/inference.py 의 아래 부분을 그대로 옮긴 것:
  maybe_preload_onnxruntime_cuda / get_execution_providers / _looks_like_gpu_resource_error
  / _create_onnx_session / _ModelSessionManager / get_mitosis_session

동작이 바뀐 부분은 없고, 예외만 errors.py 의 것으로 바꿨다.
"""
from __future__ import annotations

import gc
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import onnxruntime as ort

from .config import (
    DEFAULT_MODEL_TTL_SECONDS,
    MODEL_PATH,
    MODEL_SWEEP_INTERVAL_SECONDS,
)
from .errors import GpuResourceError, ModelNotFoundError


def maybe_preload_onnxruntime_cuda() -> None:
    preload_dlls = getattr(ort, "preload_dlls", None)
    if callable(preload_dlls):
        try:
            preload_dlls()
        except Exception as exc:  # pragma: no cover - env dependent
            print("onnxruntime.preload_dlls() failed: ", exc)


def get_execution_providers(prefer_gpu: bool = True) -> List[Any]:
    maybe_preload_onnxruntime_cuda()

    available = ort.get_available_providers()
    print("ONNX Runtime version", getattr(ort, "__version__", "unknown"))
    print("ONNX device", ort.get_device())
    print("ONNX available_providers", available)

    providers: List[Any] = []
    cuda_device_id = int(os.getenv("ORT_CUDA_DEVICE_ID", os.getenv("CUDA_DEVICE_ID", "0")))

    use_tensorrt = os.getenv("ORT_USE_TENSORRT", "0") == "1"
    if prefer_gpu and use_tensorrt and "TensorrtExecutionProvider" in available:
        providers.append(
            (
                "TensorrtExecutionProvider",
                {
                    "device_id": cuda_device_id,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": os.getenv("ORT_TRT_CACHE_PATH", "/tmp/ort_trt_cache"),
                },
            )
        )

    if prefer_gpu and "CUDAExecutionProvider" in available:
        providers.append(
            (
                "CUDAExecutionProvider",
                {
                    "device_id": cuda_device_id,
                    "arena_extend_strategy": "kSameAsRequested",
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": "1",
                    "cudnn_conv_use_max_workspace": "1",
                },
            )
        )

    providers.append("CPUExecutionProvider")
    return providers


# ONNX Runtime/CUDA가 내는 GPU 자원 관련 에러 메시지에 흔히 등장하는 표지들.
# (예: "CUDNN failure 4000: CUDNN_STATUS_INTERNAL_ERROR", "CUDA out of memory")
_GPU_ERROR_MARKERS = (
    "cudnn",
    "cublas",
    "cuda error",
    "cuda failure",
    "out of memory",
    "failed to allocate",
    "cudnn_status",
    "cuda_error",
    "gpu=",
)


def _looks_like_gpu_resource_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _GPU_ERROR_MARKERS)


def _create_onnx_session(model_path: str) -> ort.InferenceSession:
    path = Path(model_path)
    if not path.exists():
        raise ModelNotFoundError(f"ONNX model not found: {path}")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = get_execution_providers(prefer_gpu=True)
    session = ort.InferenceSession(
        str(path),
        sess_options=sess_options,
        providers=providers,
    )

    active_providers = session.get_providers()
    print("Created ONNX session model", path.name)
    print("active_providers", active_providers)

    require_gpu = os.getenv("ORT_REQUIRE_GPU", "0") == "1"
    if require_gpu and "CUDAExecutionProvider" not in active_providers and "TensorrtExecutionProvider" not in active_providers:
        raise RuntimeError(
            "GPU is required but CUDA/TensorRT execution provider is not active. "
            f"available={ort.get_available_providers()} active={active_providers}"
        )

    return session


class ModelSessionManager:
    """ONNX 세션을 lazy 로드하고 idle-timeout(TTL) 이후 언로드하여 GPU 메모리를 관리한다.

    - 세션은 최초 요청 시 생성되어 재사용되고, 마지막 사용 후 TTL이 지나면 백그라운드
      스윕 스레드가 언로드한다(참조 제거 + gc.collect 로 GPU 메모리 반환).
    - 세션 생성이 GPU 자원 문제로 실패하면 다른 모델을 모두 언로드해 메모리를 확보한 뒤
      1회 재시도한다. 재시도도 실패하면 GpuResourceError 를 던진다.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._sessions: Dict[str, ort.InferenceSession] = {}
        self._last_used: Dict[str, datetime] = {}
        self._ttl: Dict[str, float] = {}
        self._sweeper_started = False

    def _ensure_sweeper(self) -> None:
        if self._sweeper_started:
            return
        self._sweeper_started = True
        thread = threading.Thread(
            target=self._sweep_loop, name="model-idle-sweeper", daemon=True
        )
        thread.start()

    def _sweep_loop(self) -> None:
        while True:
            time.sleep(MODEL_SWEEP_INTERVAL_SECONDS)
            try:
                self.evict_idle()
            except Exception as exc:  # pragma: no cover - 방어적 로깅
                print("[model-manager] idle sweep failed:", exc)

    def get(self, model_path: str, ttl_seconds: float) -> ort.InferenceSession:
        with self._lock:
            self._ensure_sweeper()
            # 사용 시각/TTL 갱신 (요청마다 최신 keep-alive 값을 반영)
            self._last_used[model_path] = datetime.now()
            self._ttl[model_path] = ttl_seconds

            session = self._sessions.get(model_path)
            if session is not None:
                return session

            try:
                session = _create_onnx_session(model_path)
            except ModelNotFoundError:
                raise
            except Exception as exc:
                if not _looks_like_gpu_resource_error(exc):
                    raise
                # GPU OOM 추정 → 다른 모델을 모두 언로드해 메모리 확보 후 1회 재시도
                print(
                    f"[model-manager] session load failed ({exc}); "
                    "freeing GPU memory and retrying once"
                )
                self._evict_all_locked(keep_path=model_path)
                try:
                    session = _create_onnx_session(model_path)
                except Exception as retry_exc:
                    if _looks_like_gpu_resource_error(retry_exc):
                        raise GpuResourceError(str(retry_exc)) from retry_exc
                    raise

            self._sessions[model_path] = session
            return session

    def evict_idle(self) -> None:
        now = datetime.now()
        with self._lock:
            stale = []
            for path, last in list(self._last_used.items()):
                if path not in self._sessions:
                    continue
                ttl = self._ttl.get(path, DEFAULT_MODEL_TTL_SECONDS)
                if ttl <= 0:
                    continue  # 0 이하 = 영구 상주(언로드하지 않음)
                if (now - last).total_seconds() >= ttl:
                    stale.append(path)
            for path in stale:
                self._unload_locked(path)

    def _evict_all_locked(self, keep_path: Optional[str] = None) -> None:
        for path in [p for p in list(self._sessions) if p != keep_path]:
            self._unload_locked(path)

    def _unload_locked(self, model_path: str) -> None:
        session = self._sessions.pop(model_path, None)
        self._last_used.pop(model_path, None)
        self._ttl.pop(model_path, None)
        if session is not None:
            del session
            gc.collect()
            print(f"[model-manager] unloaded model: {Path(model_path).name}")


_session_manager = ModelSessionManager()


def get_mitosis_session(
    model_path: str = MODEL_PATH,
    ttl_seconds: float = DEFAULT_MODEL_TTL_SECONDS,
) -> ort.InferenceSession:
    """mitosis ONNX 세션을 얻는다 (없으면 생성, 있으면 재사용)."""
    return _session_manager.get(model_path, ttl_seconds)
