"""
api/index.py  –  Vercel Python Serverless Function
Orange3 .pkcls 파일을 받아 sklearn 모델을 추출하고 예측/분석 결과를 반환합니다.
지원 모델: RandomForest, LinearRegression, Ridge, Lasso, kNN, SVR, GradientBoosting, etc.
"""

from __future__ import annotations
import sys, types, pickle, io, json
from typing import Any

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ──────────────────────────────────────────────────────────────
# Orange3 mock  (GUI/Qt 없이 pickle 로드용)
# ──────────────────────────────────────────────────────────────
def _setup_orange_mocks():
    """Orange3 모듈을 최소 mock으로 등록해 pkcls pickle을 로드."""
    if "Orange" in sys.modules:
        return

    class AutoAttr:
        """어떤 속성 접근·호출에도 자기 자신을 반환하는 dummy 객체."""
        def __init__(self, *a, **kw): pass
        def __call__(self, *a, **kw): return AutoAttr()
        def __getattr__(self, n): return AutoAttr
        def __or__(self, o): return self
        def __ror__(self, o): return self

    AutoAttr.Flags   = AutoAttr
    AutoAttr.Palette = AutoAttr

    class FakeMod(types.ModuleType):
        def __getattr__(self, _): return AutoAttr

    # Orange 패키지 트리
    orange_pkg = types.ModuleType("Orange")
    orange_pkg.__path__ = []
    sys.modules["Orange"] = orange_pkg

    sub_mods = [
        "Orange.data", "Orange.data.domain", "Orange.data.table",
        "Orange.data.variable", "Orange.preprocess", "Orange.preprocess.impute",
        "Orange.regression", "Orange.regression.random_forest",
        "Orange.classification", "Orange.modelling",
        "Orange.widgets", "Orange.widgets.utils",
    ]
    for nm in sub_mods:
        m = types.ModuleType(nm); m.__path__ = []
        sys.modules[nm] = m
        parts = nm.split(".")
        if len(parts) > 1:
            setattr(sys.modules[".".join(parts[:-1])], parts[-1], m)

    # colorpalettes & flags 모듈: Palette.Flags 접근 처리
    for nm in ["Orange.widgets.utils.colorpalettes", "flags"]:
        fm = FakeMod(nm); fm.__path__ = []
        sys.modules[nm] = fm
        if "." in nm:
            parts = nm.split(".")
            setattr(sys.modules[".".join(parts[:-1])], parts[-1], fm)

    # 변수 클래스
    class CV:
        def __init__(self, *a, **kw): self.name = a[0] if a else ""
        def __repr__(self): return self.name

    def make_variable(*a, **kw):
        v = CV(); v.name = a[0] if a else ""; return v

    class MockDomain:
        def __init__(self, *a, **kw):
            self.attributes = list(a[0]) if a else []
            self.class_var  = a[1] if len(a) > 1 else None
            self.metas      = kw.get("metas", [])

    class MockRFR:
        def __init__(self, *a, **kw): pass

    dv = sys.modules["Orange.data.variable"]
    for attr, val in [("ContinuousVariable", CV), ("DiscreteVariable", CV),
                      ("StringVariable", CV), ("make_variable", make_variable)]:
        setattr(dv, attr, val)

    sys.modules["Orange.data.domain"].Domain = MockDomain
    sys.modules["Orange.data.table"].Table   = AutoAttr

    for a in ["Average", "ReplaceUnknowns", "ReplaceUnknownsModel"]:
        setattr(sys.modules["Orange.preprocess.impute"], a, AutoAttr)

    # 모든 Orange 회귀/분류 모델 mock
    for mod_name in ["Orange.regression.random_forest",
                     "Orange.regression", "Orange.classification", "Orange.modelling"]:
        for cls_name in ["RandomForestRegressor","LinearRegressionLearner",
                         "RidgeRegressionLearner","LassoRegressionLearner",
                         "KNNRegressionLearner","SVRLearner",
                         "GBRegressionLearner","NNRegressionLearner",
                         "TreeRegressionLearner","SGDRegressionLearner",
                         "RandomForestClassifier","LogisticRegressionLearner",
                         "KNNLearner","SVMLearner","GBClassifier"]:
            setattr(sys.modules[mod_name], cls_name, MockRFR)

_setup_orange_mocks()

# ──────────────────────────────────────────────────────────────
# sklearn 모델 추출 헬퍼
# ──────────────────────────────────────────────────────────────
def extract_sklearn_model(orange_obj: Any):
    """Orange wrapper에서 sklearn estimator를 꺼냄."""
    for attr in ["skl_model", "model", "estimator", "clf", "reg"]:
        m = getattr(orange_obj, attr, None)
        if m is not None and hasattr(m, "predict"):
            return m
    # 자체가 sklearn 모델인 경우
    if hasattr(orange_obj, "predict"):
        return orange_obj
    raise ValueError("sklearn 모델을 찾을 수 없습니다.")


def get_feature_names(orange_obj: Any) -> list[str]:
    """Orange domain에서 feature 이름 추출."""
    domain = getattr(orange_obj, "domain", None)
    if domain is None:
        return []
    attrs = getattr(domain, "attributes", [])
    return [getattr(a, "name", str(a)) for a in attrs]


def get_target_name(orange_obj: Any) -> str:
    domain = getattr(orange_obj, "domain", None)
    if domain is None:
        return "target"
    cv = getattr(domain, "class_var", None)
    return getattr(cv, "name", "target") if cv else "target"


def model_type_name(skl_model) -> str:
    return type(skl_model).__name__


def is_classifier(skl_model) -> bool:
    return hasattr(skl_model, "predict_proba")


# ──────────────────────────────────────────────────────────────
# 부분 의존도 곡선 계산
# ──────────────────────────────────────────────────────────────
def compute_pd_curves(skl_model, features: list[str],
                      medians: list[float], n_points: int = 60) -> dict:
    """각 feature별 PD curve (다른 feature는 중앙값 고정)."""
    n = len(features)
    curves = {}
    for i, feat in enumerate(features):
        # feature 범위 추정: ±2σ 수준으로 설정
        med = medians[i]
        lo, hi = med * 0.3, med * 2.5
        lo = max(lo, 0.0)
        xs = np.linspace(lo, hi, n_points)
        X = np.tile(medians, (n_points, 1))
        X[:, i] = xs
        ys = skl_model.predict(X)
        curves[feat] = {
            "x": [round(float(v), 3) for v in xs],
            "y": [round(float(v), 3) for v in ys],
        }
    return curves


# ──────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────
app = FastAPI(title="Orange Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/upload")
async def upload_model(file: UploadFile = File(...)):
    """
    .pkcls 파일 업로드 → 모델 분석 결과 반환.
    반환: features, target, model_type, feature_importances, pd_curves, medians
    """
    if not file.filename.endswith(".pkcls"):
        raise HTTPException(400, "pkcls 파일만 업로드 가능합니다.")

    raw = await file.read()
    if len(raw) > 50 * 1024 * 1024:  # 50MB 제한
        raise HTTPException(400, "파일이 너무 큽니다 (최대 50MB).")

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            obj = pickle.loads(raw)
        except Exception as e:
            raise HTTPException(422, f"모델 로드 실패: {e}")

    try:
        skl = extract_sklearn_model(obj)
    except ValueError as e:
        raise HTTPException(422, str(e))

    features   = get_feature_names(obj)
    target     = get_target_name(obj)
    mtype      = model_type_name(skl)
    n_features = getattr(skl, "n_features_in_", len(features))

    # feature 이름이 없으면 자동 생성
    if len(features) != n_features:
        features = [f"feature_{i}" for i in range(n_features)]

    # 중앙값 추정 (모델 내부에 훈련 데이터 없으므로 구간 중앙값 사용)
    medians = [1.0] * n_features  # fallback

    # feature importance
    fi = None
    if hasattr(skl, "feature_importances_"):
        fi = [round(float(v), 6) for v in skl.feature_importances_]
    elif hasattr(skl, "coef_"):
        coef = skl.coef_.ravel()
        total = float(np.sum(np.abs(coef))) or 1.0
        fi = [round(float(abs(c)) / total, 6) for c in coef]

    # 안전한 테스트 예측으로 범위 추정
    try:
        X_test = np.ones((1, n_features))
        _ = skl.predict(X_test)
        # PD curves: 각 feature를 0.5~2배 범위에서 sweep
        curves = compute_pd_curves(skl, features, medians)
    except Exception as e:
        curves = {}

    return JSONResponse({
        "ok": True,
        "features": features,
        "target": target,
        "model_type": mtype,
        "n_features": n_features,
        "is_classifier": is_classifier(skl),
        "feature_importances": fi,
        "pd_curves": curves,
        "medians": medians,
        "params": {k: str(v) for k, v in
                   (skl.get_params().items() if hasattr(skl, "get_params") else {})},
    })


class PredictRequest(BaseModel):
    values: list[float]   # feature 값 목록 (순서: features 기준)
    # 업로드된 모델은 서버 메모리에 없으므로 raw bytes를 base64로 함께 전송
    model_b64: str        # base64 인코딩된 pkcls bytes


@app.post("/api/predict")
async def predict(req: PredictRequest):
    """단일 입력벡터 → 예측값 반환."""
    import base64, warnings

    try:
        raw = base64.b64decode(req.model_b64)
    except Exception:
        raise HTTPException(400, "model_b64 디코딩 실패")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            obj = pickle.loads(raw)
        except Exception as e:
            raise HTTPException(422, f"모델 로드 실패: {e}")

    skl = extract_sklearn_model(obj)
    X   = np.array([req.values])

    try:
        pred = skl.predict(X)
        result = float(pred.ravel()[0])
    except Exception as e:
        raise HTTPException(500, f"예측 실패: {e}")

    proba = None
    if is_classifier(skl):
        try:
            proba_arr = skl.predict_proba(X)
            proba = [round(float(p), 4) for p in proba_arr.ravel()]
        except Exception:
            pass

    return JSONResponse({"ok": True, "prediction": round(result, 4), "proba": proba})


@app.post("/api/sweep")
async def sweep(req: PredictRequest):
    """
    특정 feature를 범위 sweep → 1D 예측 배열 반환.
    req.values: [feature_index, lo, hi, n_points, *fixed_values]
    """
    import base64, warnings

    try:
        raw = base64.b64decode(req.model_b64)
    except Exception:
        raise HTTPException(400, "model_b64 디코딩 실패")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            obj = pickle.loads(raw)
        except Exception as e:
            raise HTTPException(422, f"모델 로드 실패: {e}")

    skl = extract_sklearn_model(obj)

    feat_idx  = int(req.values[0])
    lo        = float(req.values[1])
    hi        = float(req.values[2])
    n_points  = int(req.values[3])
    fixed     = list(req.values[4:])

    xs = np.linspace(lo, hi, n_points)
    X  = np.tile(fixed, (n_points, 1))
    X[:, feat_idx] = xs

    try:
        ys = skl.predict(X)
    except Exception as e:
        raise HTTPException(500, f"sweep 실패: {e}")

    return JSONResponse({
        "ok": True,
        "x":  [round(float(v), 4) for v in xs],
        "y":  [round(float(v), 4) for v in ys],
    })


# Vercel serverless entry point
from mangum import Mangum
handler = Mangum(app)
