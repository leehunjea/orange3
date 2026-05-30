# Orange3 Model Predictor

Orange3에서 학습한 `.pkcls` 모델 파일을 업로드해 실시간 예측 및 시각화를 제공하는 웹앱입니다.

## 지원 모델

| 모델 | Orange3 이름 |
|------|-------------|
| Random Forest | RandomForestRegressor / Classifier |
| Linear Regression | LinearRegressionLearner |
| Ridge / Lasso | RidgeRegressionLearner / LassoRegressionLearner |
| k-Nearest Neighbors | KNNRegressionLearner |
| SVR / SVM | SVRLearner / SVMLearner |
| Gradient Boosting | GBRegressionLearner |
| Neural Network | NNRegressionLearner |

## 프로젝트 구조

```
housing-predictor/
├── api/
│   └── index.py        # FastAPI 서버리스 함수 (Vercel Python Runtime)
├── public/
│   └── index.html      # 프론트엔드 (정적 파일)
├── requirements.txt    # Python 의존성
├── vercel.json         # Vercel 배포 설정
└── README.md
```

## Vercel 배포 방법

### 1. GitHub에 올리기

```bash
git init
git add .
git commit -m "init: Orange3 model predictor"
git remote add origin https://github.com/<your-name>/orange-predictor.git
git push -u origin main
```

### 2. Vercel에서 배포

1. [vercel.com](https://vercel.com) 접속 → **Add New Project**
2. GitHub 저장소 선택
3. **Framework Preset**: Other
4. **Root Directory**: `.` (기본값)
5. **Deploy** 클릭

Vercel이 `vercel.json`을 읽어 자동으로:
- `api/index.py` → Python 서버리스 함수
- `public/` → 정적 파일 CDN 배포

### 3. 완료

배포 후 생성된 URL(예: `https://your-app.vercel.app`)에 접속해서 바로 사용 가능합니다.

---

## 로컬 테스트

```bash
pip install -r requirements.txt uvicorn

# API 서버 실행
uvicorn api.index:app --reload --port 8000

# 브라우저에서 public/index.html 열기
# (또는 live-server로 public/ 서빙)
```

> **참고**: 로컬에서 테스트할 때는 `public/index.html`의 `const API = ''` 부분을
> `const API = 'http://localhost:8000'` 으로 변경하세요.
> Vercel 배포 시에는 원래대로(`''`) 되돌리세요.

---

## 기술 스택

- **Backend**: FastAPI + Mangum (Vercel 서버리스 어댑터)
- **Frontend**: Vanilla JS + Chart.js
- **모델 로드**: pickle + Orange3 mock (sklearn 모델 직접 추출)
- **배포**: Vercel (Python Runtime + Static CDN)
