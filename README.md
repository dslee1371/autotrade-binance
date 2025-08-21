# autotrade-binance

Python 기반의 AI 연동 Binance 선물 자동 트레이딩 시스템입니다.

## 🪪 License

This project is licensed under the [MIT License](LICENSE).

---

## ⚙️ 환경변수 설정 (.env)

자동 트레이딩 시스템의 동작을 위해 다음 환경변수를 설정해야 합니다. 민감한 정보는 `.env` 파일에 저장하고 `git` 커밋에서 제외해야 합니다 (`.gitignore`에 추가 필요).

```env
# ✅ OpenAI 또는 Local AI API
OPENAI_API_KEY=sk-...                        # OpenAI API 키
LOCAL_AI_KEY=sk-...                          # 로컬 LLM API 키
LOCAL_AI_URL=http://localhost:3000/api/chat/completions  # 로컬 AI 라마3.1 엔드포인트

# ✅ Binance API
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key

# ✅ 검색/뉴스 API
SEARCHAPI_API_KEY=your_search_api_key        # 외부 뉴스 검색용 API 키

# ✅ Slack Webhook
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...  # 슬랙 경고 전송 URL

# ✅ MySQL 연결
MYSQL_USER=user1
MYSQL_PASSWORD=P@ssw0rd                     # 일반 터미널 사용 시
MYSQL_PASSWORD1=P%40ssw0rd                 # Streamlit에서 URL 인코딩된 비밀번호 사용 시

** 주의 사항 ** 따옴표(")를 사용하면 오류 발생
```
### 환경변수으로 스크릿 만들기
```
kubectl create secret generic autotrade-binance-secret \
  --from-env-file=.env \
  -n coinauto
```

## 도커 이미지 만들기
```
# 도커 이미지 빌드
docker build -t autotrade-binance:v0.1 .

# 도커 태크
docker tag autotrade-binance:v0.1 172.10.30.11:5000/auto-coin/autotrade-binance:v0.1

# 도커 푸쉬
docker push 172.10.30.11:5000/auto-coin/autotrade-binance:v0.1
```
## 헬름차트로 mysql 설치하기
```
# values.yaml 수정
auth:
  rootPassword: admin1234
  database: autotrade_db
  username: app_user
  password: app_pass123


# 설치
helm install mysql . -f values.yaml -n coinauto
```
## Deployment
```
apiVersion: apps/v1
kind: Deployment
metadata:
  name: autotrade-binance
  namespace: coinauto
  labels:
    app: autotrade-binance
spec:
  replicas: 1
  selector:
    matchLabels:
      app: autotrade-binance
  template:
    metadata:
      labels:
        app: autotrade-binance
    spec:
      containers:
        - name: autotrade-binance
          image: 172.10.30.11:5000/auto-coin/autotrade-binance:v0.1
          imagePullPolicy: IfNotPresent
          envFrom:
          - secretRef:
              name: autotrade-binance-secret
```

## 배포하기
```
kubectl create -f deployment.yaml
```

#애플리케이션 설명
## 사용 기술 및 라이브러리 구성
ccxt: 바이낸스 선물 API 연동 (포지션/가격/잔고 관리)

mysql: 데이터 저장 (트레이드 및 계정 이력) 

pandas, numpy: 시계열 데이터 및 기술적 지표 계산

requests: 외부 AI 모델 및 뉴스 API 통신

dotenv: API 키 및 민감 정보 환경변수로 관리


## 데이터베이스 구조 및 초기화
초기 실행 시 아래 테이블을 생성합니다:

trades: 거래 상태 기록 (open/closed, 가격, 수량 등)

trade_results: 개별 거래의 성과 기록

account_history: 계정 잔액 히스토리


## 트레이드 기록 및 계정 히스토리 함수
save_trade(): 포지션 진입 정보 기록

close_trade(): 포지션 종료 및 수익 계산

update_account_history(): 주기적 잔액 저장

get_active_trade_info(): 현재 오픈된 거래 정보 조회

get_last_open_trade_id(): 최근 트레이드 ID 반환

## 성과 분석 및 통계 계산
- get_recent_trade_history(n=20): 최근 N건 거래 내역

- get_trading_performance_stats()

  + 전체 승률, 평균 손익, 누적 PnL, MDD 등

- 추가 분석:

  + 시간대별 승률 (ex. 오전 vs 오후)

  + 변동성 별 승률 (low vs high volatility)
 
## 기술적 지표 분석 로직

- get_recent_trade_history(n=20): 최근 N건 거래 내역
- get_trading_performance_stats()
  + 전체 승률, 평균 손익, 누적 PnL, MDD 등
- 추가 분석:
  + 시간대별 승률 (ex. 오전 vs 오후)
  + 변동성 별 승률 (low vs high volatility)
 
## AI 모델 연동 및 응답 파싱
- LLaMA 3.1 WebUI 포맷으로 POST 요청
- 응답 예측 로그 기록: inspect_api_response()
- 파싱 로직:
  + parse_ai_response()를 통해 "long (75%)" 등 방향성과 확률 추출

