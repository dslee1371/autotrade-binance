# autotrade-binance

Python 기반의 AI 연동 Binance 선물 자동 트레이딩 시스템입니다.

## 🪪 License

This project is licensed under the [MIT License](LICENSE).

---

## ⚙️ 환경변수 설정 (.env)

자동 트레이딩 시스템의 동작을 위해 다음 환경변수를 설정해야 합니다. 민감한 정보는 `.env` 파일에 저장하고 `git` 커밋에서 제외해야 합니다 (`.gitignore`에 추가 필요).

```env
# ✅ OpenAI 또는 Local AI API
OPENAI_API_KEY="sk-..."                        # OpenAI API 키
LOCAL_AI_KEY="sk-..."                          # 로컬 LLM API 키
LOCAL_AI_URL="http://localhost:3000/api/chat/completions"  # 로컬 AI 라마3.1 엔드포인트

# ✅ Binance API
BINANCE_API_KEY="your_binance_api_key"
BINANCE_SECRET_KEY="your_binance_secret_key"

# ✅ 검색/뉴스 API
SEARCHAPI_API_KEY="your_search_api_key"        # 외부 뉴스 검색용 API 키

# ✅ Slack Webhook
SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."  # 슬랙 경고 전송 URL

# ✅ MySQL 연결
MYSQL_USER="user1"
MYSQL_PASSWORD="P@ssw0rd"                      # 일반 터미널 사용 시
MYSQL_PASSWORD1="P%40ssw0rd"                   # Streamlit에서 URL 인코딩된 비밀번호 사용 시

## 도커 이미지 만들기
docker build -t autotrade-binance:v1.0 .
