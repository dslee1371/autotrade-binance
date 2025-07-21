# 베이스 이미지
FROM python:3.10-slim

# 작업 디렉토리
WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    gcc \
    default-libmysqlclient-dev \
    build-essential \
    curl \
    && apt-get clean

# 파이썬 패키지 의존성 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 환경변수 로딩
ENV PYTHONUNBUFFERED=1

# 기본 실행 명령
CMD ["python", "autotrade.py"]

