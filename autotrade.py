import ccxt
import os
import math
import time
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
from datetime import datetime, timedelta

try:
    import mysql.connector as db_driver
    from mysql.connector import Error
    DRIVER = 'mysql-connector'
except ImportError:
    import pymysql as db_driver
    Error = db_driver.MySQLError if hasattr(db_driver, 'MySQLError') else Exception
    DRIVER = 'pymysql'


# ─── 1. MySQL 접속 정보 (환경변수 MYSQL, PASSWORD 사용) ───
# MySQL 연결 정보 (환경변수로 관리하세요)
DB_CONFIG = {
    'host': 'mysql',
    'port': 3306,
    'user': os.getenv("MYSQL_USER"),          # ex) 'myuser'
    'password': os.getenv("MYSQL_PASSWORD"),  # ex) 'mypassword'
    'database': 'mydb',
    # JDBC 옵션에 대응
    'ssl_disabled': True,
    'auth_plugin': 'mysql_native_password'
}

def get_connection():
    """MySQL 데이터베이스 연결을 리턴"""
    try:
        if DRIVER == 'mysql-connector':
            return db_driver.connect(
                host=DB_CONFIG['host'],
                port=DB_CONFIG['port'],
                user=DB_CONFIG['user'],
                password=DB_CONFIG['password'],
                database=DB_CONFIG['database'],
                auth_plugin='mysql_native_password'
            )
        else:
            return db_driver.connect(
                host=DB_CONFIG['host'],
                port=DB_CONFIG['port'],
                user=DB_CONFIG['user'],
                password=DB_CONFIG['password'],
                database=DB_CONFIG['database'],
                charset='utf8mb4',
                cursorclass=db_driver.cursors.DictCursor
            )
    except Error as e:
        print(f"Error connecting to MySQL ({DRIVER}): {e}")
        raise



# 바이낸스 세팅
api_key = os.getenv("BINANCE_API_KEY")
secret = os.getenv("BINANCE_SECRET_KEY")
local_ai_key = os.getenv("LOCAL_AI_KEY")
local_ai_url = os.getenv("LOCAL_AI_URL")
exchange = ccxt.binance({
    'apiKey': api_key,
    'secret': secret,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',
        'adjustForTimeDifference': True
    }
})
symbol = "BTC/USDT"

headers = {
    "Authorization": f"Bearer {local_ai_key}",
    "Content-Type": "application/json"
}

# 켈리 크리테리온 관련 매개변수
MAX_ACCOUNT_RISK = 0.1  # 최대 계정 리스크 (전체 자본의 10%)
BASE_ORDER_SIZE = 200  # 기본 주문 크기 (USDT)
MAX_LEVERAGE = 10  # 최대 레버리지


# ─── 2. 테이블 생성 함수 (MySQL 문법) ───
def setup_database():
    conn = get_connection()
    cursor = conn.cursor()
    # trades 테이블
    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS trades (
        id               INT AUTO_INCREMENT PRIMARY KEY,
        timestamp        DATETIME,
        action           VARCHAR(10),
        entry_price      DOUBLE,
        amount           DOUBLE,
        order_size       DOUBLE,
        leverage         INT,
        stop_loss        DOUBLE,
        take_profit      DOUBLE,
        kelly_fraction   DOUBLE,
        win_probability  DOUBLE,
        volatility       DOUBLE,
        status           VARCHAR(10) DEFAULT 'open'
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """)
    # trade_results 테이블
    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS trade_results (
        id            INT AUTO_INCREMENT PRIMARY KEY,
        trade_id      INT,
        close_timestamp DATETIME,
        close_price   DOUBLE,
        pnl           DOUBLE,
        pnl_percentage DOUBLE,
        duration      VARCHAR(50),
        result        VARCHAR(20),
        FOREIGN KEY (trade_id) REFERENCES trades(id)
            ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """)
    # account_history 테이블
    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS account_history (
        id             INT AUTO_INCREMENT PRIMARY KEY,
        timestamp      DATETIME,
        balance        DOUBLE,
        equity         DOUBLE,
        unrealized_pnl DOUBLE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """)
    conn.commit()
    conn.close()
    print("MySQL database setup complete.")

def save_trade(action, entry_price, amount, order_size, leverage,
               sl_price, tp_price, kelly_fraction, win_probability, volatility):
    # NumPy 타입을 파이썬 기본 타입으로 변환
    entry_price     = float(entry_price)
    amount          = float(amount)
    order_size      = float(order_size)
    leverage        = int(leverage)
    sl_price        = float(sl_price)
    tp_price        = float(tp_price)
    kelly_fraction  = float(kelly_fraction)
    win_probability = float(win_probability)
    volatility      = float(volatility)

    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute("""
        INSERT INTO trades
          (timestamp, action, entry_price, amount, order_size, leverage,
           stop_loss, take_profit, kelly_fraction, win_probability, volatility)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        now, action, entry_price, amount, order_size, leverage,
        sl_price, tp_price, kelly_fraction, win_probability, volatility
    ))
    conn.commit()
    conn.close()
    print(f"Trade saved with ID: {cursor.lastrowid}")
    return cursor.lastrowid


def close_trade(trade_id, close_price, result):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, action, entry_price, amount FROM trades WHERE id = %s", (trade_id,))
    row = cursor.fetchone()
    if not row:
        print(f"Trade ID {trade_id} not found.")
        conn.close()
        return
    open_ts, action, entry_price, amount = row
    if isinstance(open_ts, datetime):
        open_time = open_ts
    else:
        open_time = datetime.strptime(open_ts, '%Y-%m-%d %H:%M:%S')
    close_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    duration = str(datetime.now() - open_time)
    if action == 'long':
        pnl = (close_price - entry_price) * amount
        pnl_pct = ((close_price / entry_price) - 1) * 100
    else:
        pnl = (entry_price - close_price) * amount
        pnl_pct = ((entry_price / close_price) - 1) * 100
    cursor.execute("""
        INSERT INTO trade_results
          (trade_id, close_timestamp, close_price, pnl, pnl_percentage, duration, result)
        VALUES (%s,%s,%s,%s,%s,%s,%s)
    """, (trade_id, close_ts, close_price, pnl, pnl_pct, duration, result))
    cursor.execute("UPDATE trades SET status='closed' WHERE id=%s", (trade_id,))
    conn.commit()
    conn.close()
    print(f"Trade {trade_id} closed: PnL=${pnl:.2f} ({pnl_pct:.2f}%)")

def update_account_history(balance, equity=None, unrealized_pnl=0.0):
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if equity is None:
        equity = balance
    cursor.execute("""
        INSERT INTO account_history (timestamp, balance, equity, unrealized_pnl)
        VALUES (%s,%s,%s,%s)
    """, (now, balance, equity, unrealized_pnl))
    conn.commit()
    conn.close()

def get_last_open_trade_id():
    """마지막으로 오픈된 트레이드 ID 가져오기"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id FROM trades WHERE status = 'open' ORDER BY id DESC LIMIT 1"
    )
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

def get_active_trade_info():
    """현재 활성화된 트레이드 정보 가져오기"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, action, entry_price, amount, order_size, leverage,
               stop_loss, take_profit, timestamp
          FROM trades
         WHERE status = 'open'
      ORDER BY id DESC
         LIMIT 1
    """)
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    trade_id, action, entry_price, amount, order_size, leverage, sl, tp, ts = row
    if isinstance(ts, datetime):
        open_time = ts
    else:
        open_time = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
    return {
        'id': trade_id,
        'action': action,
        'entry_price': entry_price,
        'amount': amount,
        'order_size': order_size,
        'leverage': leverage,
        'stop_loss': sl,
        'take_profit': tp,
        'timestamp': ts,
        'duration': datetime.now() - open_time
    }

# 새로 추가한 함수: 최근 거래 내역 및 결과 가져오기
def get_recent_trade_history(limit=20):
    """최근 완료된 거래 내역 및 결과를 가져옵니다"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT t.id, t.timestamp, t.action, t.entry_price,
               tr.close_price, tr.pnl, tr.pnl_percentage, tr.result,
               t.win_probability, t.kelly_fraction, t.volatility, t.leverage
          FROM trades t
          JOIN trade_results tr ON t.id = tr.trade_id
         WHERE t.status = 'closed'
      ORDER BY tr.close_timestamp DESC
         LIMIT %s
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()

    history = []
    for (tid, ts, act, ep, cp, pnl, pnl_pct, res, wp, kf, vol, lev) in rows:
        history.append({
            'id':              tid,
            'timestamp':       ts,
            'action':          act,
            'entry_price':     ep,
            'close_price':     cp,
            'pnl':             pnl,
            'pnl_percentage':  pnl_pct,
            'result':          res,
            'win_probability': wp,
            'kelly_fraction':  kf,
            'volatility':      vol,
            'leverage':        lev
        })
    return history

# 새로 추가한 함수: 거래 성과 분석 통계
def get_trading_performance_stats():
    """거래 성과 통계를 계산하여 반환합니다"""
    conn = get_connection()
    cursor = conn.cursor()

    # 총 거래, 수익 및 손실
    cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'closed'")
    total = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM trade_results WHERE pnl > 0")
    wins = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM trade_results WHERE pnl <= 0")
    losses = cursor.fetchone()[0]

    # 평균 수익/손실, 누적 PnL
    cursor.execute("SELECT AVG(pnl) FROM trade_results WHERE pnl > 0")
    avg_profit = cursor.fetchone()[0] or 0
    cursor.execute("SELECT AVG(pnl) FROM trade_results WHERE pnl <= 0")
    avg_loss   = cursor.fetchone()[0] or 0
    cursor.execute("SELECT SUM(pnl) FROM trade_results")
    total_pnl  = cursor.fetchone()[0] or 0

    # 롱/숏 승률
    cursor.execute("""
        SELECT COUNT(*) FROM trades t
        JOIN trade_results tr ON t.id = tr.trade_id
         WHERE t.action='long' AND tr.pnl>0
    """)
    long_wins = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM trades WHERE action='long' AND status='closed'")
    total_long = cursor.fetchone()[0]
    long_wr = long_wins / total_long if total_long else 0

    cursor.execute("""
        SELECT COUNT(*) FROM trades t
        JOIN trade_results tr ON t.id = tr.trade_id
         WHERE t.action='short' AND tr.pnl>0
    """)
    short_wins = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM trades WHERE action='short' AND status='closed'")
    total_short = cursor.fetchone()[0]
    short_wr = short_wins / total_short if total_short else 0

    conn.close()
    return {
        'total_trades':       total,
        'profitable_trades':  wins,
        'losing_trades':      losses,
        'win_rate':           wins/total if total else 0,
        'avg_profit':         avg_profit,
        'avg_loss':           avg_loss,
        'total_pnl':          total_pnl,
        'long_win_rate':      long_wr,
        'short_win_rate':     short_wr,
        'total_long':         total_long,
        'total_short':        total_short
    }

# 새로 추가한 함수: 시간대별 승률 분석
def get_time_based_performance():
    """시간대별 거래 성과를 분석합니다 (4시간 간격)"""
    conn = get_connection()
    cursor = conn.cursor()
    perf = {}
    for start in range(0, 24, 4):
        end = start + 4
        # 전체 거래
        cursor.execute("""
            SELECT COUNT(*) FROM trades
             WHERE HOUR(timestamp) BETWEEN %s AND %s-1
               AND status='closed'
        """, (start, end))
        tot = cursor.fetchone()[0]
        # 수익 거래
        cursor.execute("""
            SELECT COUNT(*) FROM trades t
            JOIN trade_results tr ON t.id=tr.trade_id
             WHERE HOUR(t.timestamp) BETWEEN %s AND %s-1
               AND tr.pnl>0
        """, (start, end))
        win = cursor.fetchone()[0]
        perf[f"{start:02d}-{end:02d}"] = {
            'trades': tot,
            'wins':   win,
            'win_rate': win/tot if tot else 0
        }
    conn.close()
    return perf

# 새로 추가한 함수: 변동성 기반 성과 분석
def get_volatility_based_performance():
    """변동성 범위별 거래 성과를 분석합니다"""
    conn = get_connection()
    cursor = conn.cursor()
    ranges = [(0,1),(1,2),(2,3),(3,1e9)]
    perf = {}
    for i,(vmin,vmax) in enumerate(ranges,1):
        # 전체 거래
        cursor.execute("""
            SELECT COUNT(*) FROM trades
             WHERE volatility BETWEEN %s AND %s
               AND status='closed'
        """, (vmin, vmax))
        tot = cursor.fetchone()[0]
        # 수익 거래
        cursor.execute("""
            SELECT COUNT(*) FROM trades t
            JOIN trade_results tr ON t.id=tr.trade_id
             WHERE t.volatility BETWEEN %s AND %s
               AND tr.pnl>0
        """, (vmin, vmax))
        win = cursor.fetchone()[0]
        perf[f"vol_{i}"] = {
            'range':    f"{vmin}-{vmax if vmax<1e9 else 'inf'}",
            'trades':   tot,
            'wins':     win,
            'win_rate': win/tot if tot else 0
        }
    conn.close()
    return perf

print("\n=== Adaptive Bitcoin Trading Bot with Kelly Criterion Started ===")
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Trading Pair:", symbol)
print("Max Leverage:", MAX_LEVERAGE)
print("Base Order Size:", BASE_ORDER_SIZE, "USDT")
print("Max Account Risk:", MAX_ACCOUNT_RISK * 100, "%")
print("Timeframes: 15m (short), 1h (medium), 4h (long)")
print("Moving Averages: MA5, MA20, MA50, MA100")
print("News Source: Latest Breaking News from CryptoCompare")
print("Strategy: Kelly Criterion for optimal position sizing")
print(
    "Database: MySQL "
    f"(Host: {DB_CONFIG['host']}:{DB_CONFIG['port']}, "
    f"Database: {DB_CONFIG['database']})"
)
print("=========================================================\n")

# 데이터베이스 초기 설정
setup_database()

def calculate_kelly_criterion(win_probability, win_loss_ratio):
    """
    켈리 크리테리온을 계산하는 함수
    
    Parameters:
    - win_probability: 승리 확률 (0~1 사이의 값)
    - win_loss_ratio: 승리 시 이익과 손실 시 손실의 비율 (승리 시 이익 / 손실 시 손실)
    
    Returns:
    - 최적의 배팅 비율 (0~1 사이의 값)
    """
    # 승률이 0.5 미만이면 short 포지션으로 변경
    actual_win_prob = win_probability if win_probability >= 0.5 else 1 - win_probability
    
    # 켈리 공식: f* = (p*b - q) / b = (p*b - (1-p)) / b
    # p: 승리 확률, q: 패배 확률 (1-p), b: 승리/패배 비율
    kelly_fraction = (actual_win_prob * win_loss_ratio - (1 - actual_win_prob)) / win_loss_ratio
    
    # 켈리 값이 음수일 경우 배팅하지 않음 (0 반환)
    if kelly_fraction <= 0:
        return 0
    
    # Full Kelly는 변동성이 높으므로 Half Kelly 사용 (보수적 접근)
    half_kelly = kelly_fraction * 0.5
    
    # 계정 최대 리스크로 제한
    limited_kelly = min(half_kelly, MAX_ACCOUNT_RISK)
    
    return limited_kelly

def get_account_balance():
    """계정 잔액을 가져오는 함수"""
    try:
        balance = exchange.fetch_balance()
        usdt_balance = balance['USDT']['free']
        return usdt_balance
    except Exception as e:
        print(f"Error fetching account balance: {e}")
        return 10000  # 기본값 설정 (오류 시)

def get_latest_crypto_news(limit=10):
    """
    최신 암호화폐 속보 뉴스를 가져오는 함수
    
    Parameters:
    - limit: 가져올 뉴스 항목 수 (기본값: 10)
    
    Returns:
    - 뉴스 항목 목록 (각 항목은 딕셔너리 형태)
    """
    try:
        url = f"https://www.investing.com/rss/news_25.rss"
        
        print(f"Fetching latest breaking news from Investing.com API...")
        response = requests.get(url)
        
        if response.status_code == 200:
            # RSS 피드 파싱
            import xml.etree.ElementTree as ET
            tree = ET.fromstring(response.content)
            news_items = tree.findall('.//item')
            
            news_list = []
            for item in news_items:
                try:
                    pub_date = item.findtext('pubDate')
                    
                    # 다양한 날짜 형식 시도
                    try:
                        # 원래 형식 시도
                        published_date = datetime.strptime(pub_date, '%a, %d %b %Y %H:%M:%S +0000')
                    except ValueError:
                        try:
                            # 새로운 형식 시도
                            published_date = datetime.strptime(pub_date, '%Y-%m-%d %H:%M:%S')
                        except ValueError:
                            # 실패하면 날짜 문자열 그대로 사용
                            date_str = pub_date
                            published_date = None
                    
                    # 날짜를 문자열로 변환
                    if published_date:
                        date_str = published_date.strftime('%Y-%m-%d %H:%M:%S')
                    
                    # 제목, 날짜, 출처만 저장하여 토큰 절약
                    news_list.append({
                        'title': item.findtext('title'),
                        'date': date_str,
                        'source': 'Investing.com'
                    })
                except Exception as e:
                    print(f"Error parsing an item: {e}")
                    continue
            
            print(f"Retrieved {len(news_list)} latest news headlines")
            return news_list[:limit]  # limit 개수만큼만 반환
        else:
            print(f"API Error: Status code {response.status_code}")
            return []
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []
    

def display_news(news_list):
    """뉴스 목록을 보기 좋게 출력하는 함수"""
    if not news_list:
        print("No news available.")
        return
        
    print("\n===== 최신 암호화폐 뉴스 =====")
    for idx, item in enumerate(news_list, 1):
        print(f"{idx}. {item['title']}")
        print(f"   출처: {item['source']} ({item['date']})")
    print("=======================\n")

def calculate_market_volatility(df):
    """시장 변동성을 계산하는 함수"""
    # 간단한 가격 변동성 계산
    returns = df['close'].pct_change().dropna()
    volatility = returns.std() * 100  # 퍼센트로 변환
    return volatility

def get_dataframe_with_indicators(ohlcv_data, timeframe, limit=100):
    """
    OHLCV 데이터를 DataFrame으로 변환하고 이동평균선(MA)을 계산하는 함수
    
    Parameters:
    - ohlcv_data: OHLCV 데이터
    - timeframe: 시간대 문자열 (예: '15m', '1h', '4h')
    - limit: 반환할 캔들 수 (기본값: 100)
    
    Returns:
    - 이동평균선(MA)이 포함된 DataFrame
    """
    # 충분한 데이터가 있는지 확인 (최소 100개 이상의 캔들 필요)
    if len(ohlcv_data) < 100:
        print(f"경고: 충분한 데이터가 없습니다. 필요: 100, 현재: {len(ohlcv_data)}")
    
    # DataFrame 생성
    df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['timeframe'] = timeframe
    
    # 이동평균선(MA) 계산
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()
    df['MA100'] = df['close'].rolling(window=100).mean()
    
    # 지수 이동평균선(EMA) 계산
    df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    # MACD 계산
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    
    # RSI 계산 (14 기간)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 분석을 위해 제한된 수의 최근 캔들만 반환
    return df.tail(limit)

def get_simplified_dataframe(df, limit=20):
    """분석용 DataFrame을 간소화하여 토큰 절약"""
    # 중요 정보만 포함하여 DataFrame 축소
    simplified_df = df.tail(limit)[['timestamp', 'open', 'high', 'low', 'close', 'MA5', 'MA20', 'MA50', 'MA100', 'RSI', 'MACD', 'timeframe']]
    return simplified_df

def analyze_ma_crossovers(df):
    """이동평균선 교차점 분석"""
    current_close = df['close'].iloc[-1]
    ma5 = df['MA5'].iloc[-1]
    ma20 = df['MA20'].iloc[-1]
    ma50 = df['MA50'].iloc[-1]
    ma100 = df['MA100'].iloc[-1]
    
    # 가격과 MA의 관계
    price_above_ma5 = current_close > ma5
    price_above_ma20 = current_close > ma20
    price_above_ma50 = current_close > ma50
    price_above_ma100 = current_close > ma100
    
    # 최근 골든 크로스/데드 크로스 감지 (MA5 vs MA20)
    golden_cross = (df['MA5'].iloc[-2] <= df['MA20'].iloc[-2]) and (df['MA5'].iloc[-1] > df['MA20'].iloc[-1])
    dead_cross = (df['MA5'].iloc[-2] >= df['MA20'].iloc[-2]) and (df['MA5'].iloc[-1] < df['MA20'].iloc[-1])
    
    # 추세 분석
    short_trend = "bullish" if price_above_ma5 and price_above_ma20 else "bearish" if not price_above_ma5 and not price_above_ma20 else "neutral"
    medium_trend = "bullish" if price_above_ma50 else "bearish"
    long_trend = "bullish" if price_above_ma100 else "bearish"
    
    # MA 값들의 기울기 확인 (상승/하락 추세 강도)
    ma5_slope = df['MA5'].iloc[-1] - df['MA5'].iloc[-5]
    ma20_slope = df['MA20'].iloc[-1] - df['MA20'].iloc[-5]
    
    # 분석 결과
    result = {
        'ma5': ma5,
        'ma20': ma20,
        'ma50': ma50,
        'ma100': ma100,
        'price_vs_ma5': "above" if price_above_ma5 else "below",
        'price_vs_ma20': "above" if price_above_ma20 else "below",
        'price_vs_ma50': "above" if price_above_ma50 else "below",
        'price_vs_ma100': "above" if price_above_ma100 else "below",
        'golden_cross': golden_cross,
        'dead_cross': dead_cross,
        'short_trend': short_trend,
        'medium_trend': medium_trend,
        'long_trend': long_trend,
        'ma5_slope': "rising" if ma5_slope > 0 else "falling",
        'ma20_slope': "rising" if ma20_slope > 0 else "falling"
    }
    
    return result

def get_trend_status_string(analysis):
    """이동평균선 분석 결과를 문자열로 반환"""
    status = f"MA5: ${analysis['ma5']:.2f} ({analysis['ma5_slope']}), "
    status += f"MA20: ${analysis['ma20']:.2f} ({analysis['ma20_slope']}), "
    status += f"MA50: ${analysis['ma50']:.2f}, MA100: ${analysis['ma100']:.2f}\n"
    
    status += f"Price vs MAs: MA5={analysis['price_vs_ma5']}, MA20={analysis['price_vs_ma20']}, "
    status += f"MA50={analysis['price_vs_ma50']}, MA100={analysis['price_vs_ma100']}\n"
    
    if analysis['golden_cross']:
        status += "⚠️ GOLDEN CROSS detected (MA5 crossed above MA20)\n"
    if analysis['dead_cross']:
        status += "⚠️ DEAD CROSS detected (MA5 crossed below MA20)\n"
    
    status += f"Trend Analysis: Short-term={analysis['short_trend']}, "
    status += f"Medium-term={analysis['medium_trend']}, Long-term={analysis['long_trend']}"
    
    return status

def parse_ai_response(response_text):
    """
    Enhanced AI response parser to extract action and confidence probability
    from various response formats.
    
    Returns:
    - (action, probability) tuple where action is 'long', 'short', or 'none'
      and probability is a float between 0 and 1
    """
    try:
        # Log the raw response for debugging
        print(f"Raw AI response to parse: {response_text}")
        
        # Clean the text
        cleaned_text = response_text.strip().lower()
        
        # Initialize default values
        action = 'none'
        probability = 0.0
        
        # Try multiple regex patterns to extract information
        import re
        
        # Pattern 1: **direction: action** and **probability: value**
        direction_match = re.search(r"\*\*direction:\s*(\w+)\*\*", cleaned_text)
        probability_match = re.search(r"\*\*probability:\s*([\d.]+)\*\*", cleaned_text)
        
        if direction_match and probability_match:
            action = direction_match.group(1).strip()
            try:
                probability = float(probability_match.group(1))
            except ValueError:
                print(f"Could not convert probability to float: {probability_match.group(1)}")
                probability = 0.5
                
        # Pattern 2: action:probability format
        elif ':' in cleaned_text:
            # Find all instances of "long:X.XX" or "short:X.XX"
            action_prob_matches = re.findall(r"(long|short):([\d.]+)", cleaned_text)
            
            if action_prob_matches:
                # Use the last instance if multiple found
                action, prob_str = action_prob_matches[-1]
                try:
                    probability = float(prob_str)
                except ValueError:
                    print(f"Could not convert probability to float: {prob_str}")
                    probability = 0.5
        
        # Pattern 3: Look for the words "long" or "short" and assign default probability
        else:
            long_matches = re.findall(r"\b(long)\b", cleaned_text)
            short_matches = re.findall(r"\b(short)\b", cleaned_text)
            
            long_count = len(long_matches)
            short_count = len(short_matches)
            
            if long_count > 0 and short_count > 0:
                # If both appear, check which appears last
                last_long_pos = cleaned_text.rfind('long')
                last_short_pos = cleaned_text.rfind('short')
                
                if last_long_pos > last_short_pos:
                    action = 'long'
                else:
                    action = 'short'
                
                probability = 0.6  # Default confidence when both are mentioned
            elif long_count > 0:
                action = 'long'
                probability = 0.6
            elif short_count > 0:
                action = 'short'
                probability = 0.6
        
        # Validate action
        if action not in ['long', 'short', 'none']:
            print(f"Invalid action: {action}. Defaulting to 'none'.")
            action = 'none'
            probability = 0.0
            
        # Validate probability range
        if probability < 0 or probability > 1:
            print(f"Invalid probability: {probability}. Should be between 0 and 1. Using 0.5.")
            probability = 0.5 if action != 'none' else 0.0
            
        print(f"Parsed decision: {action.upper()} with {probability:.2%} confidence")
        return action, probability
        
    except Exception as e:
        print(f"Error parsing AI response: {e}")
        return 'none', 0.0
def inspect_api_response(response):
    """
    Inspects and logs the structure of an API response to help debug integration issues.
    
    Parameters:
    - response: The response object from requests
    
    Returns:
    - A dictionary containing useful information about the response
    """
    inspection_result = {
        'status_code': response.status_code,
        'headers': dict(response.headers),
        'content_type': response.headers.get('Content-Type', 'unknown'),
        'length': len(response.text),
        'is_json': False,
        'structure': None,
        'raw_text': response.text[:1000] + '...' if len(response.text) > 1000 else response.text  # Limit text length
    }
    
    # Try to parse as JSON
    try:
        json_data = response.json()
        inspection_result['is_json'] = True
        
        # Get the top-level keys
        inspection_result['json_keys'] = list(json_data.keys())
        
        # Create a simplified structure representation
        structure = {}
        
        def map_structure(obj, max_depth=3, current_depth=0):
            """Helper function to map the structure of nested objects"""
            if current_depth >= max_depth:
                return "..."
            
            if isinstance(obj, dict):
                return {k: map_structure(v, max_depth, current_depth + 1) for k, v in list(obj.items())[:5]}  # Limit to first 5 items
            elif isinstance(obj, list):
                if not obj:
                    return []
                if len(obj) == 1:
                    return [map_structure(obj[0], max_depth, current_depth + 1)]
                else:
                    return [map_structure(obj[0], max_depth, current_depth + 1), "..."]  # Just show the first item's structure
            elif isinstance(obj, (str, int, float, bool)) or obj is None:
                return type(obj).__name__
            else:
                return type(obj).__name__
        
        inspection_result['structure'] = map_structure(json_data)
        
        # Check for common response patterns
        if 'choices' in json_data and isinstance(json_data['choices'], list):
            inspection_result['openai_compatible'] = True
            
            choice = json_data['choices'][0] if json_data['choices'] else {}
            if 'message' in choice and 'content' in choice['message']:
                inspection_result['content_path'] = 'choices[0].message.content'
            elif 'text' in choice:
                inspection_result['content_path'] = 'choices[0].text'
        
        elif 'response' in json_data:
            inspection_result['content_path'] = 'response'
        
        # Log sample data for the first few keys
        sample_data = {}
        for key in list(json_data.keys())[:3]:  # First 3 keys
            value = json_data[key]
            if isinstance(value, (str, int, float, bool)) or value is None:
                sample_data[key] = value
            elif isinstance(value, list) and value:
                sample_data[key] = f"List with {len(value)} items, first item: {str(value[0])[:100]}"
            else:
                sample_data[key] = f"{type(value).__name__} with {len(value) if hasattr(value, '__len__') else 'unknown'} items"
        
        inspection_result['sample_data'] = sample_data
        
    except ValueError:
        # Not JSON
        inspection_result['is_json'] = False
    
    # Print inspection results
    print("\n=== API Response Inspection ===")
    print(f"Status Code: {inspection_result['status_code']}")
    print(f"Content-Type: {inspection_result['content_type']}")
    print(f"Response Length: {inspection_result['length']} characters")
    print(f"Is JSON: {inspection_result['is_json']}")
    
    if inspection_result['is_json']:
        print(f"Top-level keys: {', '.join(inspection_result['json_keys'])}")
        if 'content_path' in inspection_result:
            print(f"Likely content path: {inspection_result['content_path']}")
    
    print("Response structure:")
    import json
    print(json.dumps(inspection_result['structure'], indent=2))
    
    print("Raw response preview:")
    print(inspection_result['raw_text'][:500] + "..." if len(inspection_result['raw_text']) > 500 else inspection_result['raw_text'])
    print("==============================\n")
    
    return inspection_result

def sync_closed_by_exchange():
    """
    DB에는 open인데, 실제 거래소에선 포지션이 사라진 경우
    (거래소 SL/TP·강제청산·수동청산) 를 감지해서 강제 종료 처리합니다.
    """
    active = get_active_trade_info()
    if not active:
        return False

    # 바이낸스에서 현재 포지션 조회
    positions = exchange.fetch_positions([symbol])
    # BTC/USDT 포지션만 뽑아서 수량 확인
    amt = 0
    for p in positions:
        if p['symbol'] == 'BTC/USDT:USDT':
            amt = float(p['info']['positionAmt'])
            break

    # 포지션이 0 이면 이미 닫힌 상태
    if amt == 0:
        # 현재가 재조회
        current_price = exchange.fetch_ticker(symbol)['last']
        print(f"Exchange에서 포지션 사라짐 → Trade #{active['id']} 강제 종료")
        close_trade(active['id'], current_price, 'exchange_sl_tp')
        return True

    return False


def check_and_close_active_trades():
    # 0) 거래소 종료 반영 우선
    if sync_closed_by_exchange():
        return True

    # 1) 기존 로직: DB 기준 SL/TP 체크
    active_trade = get_active_trade_info()
    if not active_trade:
        return False

    ticker = exchange.fetch_ticker(symbol)
    current_price = ticker['last']
    should_close = False
    close_reason = ""

    if active_trade['action'] == 'long':
        if current_price <= active_trade['stop_loss']:
            should_close, close_reason = True, "stop_loss"
        elif current_price >= active_trade['take_profit']:
            should_close, close_reason = True, "take_profit"
    else:
        if current_price >= active_trade['stop_loss']:
            should_close, close_reason = True, "stop_loss"
        elif current_price <= active_trade['take_profit']:
            should_close, close_reason = True, "take_profit"

    if should_close:
        print(f"Closing trade #{active_trade['id']} at {current_price} due to {close_reason}")
        close_trade(active_trade['id'], current_price, close_reason)
        return True

    return False



# 활성 트레이드 ID 추적 변수
active_trade_id = None

while True:
    try:
        # 활성화된 트레이드 확인 및 필요시 종료
        check_and_close_active_trades()

        # 현재 시간 및 가격 조회
        current_time = datetime.now().strftime('%H:%M:%S')
        current_price = exchange.fetch_ticker(symbol)['last']
        print(f"\n[{current_time}] Current BTC Price: ${current_price:,.2f}")

        # 계정 잔액 확인
        account_balance = get_account_balance()
        print(f"Available Account Balance: ${account_balance:,.2f} USDT")
        
        # 일정 간격으로 계정 내역 저장
        if datetime.now().minute % 10 == 0 and datetime.now().second < 10:  # 10분마다 저장
            update_account_history(account_balance)

        # 포지션 확인
        current_side = None
        amount = 0
        positions = exchange.fetch_positions([symbol])
        for position in positions:
            if position['symbol'] == 'BTC/USDT:USDT':
                amt = float(position['info']['positionAmt'])
                if amt > 0:
                    current_side = 'long'
                    amount = amt
                elif amt < 0:
                    current_side = 'short'
                    amount = abs(amt)
        
        # 시장 변동성 분석을 위한 데이터 수집
        ohlcv_volatility = exchange.fetch_ohlcv("BTC/USDT", timeframe="1h", limit=24)  # 24시간 데이터
        df_volatility = pd.DataFrame(ohlcv_volatility, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        market_volatility = calculate_market_volatility(df_volatility)
        
        print(f"Market Volatility: {market_volatility:.2f}%")
        
        # 변동성에 따른 SL/TP 조정
        sl_tp_percentage = 0.5  # 기본 0.5%
        if market_volatility > 2.0:  # 변동성이 높을 경우
            sl_tp_percentage = 1.5  # SL/TP 범위 확장
            print(f"High volatility detected! Adjusting SL/TP to ±{sl_tp_percentage}%")
        elif market_volatility > 3.0:  # 매우 높은 변동성
            sl_tp_percentage = 2.0  # SL/TP 범위 더 확장
            print(f"Very high volatility detected! Adjusting SL/TP to ±{sl_tp_percentage}%")
        
        if current_side:
            print(f"Current Position: {current_side.upper()} {amount} BTC")
            
            # 활성 트레이드 정보 데이터베이스에서 조회
            active_trade = get_active_trade_info()
            if active_trade:
                print(f"Trade ID: {active_trade['id']}, Entry: ${active_trade['entry_price']:.2f}, Duration: {active_trade['duration']}")
                
                # 현재 PnL 계산
                if active_trade['action'] == 'long':
                    current_pnl = (current_price - active_trade['entry_price']) * active_trade['amount']
                    pnl_percentage = ((current_price / active_trade['entry_price']) - 1) * 100
                else:  # short
                    current_pnl = (active_trade['entry_price'] - current_price) * active_trade['amount']
                    pnl_percentage = ((active_trade['entry_price'] / current_price) - 1) * 100
                
                print(f"Current P&L: ${current_pnl:.2f} ({pnl_percentage:.2f}%)")
                
                # SL/TP 도달 확인 (실제로는 바이낸스가 처리하지만 데이터베이스 기록을 위해)
                if (active_trade['action'] == 'long' and current_price <= active_trade['stop_loss']) or \
                   (active_trade['action'] == 'short' and current_price >= active_trade['stop_loss']):
                    # Stop Loss 히트
                    print(f"⚠️ Stop Loss triggered at ${current_price:.2f}")
                    close_trade(active_trade['id'], current_price, 'stop_loss')
                    active_trade_id = None
                    
                elif (active_trade['action'] == 'long' and current_price >= active_trade['take_profit']) or \
                     (active_trade['action'] == 'short' and current_price <= active_trade['take_profit']):
                    # Take Profit 히트
                    print(f"✅ Take Profit triggered at ${current_price:.2f}")
                    close_trade(active_trade['id'], current_price, 'take_profit')
                    active_trade_id = None
        else:
            # 포지션이 없을 경우, 남아있는 미체결 주문 취소
            try:
                open_orders = exchange.fetch_open_orders(symbol)
                if open_orders:
                    for order in open_orders:
                        exchange.cancel_order(order['id'], symbol)
                    print("Cancelled remaining open orders for", symbol)
                else:
                    print("No remaining open orders to cancel.")
            except Exception as e:
                print("Error cancelling orders:", e)
            time.sleep(3)
            print("No position. Analyzing market...")

            # 단기(15분봉) 데이터 수집
            ohlcv_short = exchange.fetch_ohlcv("BTC/USDT", timeframe="15m", limit=150)  # 15분봉 (약 37시간 분량)
            df_short_full = get_dataframe_with_indicators(ohlcv_short, '15m')
            df_short = get_simplified_dataframe(df_short_full, 24)  # 최근 24개 캔들만 분석에 사용
            
            # 중기(1시간봉) 데이터 수집
            ohlcv_medium = exchange.fetch_ohlcv("BTC/USDT", timeframe="1h", limit=150)  # 1시간봉 (약 6일 분량)
            df_medium_full = get_dataframe_with_indicators(ohlcv_medium, '1h')
            df_medium = get_simplified_dataframe(df_medium_full, 12)  # 최근 12개 캔들만 분석에 사용
            
            # 장기(4시간봉) 데이터 수집
            ohlcv_long = exchange.fetch_ohlcv("BTC/USDT", timeframe="4h", limit=150)  # 4시간봉 (약 25일 분량)
            df_long_full = get_dataframe_with_indicators(ohlcv_long, '4h')
            df_long = get_simplified_dataframe(df_long_full, 6)  # 최근 6개 캔들만 분석에 사용
            
            # 이동평균선 분석
            short_term_ma_analysis = analyze_ma_crossovers(df_short_full)
            medium_term_ma_analysis = analyze_ma_crossovers(df_medium_full)
            long_term_ma_analysis = analyze_ma_crossovers(df_long_full)
            
            # 이동평균선 분석 결과 출력
            print("\n=== 15분봉 이동평균선 분석 ===")
            print(get_trend_status_string(short_term_ma_analysis))
            
            print("\n=== 1시간봉 이동평균선 분석 ===")
            print(get_trend_status_string(medium_term_ma_analysis))
            
            print("\n=== 4시간봉 이동평균선 분석 ===")
            print(get_trend_status_string(long_term_ma_analysis))
            
            # 최신 뉴스 데이터 수집 (10개)
            news_data = get_latest_crypto_news(limit=10)
            display_news(news_data)  # 수집된 뉴스 출력
            
            # 데이터 요약 (토큰 절약)
            short_term_summary = f"최근 {len(df_short)}개 15분봉: 시작가 ${df_short['open'].iloc[0]:.2f}, 현재가 ${df_short['close'].iloc[-1]:.2f}"
            medium_term_summary = f"최근 {len(df_medium)}개 1시간봉: 시작가 ${df_medium['open'].iloc[0]:.2f}, 현재가 ${df_medium['close'].iloc[-1]:.2f}"
            long_term_summary = f"최근 {len(df_long)}개 4시간봉: 시작가 ${df_long['open'].iloc[0]:.2f}, 현재가 ${df_long['close'].iloc[-1]:.2f}"
            
            # 가격 변화율 계산
            short_change = ((df_short['close'].iloc[-1] / df_short['open'].iloc[0]) - 1) * 100
            medium_change = ((df_medium['close'].iloc[-1] / df_medium['open'].iloc[0]) - 1) * 100
            long_change = ((df_long['close'].iloc[-1] / df_long['open'].iloc[0]) - 1) * 100
            
            price_changes = f"단기 변화율: {short_change:.2f}%, 중기 변화율: {medium_change:.2f}%, 장기 변화율: {long_change:.2f}%"
            
            # 추세선 데이터 요약 (MA 위치 및 크로스)
            short_term_trend = f"15분봉 MA: 가격 vs MA5({short_term_ma_analysis['price_vs_ma5']}), MA20({short_term_ma_analysis['price_vs_ma20']}), MA50({short_term_ma_analysis['price_vs_ma50']})"
            if short_term_ma_analysis['golden_cross']:
                short_term_trend += ", 골든크로스 감지(MA5>MA20)"
            if short_term_ma_analysis['dead_cross']:
                short_term_trend += ", 데드크로스 감지(MA5<MA20)"
                
            medium_term_trend = f"1시간봉 MA: 가격 vs MA5({medium_term_ma_analysis['price_vs_ma5']}), MA20({medium_term_ma_analysis['price_vs_ma20']}), MA50({medium_term_ma_analysis['price_vs_ma50']})"
            if medium_term_ma_analysis['golden_cross']:
                medium_term_trend += ", 골든크로스 감지(MA5>MA20)"
            if medium_term_ma_analysis['dead_cross']:
                medium_term_trend += ", 데드크로스 감지(MA5<MA20)"
            
            # 뉴스 제목만 추출 (토큰 절약)
            news_titles = "\n".join([f"- {item['title']} ({item['source']})" for item in news_data]) if news_data else "뉴스 데이터 없음"
            
            # RSI 및 MACD 값 확인
            short_term_rsi = df_short_full['RSI'].iloc[-1]
            short_term_macd = df_short_full['MACD'].iloc[-1]
            short_term_macd_signal = df_short_full['MACD_signal'].iloc[-1]
            short_term_macd_hist = df_short_full['MACD_hist'].iloc[-1]
            
            # 기술적 지표 요약
            technical_indicators = f"15분봉 RSI: {short_term_rsi:.2f} (과매수>70, 과매도<30), MACD: {short_term_macd:.2f}, Signal: {short_term_macd_signal:.2f}, Hist: {short_term_macd_hist:.2f}"
            
            # 새로 추가: 거래 내역 및 성과 분석을 가져오기
            recent_trades = get_recent_trade_history(10)  # 최근 10개 거래 내역
            trading_stats = get_trading_performance_stats()  # 거래 성과 통계
            time_performance = get_time_based_performance()  # 시간대별 성과
            volatility_performance = get_volatility_based_performance()  # 변동성 기반 성과
            
            # 거래 내역 요약
            trade_history_summary = ""
            if recent_trades:
                win_count = sum(1 for trade in recent_trades if trade['pnl'] > 0)
                loss_count = len(recent_trades) - win_count
                win_rate = win_count / len(recent_trades) if recent_trades else 0
                
                trade_history_summary = f"최근 {len(recent_trades)}개 거래 요약: 승리 {win_count}건, 패배 {loss_count}건, 승률 {win_rate:.2%}\n"
                
                # 최근 10개 거래 요약
                trade_history_summary += "거래 내역:\n"
                for i, trade in enumerate(recent_trades[:5], 1):  # 최근 5개만 표시
                    result_emoji = "✅" if trade['pnl'] > 0 else "❌"
                    trade_history_summary += f"{i}. {result_emoji} {trade['action'].upper()}, " + \
                                            f"진입가: ${trade['entry_price']:.2f}, " + \
                                            f"청산가: ${trade['close_price']:.2f}, " + \
                                            f"PnL: ${trade['pnl']:.2f} ({trade['pnl_percentage']:.2f}%), " + \
                                            f"예측확률: {trade['win_probability']:.2f}, " + \
                                            f"켈리비율: {trade['kelly_fraction']:.4f}\n"
            else:
                trade_history_summary = "거래 내역이 없습니다."
            
            # 거래 성과 통계 요약
            performance_summary = ""
            if trading_stats['total_trades'] > 0:
                performance_summary = f"총 거래: {trading_stats['total_trades']}건, 전체 승률: {trading_stats['win_rate']:.2%}, " + \
                                    f"평균 수익: ${trading_stats['avg_profit']:.2f}, 평균 손실: ${trading_stats['avg_loss']:.2f}, " + \
                                    f"누적 PnL: ${trading_stats['total_pnl']:.2f}\n" + \
                                    f"롱 승률: {trading_stats['long_win_rate']:.2%} ({trading_stats['total_long']}건), " + \
                                    f"숏 승률: {trading_stats['short_win_rate']:.2%} ({trading_stats['total_short']}건)"
            else:
                performance_summary = "아직 충분한 거래 데이터가 없습니다."
            
            # 시간대별 성과 요약
            time_summary = "시간대별 승률:\n"
            for time_slot, data in time_performance.items():
                if data['trades'] > 0:
                    time_summary += f"{time_slot}시: {data['win_rate']:.2%} ({data['wins']}/{data['trades']}), "
            
            # 변동성 기반 성과 요약
            volatility_summary = "변동성별 승률:\n"
            for vol_name, data in volatility_performance.items():
                if data['trades'] > 0:
                    volatility_summary += f"{data['range']}%: {data['win_rate']:.2%} ({data['wins']}/{data['trades']}), "
            

            # AI 분석 요청 - 추세선 데이터와 거래 내역 포함
            payload = {
            "model": "llama3.1:latest",  # ✅ 콜론(:) 사용
            "chatId": "weather-chat-001",  # ✅ chatID → chatId로 수정 (Open WebUI API 표준)
            "messages": [
                {
            "role": "system",
            "content": """You are an aggressive crypto trading expert using Kelly criterion to optimize position sizing. 
            Analyze market data, trend lines, latest breaking news, and past trading performance to predict Bitcoin price movement.

            IMPORTANT - REFLECT ON PAST TRADES: Analyze the provided trade history to learn what worked and what didn't. 
            Recognize patterns of success and failure, and adapt your strategy accordingly.

            Consider:
            1. Which timeframes have had better predictive success
            2. Whether long or short trades have been more successful
            3. How volatility affects trade outcomes
            4. If certain times of day yield better results
            5. How accurate your previous win probability estimations were

            Adapt your confidence level based on this historical performance.

            IMPORTANT - RESPONSE FORMAT:
            You must respond with a direction and probability in the following EXACT format at the end of your analysis:
            **direction: long** or **direction: short**
            **probability: 0.XX** (where 0.XX is a number between 0.5 and 1.0)

            For example:
            **direction: long**
            **probability: 0.75**

            Or:
            **direction: short**
            **probability: 0.67**

            Probability must be between a value of 0.5 and 1.0, representing your confidence level in the prediction.
            Higher values indicate stronger signals. A probability of 0.5 indicates no edge (random chance)."""
                    },
                    {
                        "role": "user",
                        "content": f"""Bitcoin price analysis:

            {short_term_summary}
            {medium_term_summary}
            {long_term_summary}

            {price_changes}

            Trend Line Analysis:
            {short_term_trend}
            {medium_term_trend}

            Technical Indicators:
            {technical_indicators}

            Latest Crypto News Headlines:
            {news_titles}

            Market Volatility: {market_volatility:.2f}%

            Past Trading Performance:
            {trade_history_summary}

            Trading Statistics:
            {performance_summary}

            {time_summary}

            {volatility_summary}

            Consider short-term (15m) trends for immediate signals, while validating with medium-term (1h) trend for overall direction.
            Based on this data, trend lines, latest breaking news, and past trading performance, provide your trading decision as direction:probability (e.g., "long:0.78" or "short:0.63")."""
                    }
                ]
            }

            response = requests.post(local_ai_url, headers=headers, json=payload)

            if response.status_code == 200:

                # Replace lines around 1460 in your autotrade-v1.0.3.py file
                # This is the problematic code:
                # ai_response = response.choices[0].message.content.lower().strip()

                # Replace with this more robust solution:
                try:
                    # First, check if response is a requests Response object
                    if hasattr(response, 'json'):
                        try:
                            # Try to parse as JSON
                            response_data = response.json()
                            
                            # Print the response structure for debugging
                            print(f"API Response structure: {list(response_data.keys())}")
                            
                            # Try to extract content from various possible response formats
                            if 'choices' in response_data and len(response_data['choices']) > 0:
                                if isinstance(response_data['choices'][0], dict):
                                    if 'message' in response_data['choices'][0] and 'content' in response_data['choices'][0]['message']:
                                        ai_response = response_data['choices'][0]['message']['content']
                                    elif 'text' in response_data['choices'][0]:
                                        ai_response = response_data['choices'][0]['text']
                                    else:
                                        ai_response = str(response_data['choices'][0])
                                else:
                                    ai_response = str(response_data['choices'][0])
                            elif 'response' in response_data:
                                ai_response = response_data['response']
                            elif 'output' in response_data:
                                ai_response = response_data['output']
                            elif 'generated_text' in response_data:
                                ai_response = response_data['generated_text']
                            elif 'result' in response_data:
                                ai_response = response_data['result']
                            elif 'content' in response_data:
                                ai_response = response_data['content']
                            else:
                                # Just convert the whole response to string as a fallback
                                ai_response = str(response_data)
                                
                        except ValueError:
                            # If JSON parsing fails, use text directly
                            ai_response = response.text
                    else:
                        # If not a requests Response, treat as direct text
                        ai_response = str(response)
                        
                    # Clean up the response
                    ai_response = ai_response.lower().strip()
                    print(f"AI Analysis: {ai_response}")
                    
                except Exception as e:
                    print(f"Error parsing API response: {e}")
                    print(f"Raw response type: {type(response)}")
                    print(f"Raw response: {str(response)[:500]}...")
                    ai_response = ""  # Set default empty response
                    time.sleep(30)
                    continue

            #ai_response = response.choices[0].message.content.lower().strip()

            try:
                response_data = response.json()
                ai_response = response_data['choices'][0]['message']['content'].lower().strip()
            except (ValueError, KeyError) as e:
                print(f"Error parsing response: {e}")
                print(f"Response content: {response.text[:500]}...")
                ai_response = ""  # Set a default empty response
                time.sleep(30)
                continue

            print(f"AI Analysis: {ai_response}")
            
            # AI 응답 파싱
            action, win_probability = parse_ai_response(ai_response)
            
            if action == 'none' or win_probability < 0.5:
                print("No clear trading signal or confidence too low. Skipping trade.")
                time.sleep(30)
                continue
                
            print(f"Trading Signal: {action.upper()} with {win_probability:.2%} confidence")
            
            # 승리/손실 비율 계산 (Risk-Reward Ratio)
            # SL/TP 비율을 사용하여 계산: TP/SL
            win_loss_ratio = 1.0  # 기본값 (symmetric 1:1)
            
            # Kelly Criterion 계산
            kelly_fraction = calculate_kelly_criterion(win_probability, win_loss_ratio)
            print(f"Kelly Criterion Fraction: {kelly_fraction:.4f} ({kelly_fraction:.2%})")
            
            if kelly_fraction <= 0:
                print("Kelly criterion suggests not to trade. Skipping this opportunity.")
                time.sleep(30)
                continue
                
            # 적용할 자본 계산 (계정 잔액 * 켈리 비율)
            kelly_capital = account_balance * kelly_fraction
            
            # 최대 레버리지 적용
            leverage_level = min(MAX_LEVERAGE, math.ceil(5 * win_probability))  # 승률에 따라 레버리지 조정 (최대 MAX_LEVERAGE)
            leveraged_capital = kelly_capital * leverage_level
            
            # 레버리지 설정
            exchange.set_leverage(leverage_level, symbol)
            print(f"Setting leverage to {leverage_level}x")
            
            # 기본 주문 크기와 켈리 크리테리온 비교하여 최종 주문 크기 결정
            # 최소 BASE_ORDER_SIZE, 최대 leveraged_capital
            final_order_size = max(BASE_ORDER_SIZE, min(leveraged_capital, account_balance * MAX_ACCOUNT_RISK * leverage_level))
            
            # 주문 수량 계산 (USDT -> BTC)
            amount = math.ceil((final_order_size / current_price) * 1000) / 1000
            print(f"Order Amount: {amount} BTC (${final_order_size:.2f} USDT)")

            # 포지션 진입 및 SL/TP 주문
            if action == "long":
                order = exchange.create_market_buy_order(symbol, amount)
                entry_price = current_price
                sl_price = round(entry_price * (1 - sl_tp_percentage/100), 2)
                tp_price = round(entry_price * (1 + sl_tp_percentage/100), 2)
                
                # SL/TP 주문 생성
                exchange.create_order(symbol, 'STOP_MARKET', 'sell', amount, None, {'stopPrice': sl_price})
                exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', 'sell', amount, None, {'stopPrice': tp_price})
                
                # 트레이딩 내역 데이터베이스에 저장
                active_trade_id = save_trade(
                    action="long",
                    entry_price=entry_price,
                    amount=amount,
                    order_size=final_order_size,
                    leverage=leverage_level,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    kelly_fraction=kelly_fraction,
                    win_probability=win_probability,
                    volatility=market_volatility
                )
                
                print(f"\n=== LONG Position Opened ===")
                print(f"Entry: ${entry_price:,.2f}")
                print(f"Stop Loss: ${sl_price:,.2f} (-{sl_tp_percentage}%)")
                print(f"Take Profit: ${tp_price:,.2f} (+{sl_tp_percentage}%)")
                print(f"Position Size: {amount} BTC (${final_order_size:.2f} USDT)")
                print(f"Leverage: {leverage_level}x")
                print(f"Kelly Fraction: {kelly_fraction:.2%}")
                print(f"Win Probability: {win_probability:.2%}")
                print(f"Trade ID: {active_trade_id}")
                print("===========================")

            elif action == "short":
                order = exchange.create_market_sell_order(symbol, amount)
                entry_price = current_price
                sl_price = round(entry_price * (1 + sl_tp_percentage/100), 2)
                tp_price = round(entry_price * (1 - sl_tp_percentage/100), 2)
                
                # SL/TP 주문 생성
                exchange.create_order(symbol, 'STOP_MARKET', 'buy', amount, None, {'stopPrice': sl_price})
                exchange.create_order(symbol, 'TAKE_PROFIT_MARKET', 'buy', amount, None, {'stopPrice': tp_price})
                
                # 트레이딩 내역 데이터베이스에 저장
                active_trade_id = save_trade(
                    action="short",
                    entry_price=entry_price,
                    amount=amount,
                    order_size=final_order_size,
                    leverage=leverage_level,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    kelly_fraction=kelly_fraction,
                    win_probability=win_probability,
                    volatility=market_volatility
                )
                
                print(f"\n=== SHORT Position Opened ===")
                print(f"Entry: ${entry_price:,.2f}")
                print(f"Stop Loss: ${sl_price:,.2f} (+{sl_tp_percentage}%)")
                print(f"Take Profit: ${tp_price:,.2f} (-{sl_tp_percentage}%)")
                print(f"Position Size: {amount} BTC (${final_order_size:.2f} USDT)")
                print(f"Leverage: {leverage_level}x")
                print(f"Kelly Fraction: {kelly_fraction:.2%}")
                print(f"Win Probability: {win_probability:.2%}")
                print(f"Trade ID: {active_trade_id}")
                print("============================")
            else:
                print("action이 'long' 또는 'short'가 아니므로 주문을 실행하지 않습니다.")
        
        time.sleep(30)  # 30초마다 체크

    except Exception as e:
        print(f"\n Error: {e}")
        time.sleep(5)
