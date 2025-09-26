import ccxt
import json
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
        status           VARCHAR(10) DEFAULT 'open',
        entry_plan       TEXT,
        filled_amount    DOUBLE DEFAULT 0,
        filled_notional  DOUBLE DEFAULT 0,
        avg_entry_price  DOUBLE,
        risk_percent     DOUBLE,
        reward_percent   DOUBLE,
        sl_order_id      VARCHAR(100),
        tp_order_id      VARCHAR(100)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """)

    # Ensure new columns exist for existing deployments
    column_definitions = {
        'entry_plan': "TEXT",
        'filled_amount': "DOUBLE DEFAULT 0",
        'filled_notional': "DOUBLE DEFAULT 0",
        'avg_entry_price': "DOUBLE",
        'risk_percent': "DOUBLE",
        'reward_percent': "DOUBLE",
        'sl_order_id': "VARCHAR(100)",
        'tp_order_id': "VARCHAR(100)"
    }

    for column_name, definition in column_definitions.items():
        cursor.execute(
            """
            SELECT COUNT(*)
              FROM INFORMATION_SCHEMA.COLUMNS
             WHERE TABLE_SCHEMA = %s
               AND TABLE_NAME = 'trades'
               AND COLUMN_NAME = %s
            """,
            (DB_CONFIG['database'], column_name)
        )
        if cursor.fetchone()[0] == 0:
            cursor.execute(f"ALTER TABLE trades ADD COLUMN {column_name} {definition}")

    # trade_orders table to track ladder execution
    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS trade_orders (
        id             INT AUTO_INCREMENT PRIMARY KEY,
        trade_id       INT NOT NULL,
        order_id       VARCHAR(100) NOT NULL,
        symbol         VARCHAR(20),
        side           VARCHAR(10),
        price          DOUBLE,
        amount         DOUBLE,
        status         VARCHAR(20),
        filled_amount  DOUBLE,
        remaining      DOUBLE,
        avg_price      DOUBLE,
        order_type     VARCHAR(20),
        created_at     DATETIME,
        updated_at     DATETIME,
        UNIQUE KEY uniq_trade_order (trade_id, order_id),
        FOREIGN KEY (trade_id) REFERENCES trades(id)
            ON DELETE CASCADE
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
               sl_price, tp_price, kelly_fraction, win_probability,
               volatility, risk_percent, reward_percent,
               entry_plan=None, ladder_orders=None):
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
    risk_percent    = float(risk_percent)
    reward_percent  = float(reward_percent)

    entry_plan_json = json.dumps(entry_plan) if entry_plan else None

    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute("""
        INSERT INTO trades
          (timestamp, action, entry_price, amount, order_size, leverage,
           stop_loss, take_profit, kelly_fraction, win_probability, volatility,
           status, entry_plan, filled_amount, filled_notional, avg_entry_price,
           risk_percent, reward_percent, sl_order_id, tp_order_id)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        now, action, entry_price, amount, order_size, leverage,
        sl_price, tp_price, kelly_fraction, win_probability, volatility,
        'open', entry_plan_json, 0.0, 0.0, None,
        risk_percent, reward_percent, None, None
    ))
    trade_id = cursor.lastrowid
    conn.commit()
    conn.close()

    if ladder_orders:
        record_trade_orders(trade_id, ladder_orders)

    print(f"Trade saved with ID: {trade_id}")
    return trade_id


def record_trade_orders(trade_id, orders):
    """Persist ladder child orders for a trade."""
    if not orders:
        return

    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    for order in orders:
        order_id = order.get('id')
        if not order_id:
            continue

        order_price = order.get('price') or order.get('info', {}).get('price')
        order_price = float(order_price) if order_price is not None else None
        avg_price = order.get('average') if order.get('average') is not None else None

        status = order.get('status') or 'open'
        order_type = order.get('type')

        cursor.execute("""
            INSERT INTO trade_orders
                (trade_id, order_id, symbol, side, price, amount, status,
                 filled_amount, remaining, avg_price, order_type, created_at, updated_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON DUPLICATE KEY UPDATE
                price=VALUES(price),
                amount=VALUES(amount),
                status=VALUES(status),
                filled_amount=VALUES(filled_amount),
                remaining=VALUES(remaining),
                avg_price=VALUES(avg_price),
                order_type=VALUES(order_type),
                updated_at=VALUES(updated_at)
        """,
        (
            trade_id,
            str(order_id),
            order.get('symbol'),
            order.get('side'),
            order_price,
            float(order.get('amount') or 0.0),
            status,
            float(order.get('filled') or 0.0),
            float(order.get('remaining') or 0.0),
            avg_price,
            order_type,
            now,
            now
        ))

    conn.commit()
    conn.close()


def get_trade_orders(trade_id):
    """Fetch stored ladder orders for a trade."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, order_id, symbol, side, price, amount, status,
               filled_amount, remaining, avg_price, order_type
          FROM trade_orders
         WHERE trade_id = %s
    """, (trade_id,))
    rows = cursor.fetchall()
    conn.close()

    orders = []
    for row in rows:
        (row_id, order_id, symbol_value, side, price, amount, status,
         filled_amount, remaining, avg_price, order_type) = row
        orders.append({
            'db_id': row_id,
            'order_id': order_id,
            'symbol': symbol_value,
            'side': side,
            'price': float(price) if price is not None else None,
            'amount': float(amount) if amount is not None else 0.0,
            'status': status,
            'filled_amount': float(filled_amount) if filled_amount is not None else 0.0,
            'remaining': float(remaining) if remaining is not None else 0.0,
            'avg_price': float(avg_price) if avg_price is not None else None,
            'order_type': order_type
        })
    return orders


def refresh_trade_fill(trade_id):
    """Recompute aggregate fills for a trade and persist them."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            COALESCE(SUM(filled_amount), 0),
            COALESCE(SUM(filled_amount * COALESCE(avg_price, price)), 0)
          FROM trade_orders
         WHERE trade_id = %s
    """, (trade_id,))
    total_filled, notional = cursor.fetchone()

    avg_entry = None
    if total_filled and total_filled > 0:
        avg_entry = notional / total_filled if notional else 0

    cursor.execute("""
        UPDATE trades
           SET filled_amount = %s,
               filled_notional = %s,
               avg_entry_price = %s
         WHERE id = %s
    """, (float(total_filled or 0.0), float(notional or 0.0), avg_entry, trade_id))
    conn.commit()
    conn.close()

    return float(total_filled or 0.0), (float(avg_entry) if avg_entry is not None else None)


def sync_trade_orders_with_exchange(trade_id, symbol_code):
    """Poll exchange for ladder order updates."""
    orders = get_trade_orders(trade_id)
    if not orders:
        return 0.0, None

    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    any_updates = False

    for order in orders:
        if order['status'] in ('closed', 'canceled'):
            continue
        try:
            remote = exchange.fetch_order(order['order_id'], symbol_code)
        except Exception as e:
            print(f"Failed to sync order {order['order_id']}: {e}")
            continue

        any_updates = True
        remote_price = remote.get('price')
        remote_price = float(remote_price) if remote_price is not None else None
        remote_avg = remote.get('average')
        remote_avg = float(remote_avg) if remote_avg is not None else None
        remote_amount = float(remote.get('amount') or 0.0)
        remote_filled = float(remote.get('filled') or 0.0)
        remote_remaining = float(remote.get('remaining') or 0.0)
        cursor.execute("""
            UPDATE trade_orders
               SET status=%s,
                   filled_amount=%s,
                   remaining=%s,
                   avg_price=%s,
                   price=%s,
                   amount=%s,
                   updated_at=%s
             WHERE id=%s
        """, (
            remote.get('status'),
            remote_filled,
            remote_remaining,
            remote_avg,
            remote_price,
            remote_amount,
            now,
            order['db_id']
        ))

    if any_updates:
        conn.commit()
    conn.close()

    return refresh_trade_fill(trade_id)


def cancel_trade_orders(trade_id, symbol_code):
    """Cancel any outstanding child orders on the exchange."""
    orders = get_trade_orders(trade_id)
    for order in orders:
        if order['status'] in ('closed', 'canceled'):
            continue
        try:
            exchange.cancel_order(order['order_id'], symbol_code)
        except Exception as e:
            print(f"Failed to cancel order {order['order_id']}: {e}")


def cancel_protective_orders(trade):
    """Cancel associated stop-loss and take-profit orders."""
    for oid in [trade.get('sl_order_id'), trade.get('tp_order_id')]:
        if not oid:
            continue
        try:
            exchange.cancel_order(oid, symbol)
        except Exception as e:
            print(f"Failed to cancel protective order {oid}: {e}")


def _extract_stop_price(order_info):
    if not order_info:
        return None
    info = order_info.get('info', {}) if isinstance(order_info, dict) else {}
    for key in ('stopPrice', 'triggerPrice', 'price'):
        value = None
        if isinstance(order_info, dict):
            value = order_info.get(key)
        if value is None and isinstance(info, dict):
            value = info.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def ensure_protective_orders(trade):
    """Ensure SL/TP orders reflect the latest fills and averages."""
    filled_amount = float(trade.get('filled_amount') or 0.0)
    avg_entry = trade.get('avg_entry_price')
    if filled_amount <= 0 or avg_entry in (None, 0):
        return trade

    risk_percent = trade.get('risk_percent') or 0.0
    reward_percent = trade.get('reward_percent') or 0.0
    action = trade.get('action')

    if not risk_percent:
        stop_loss = trade.get('stop_loss')
        if stop_loss:
            if action == 'long':
                risk_percent = abs((avg_entry - stop_loss) / avg_entry) * 100
            else:
                risk_percent = abs((stop_loss - avg_entry) / avg_entry) * 100
    if not reward_percent:
        take_profit = trade.get('take_profit')
        if take_profit:
            if action == 'long':
                reward_percent = abs((take_profit - avg_entry) / avg_entry) * 100
            else:
                reward_percent = abs((avg_entry - take_profit) / avg_entry) * 100

    if action == 'long':
        sl_price = round(avg_entry * (1 - (risk_percent / 100.0)), 2) if risk_percent else trade.get('stop_loss')
        tp_price = round(avg_entry * (1 + (reward_percent / 100.0)), 2) if reward_percent else trade.get('take_profit')
        exit_side = 'sell'
    else:
        sl_price = round(avg_entry * (1 + (risk_percent / 100.0)), 2) if risk_percent else trade.get('stop_loss')
        tp_price = round(avg_entry * (1 - (reward_percent / 100.0)), 2) if reward_percent else trade.get('take_profit')
        exit_side = 'buy'

    sl_price = float(sl_price) if sl_price is not None else None
    tp_price = float(tp_price) if tp_price is not None else None

    sl_order_id = trade.get('sl_order_id')
    tp_order_id = trade.get('tp_order_id')

    def manage_order(existing_id, target_price, order_type):
        if target_price is None:
            if existing_id:
                try:
                    exchange.cancel_order(existing_id, symbol)
                except Exception as exc:
                    print(f"Failed to cancel {order_type} order {existing_id}: {exc}")
            return None

        existing_order = None
        if existing_id:
            try:
                existing_order = exchange.fetch_order(existing_id, symbol)
            except Exception as exc:
                print(f"Failed to fetch existing {order_type} order {existing_id}: {exc}")
                existing_order = None

        needs_update = True
        if existing_order:
            current_price = _extract_stop_price(existing_order)
            current_amount = float(existing_order.get('amount') or 0.0)
            if (current_price is not None and abs(current_price - target_price) < 1e-6
                    and abs(current_amount - filled_amount) < 1e-6
                    and existing_order.get('status') not in ('canceled', 'closed')):
                needs_update = False

        if not needs_update:
            return existing_id

        if existing_id:
            try:
                exchange.cancel_order(existing_id, symbol)
            except Exception as exc:
                print(f"Failed to cancel {order_type} order {existing_id}: {exc}")

        params = {'stopPrice': target_price, 'reduceOnly': True}
        try:
            new_order = exchange.create_order(symbol, order_type, exit_side, filled_amount, None, params)
            return new_order.get('id')
        except Exception as exc:
            print(f"Failed to place {order_type} order: {exc}")
            return existing_id

    new_sl_id = manage_order(sl_order_id, sl_price, 'STOP_MARKET')
    new_tp_id = manage_order(tp_order_id, tp_price, 'TAKE_PROFIT_MARKET')

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE trades
           SET stop_loss=%s,
               take_profit=%s,
               risk_percent=%s,
               reward_percent=%s,
               sl_order_id=%s,
               tp_order_id=%s
         WHERE id=%s
    """, (
        sl_price,
        tp_price,
        float(risk_percent or 0.0),
        float(reward_percent or 0.0),
        new_sl_id,
        new_tp_id,
        trade['id']
    ))
    conn.commit()
    conn.close()

    trade['stop_loss'] = sl_price
    trade['take_profit'] = tp_price
    trade['risk_percent'] = risk_percent
    trade['reward_percent'] = reward_percent
    trade['sl_order_id'] = new_sl_id
    trade['tp_order_id'] = new_tp_id
    return trade


def close_trade(trade_id, close_price, result):
    close_price = float(close_price)

    # Final sync before closing
    try:
        sync_trade_orders_with_exchange(trade_id, symbol)
    except Exception as e:
        print(f"Failed to sync orders before closing trade {trade_id}: {e}")

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT timestamp, action, avg_entry_price, filled_amount,
               risk_percent, reward_percent, sl_order_id, tp_order_id
          FROM trades WHERE id = %s
    """, (trade_id,))
    row = cursor.fetchone()
    if not row:
        print(f"Trade ID {trade_id} not found.")
        conn.close()
        return

    open_ts, action, avg_entry_price, filled_amount, risk_pct, reward_pct, sl_order_id, tp_order_id = row
    if isinstance(open_ts, datetime):
        open_time = open_ts
    else:
        open_time = datetime.strptime(open_ts, '%Y-%m-%d %H:%M:%S')

    trade_snapshot = {
        'id': trade_id,
        'action': action,
        'sl_order_id': sl_order_id,
        'tp_order_id': tp_order_id
    }
    cancel_protective_orders(trade_snapshot)
    cancel_trade_orders(trade_id, symbol)

    amount = float(filled_amount or 0.0)
    entry_price = float(avg_entry_price or 0.0)
    close_ts_dt = datetime.now()
    close_ts = close_ts_dt.strftime('%Y-%m-%d %H:%M:%S')
    duration = str(close_ts_dt - open_time)

    if amount <= 0 or entry_price <= 0:
        pnl = 0.0
        pnl_pct = 0.0
    elif action == 'long':
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

    cursor.execute("""
        UPDATE trades
           SET status='closed',
               sl_order_id=NULL,
               tp_order_id=NULL
         WHERE id=%s
    """, (trade_id,))
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
               stop_loss, take_profit, timestamp, entry_plan,
               filled_amount, avg_entry_price, filled_notional,
               risk_percent, reward_percent, sl_order_id, tp_order_id
          FROM trades
         WHERE status = 'open'
      ORDER BY id DESC
         LIMIT 1
    """)
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    (trade_id, action, entry_price, amount, order_size, leverage,
     sl, tp, ts, plan_raw, filled_amount, avg_entry_price, filled_notional,
     risk_percent, reward_percent, sl_order_id, tp_order_id) = row
    if isinstance(ts, datetime):
        open_time = ts
    else:
        open_time = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
    try:
        entry_plan = json.loads(plan_raw) if plan_raw else None
    except (TypeError, json.JSONDecodeError):
        entry_plan = None
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
        'duration': datetime.now() - open_time,
        'entry_plan': entry_plan,
        'filled_amount': filled_amount,
        'avg_entry_price': avg_entry_price,
        'filled_notional': filled_notional,
        'risk_percent': risk_percent,
        'reward_percent': reward_percent,
        'sl_order_id': sl_order_id,
        'tp_order_id': tp_order_id
    }

# 새로 추가한 함수: 최근 거래 내역 및 결과 가져오기
def get_recent_trade_history(limit=20):
    """최근 완료된 거래 내역 및 결과를 가져옵니다"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT t.id, t.timestamp, t.action, t.entry_price,
               tr.close_price, tr.pnl, tr.pnl_percentage, tr.result,
               t.win_probability, t.kelly_fraction, t.volatility, t.leverage,
               t.entry_plan
          FROM trades t
          JOIN trade_results tr ON t.id = tr.trade_id
         WHERE t.status = 'closed'
      ORDER BY tr.close_timestamp DESC
         LIMIT %s
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()

    history = []
    for (tid, ts, act, ep, cp, pnl, pnl_pct, res, wp, kf, vol, lev, plan_raw) in rows:
        try:
            entry_plan = json.loads(plan_raw) if plan_raw else None
        except (TypeError, json.JSONDecodeError):
            entry_plan = None
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
            'leverage':        lev,
            'entry_plan':      entry_plan
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
    """켈리 크리테리온을 계산한다.

    승률이 0.5 미만이면 음의 기대값으로 간주하여 거래를 건너뛴다. 승률과
    손익비는 그대로 사용하며, 켈리 수식으로 계산한 뒤 절반만 사용하고
    계정 최대 리스크 한도를 적용한다.
    """

    if win_probability <= 0.5 or win_loss_ratio <= 0:
        return 0

    kelly_fraction = (win_probability * win_loss_ratio - (1 - win_probability)) / win_loss_ratio

    if kelly_fraction <= 0:
        return 0

    half_kelly = kelly_fraction * 0.5

    return min(half_kelly, MAX_ACCOUNT_RISK)

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


def plan_ladder_entries(base_price, direction, volatility, steps=7):
    """Generate a ladder of entry orders scaled by market volatility."""
    if steps <= 0:
        return []

    direction = direction.lower()
    if direction not in {"long", "short"}:
        raise ValueError(f"Unsupported direction for ladder planning: {direction}")

    # Scale spacing using volatility so quiet markets keep entries tight while volatile markets space out more
    raw_scale = volatility / 2.0 if np.isfinite(volatility) else 1.0
    vol_scale = float(np.clip(raw_scale, 0.5, 2.0))
    base_offsets = np.linspace(0.5, 3.5, steps) * vol_scale

    # Weight earlier orders slightly less to reserve size for deeper pullbacks
    fraction_weights = np.linspace(1.0, 2.0, steps)
    fraction_values = (fraction_weights / fraction_weights.sum()).tolist()

    plan = []
    allocated = 0.0
    for idx, (offset, fraction) in enumerate(zip(base_offsets.tolist(), fraction_values)):
        if idx == steps - 1:
            # Ensure the final leg captures any rounding remainder
            fraction = max(0.0, 1.0 - allocated)
        allocated += fraction
        if fraction <= 0:
            continue

        pct_offset = -offset if direction == "long" else offset
        target_price = round(base_price * (1 + pct_offset / 100), 2)
        plan.append({
            'offset_pct': round(float(pct_offset), 4),
            'target_price': target_price,
            'fraction': round(float(fraction), 4)
        })

    return plan


def execute_entry_plan(symbol, direction, total_notional, entry_plan):
    """Place ladder orders based on the generated entry plan."""
    if not entry_plan or total_notional <= 0:
        return [], 0.0, 0.0

    side = 'buy' if direction == 'long' else 'sell'
    placed_orders = []
    committed_notional = 0.0
    cumulative_amount = 0.0

    for idx, leg in enumerate(entry_plan, start=1):
        target_price = leg['target_price']
        leg_fraction = leg['fraction']
        allocated_notional = round(total_notional * leg_fraction, 2)
        remaining_notional = round(max(total_notional - committed_notional, 0.0), 2)
        if idx == len(entry_plan):
            leg_notional = remaining_notional
        else:
            leg_notional = min(allocated_notional, remaining_notional)

        if leg_notional <= 0 or target_price <= 0:
            continue

        leg_amount = round(leg_notional / target_price, 3)
        if leg_amount <= 0:
            continue

        try:
            if direction == 'long':
                order = exchange.create_limit_buy_order(symbol, leg_amount, target_price)
            else:
                order = exchange.create_limit_sell_order(symbol, leg_amount, target_price)
            placed_orders.append(order)
            committed_notional += leg_notional
            cumulative_amount += leg_amount
            print(
                f"Placed ladder order {idx}/{len(entry_plan)}: {side.upper()} {leg_amount} BTC @ ${target_price:.2f} "
                f"(offset {leg['offset_pct']:.2f}%)"
            )
        except Exception as e:
            print(f"Failed to place ladder order {idx}: {e}")

        if committed_notional >= total_notional:
            break

    return placed_orders, cumulative_amount, committed_notional

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


def generate_rule_based_signal(short_df, medium_df):
    """기술적 지표를 활용한 규칙 기반 진입 시그널을 생성한다."""

    latest_short = short_df.iloc[-1]
    latest_medium = medium_df.iloc[-1]

    signal_components = []

    signal_components.append(1 if latest_short['close'] > latest_short['MA20'] else -1)
    signal_components.append(1 if latest_short['MA5'] > latest_short['MA20'] else -1)
    signal_components.append(1 if latest_medium['close'] > latest_medium['MA50'] else -1)
    signal_components.append(1 if latest_short['MACD'] > latest_short['MACD_signal'] else -1)

    rsi = latest_short['RSI']
    if rsi >= 60:
        signal_components.append(1)
    elif rsi <= 40:
        signal_components.append(-1)

    score = sum(signal_components)
    positives = sum(1 for s in signal_components if s > 0)
    negatives = sum(1 for s in signal_components if s < 0)
    total_votes = positives + negatives

    if total_votes == 0:
        return 'none', 0.0

    if score >= 2:
        action = 'long'
    elif score <= -2:
        action = 'short'
    else:
        action = 'none'

    if action == 'none':
        return action, 0.0

    conviction = max(positives, negatives) / total_votes
    probability = 0.55 + min(0.35, conviction * 0.3)

    return action, min(probability, 0.9)

def parse_ai_response(response_text):
    """지정된 포맷의 AI 응답에서 방향과 확률을 추출한다."""

    if not response_text:
        return 'none', 0.0

    try:
        import re

        cleaned_text = response_text.strip().lower()
        print(f"Raw AI response to parse: {cleaned_text}")

        direction_match = re.search(r"\*\*direction:\s*(long|short)\*\*", cleaned_text)
        probability_match = re.search(r"\*\*probability:\s*((?:0?\.\d+)|(?:1\.0+))\*\*", cleaned_text)

        if not direction_match or not probability_match:
            inline_match = re.search(r"(long|short)\s*[:|-]\s*((?:0?\.\d+)|(?:1\.0+))", cleaned_text)
            if inline_match:
                direction_match, probability_match = inline_match, inline_match
                action = inline_match.group(1)
                probability = float(inline_match.group(2))
            else:
                print("AI response missing required direction/probability format. Skipping.")
                return 'none', 0.0
        else:
            action = direction_match.group(1)
            probability = float(probability_match.group(1))

        if probability < 0.5 or probability > 1:
            print(f"Probability {probability} outside allowed range. Skipping signal.")
            return 'none', 0.0

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

    active_trade = get_active_trade_info()
    if not active_trade:
        return False

    trade_id = active_trade['id']

    try:
        filled_amount, avg_entry = sync_trade_orders_with_exchange(trade_id, symbol)
    except Exception as e:
        print(f"Failed to sync ladder orders for trade {trade_id}: {e}")
        filled_amount = active_trade.get('filled_amount') or 0.0
        avg_entry = active_trade.get('avg_entry_price')

    refreshed_trade = get_active_trade_info()
    if refreshed_trade:
        active_trade = refreshed_trade

    filled_amount = float(active_trade.get('filled_amount') or filled_amount or 0.0)
    avg_entry = active_trade.get('avg_entry_price') or avg_entry

    if filled_amount <= 0:
        # 아직 체결이 없으면 보호 주문을 걸지 않고 대기
        return False

    active_trade['filled_amount'] = filled_amount
    active_trade['avg_entry_price'] = avg_entry
    active_trade = ensure_protective_orders(active_trade)

    ticker = exchange.fetch_ticker(symbol)
    current_price = ticker['last']

    sl_price = active_trade.get('stop_loss')
    tp_price = active_trade.get('take_profit')

    if sl_price is None or tp_price is None:
        return False

    if active_trade['action'] == 'long':
        if current_price <= sl_price:
            print(f"Closing trade #{trade_id} at {current_price} due to stop_loss")
            close_trade(trade_id, current_price, "stop_loss")
            return True
        if current_price >= tp_price:
            print(f"Closing trade #{trade_id} at {current_price} due to take_profit")
            close_trade(trade_id, current_price, "take_profit")
            return True
    else:
        if current_price >= sl_price:
            print(f"Closing trade #{trade_id} at {current_price} due to stop_loss")
            close_trade(trade_id, current_price, "stop_loss")
            return True
        if current_price <= tp_price:
            print(f"Closing trade #{trade_id} at {current_price} due to take_profit")
            close_trade(trade_id, current_price, "take_profit")
            return True

    return False



# 활성 트레이드 ID 추적 변수
active_trade_id = None
last_processed_candle = None

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
        risk_percent = 1.0
        if market_volatility >= 3.0:
            risk_percent = 2.5
            print(f"Very high volatility detected! Adjusting SL to {risk_percent:.1f}%")
        elif market_volatility >= 2.0:
            risk_percent = 1.5
            print(f"High volatility detected! Adjusting SL to {risk_percent:.1f}%")

        reward_percent = risk_percent * 1.5

        if current_side:
            print(f"Current Position: {current_side.upper()} {amount} BTC")
            
            # 활성 트레이드 정보 데이터베이스에서 조회
            active_trade = get_active_trade_info()
            if active_trade:
                print(f"Trade ID: {active_trade['id']}, Entry: ${active_trade['entry_price']:.2f}, Duration: {active_trade['duration']}")
                
                # 현재 PnL 계산
                basis_price = active_trade.get('avg_entry_price') or active_trade['entry_price']
                basis_amount = active_trade.get('filled_amount') or active_trade['amount']
                if active_trade['action'] == 'long':
                    current_pnl = (current_price - basis_price) * basis_amount
                    pnl_percentage = ((current_price / basis_price) - 1) * 100 if basis_price else 0
                else:  # short
                    current_pnl = (basis_price - current_price) * basis_amount
                    pnl_percentage = ((basis_price / current_price) - 1) * 100 if current_price else 0
                
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
            latest_candle_time = df_short_full['timestamp'].iloc[-1]

            if last_processed_candle == latest_candle_time:
                print("Waiting for current 15m candle to close before re-evaluating.")
                time.sleep(30)
                continue

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

            bias_score = 0
            long_trend = long_term_ma_analysis.get('long_trend', 'neutral')
            if long_trend == 'bullish':
                bias_score += 1
            elif long_trend == 'bearish':
                bias_score -= 1

            change_threshold = 0.5
            if long_change >= change_threshold:
                bias_score += 1
            elif long_change <= -change_threshold:
                bias_score -= 1

            if bias_score >= 1:
                long_bias = 'bullish'
            elif bias_score <= -1:
                long_bias = 'bearish'
            else:
                long_bias = 'neutral'

            print(f"Long-term directional bias: {long_bias.upper()} (score={bias_score}, 4h change={long_change:.2f}%)")

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
            

            rule_action, rule_probability = generate_rule_based_signal(df_short_full, df_medium_full)

            if long_bias == 'bullish' and rule_action == 'short':
                print("Long-term bias is bullish. Skipping conflicting short rule signal.")
                last_processed_candle = latest_candle_time
                time.sleep(30)
                continue
            if long_bias == 'bearish' and rule_action == 'long':
                print("Long-term bias is bearish. Skipping conflicting long rule signal.")
                last_processed_candle = latest_candle_time
                time.sleep(30)
                continue

            if rule_action == 'none':
                print("Rule-based strategy found no actionable edge. Waiting for the next candle.")
                last_processed_candle = latest_candle_time
                time.sleep(30)
                continue

            print(f"Rule-based signal: {rule_action.upper()} with {rule_probability:.2%} confidence")

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
            
            ai_action, ai_probability = parse_ai_response(ai_response)

            if long_bias == 'bullish' and ai_action == 'short':
                print("AI suggested a short, but the long-term bias is bullish. Ignoring AI override.")
                ai_action = 'none'
            elif long_bias == 'bearish' and ai_action == 'long':
                print("AI suggested a long, but the long-term bias is bearish. Ignoring AI override.")
                ai_action = 'none'

            final_action = rule_action
            final_probability = rule_probability

            if ai_action == 'none':
                print("AI confirmation unavailable. Proceeding with rule-based signal only.")
            else:
                if ai_action != rule_action:
                    print("AI disagrees with the rule-based signal. Skipping this setup.")
                    last_processed_candle = latest_candle_time
                    time.sleep(30)
                    continue
                final_probability = min(1.0, (rule_probability + ai_probability) / 2)

            win_probability = final_probability

            if (long_bias == 'bullish' and final_action == 'short') or (long_bias == 'bearish' and final_action == 'long'):
                print("Combined signal conflicts with long-term bias. Skipping trade.")
                last_processed_candle = latest_candle_time
                time.sleep(30)
                continue

            if win_probability < 0.5 or final_action == 'none':
                print("Combined signal lacks edge. Skipping trade.")
                last_processed_candle = latest_candle_time
                time.sleep(30)
                continue

            print(f"Trading Signal: {final_action.upper()} with {win_probability:.2%} confidence")

            entry_plan = plan_ladder_entries(current_price, final_action, market_volatility)
            if not entry_plan:
                print("No ladder entry plan generated. Skipping trade execution.")
                last_processed_candle = latest_candle_time
                time.sleep(30)
                continue

            print("Entry ladder plan (offset %, target price, fraction):")
            for leg in entry_plan:
                print(
                    f"  {leg['offset_pct']:+.2f}% -> ${leg['target_price']:.2f} ({leg['fraction']:.2%} of position)"
                )

            # 승리/손실 비율 계산 (Risk-Reward Ratio)
            win_loss_ratio = reward_percent / risk_percent if risk_percent else 0

            # Kelly Criterion 계산
            kelly_fraction = calculate_kelly_criterion(win_probability, win_loss_ratio)
            print(f"Kelly Criterion Fraction: {kelly_fraction:.4f} ({kelly_fraction:.2%})")
            
            if kelly_fraction <= 0:
                print("Kelly criterion suggests not to trade. Skipping this opportunity.")
                last_processed_candle = latest_candle_time
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
            
            print(f"Order Notional Target: ${final_order_size:.2f} USDT")

            # 포지션 진입 준비
            last_processed_candle = latest_candle_time

            if final_action == "long":
                ladder_orders, ladder_amount, committed_notional = execute_entry_plan(
                    symbol, "long", final_order_size, entry_plan
                )
                if not ladder_orders:
                    print("Unable to place ladder orders for LONG setup. Skipping trade.")
                    time.sleep(30)
                    continue

                entry_price = current_price
                sl_price = round(entry_price * (1 - risk_percent/100), 2)
                tp_price = round(entry_price * (1 + reward_percent/100), 2)

                # 트레이딩 내역 데이터베이스에 저장
                active_trade_id = save_trade(
                    action="long",
                    entry_price=entry_price,
                    amount=ladder_amount,
                    order_size=committed_notional,
                    leverage=leverage_level,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    kelly_fraction=kelly_fraction,
                    win_probability=win_probability,
                    volatility=market_volatility,
                    risk_percent=risk_percent,
                    reward_percent=reward_percent,
                    entry_plan=entry_plan,
                    ladder_orders=ladder_orders
                )

                print(f"\n=== LONG Ladder Prepared ===")
                print(f"Reference Price: ${entry_price:,.2f}")
                print(f"Stop Loss: ${sl_price:,.2f} (-{risk_percent:.2f}%)")
                print(f"Take Profit: ${tp_price:,.2f} (+{reward_percent:.2f}%)")
                print(f"Total Position Size: {ladder_amount} BTC (${committed_notional:.2f} USDT)")
                print(f"Leverage: {leverage_level}x")
                print(f"Kelly Fraction: {kelly_fraction:.2%}")
                print(f"Win Probability: {win_probability:.2%}")
                print(f"Ladder Orders Planned: {len(ladder_orders)}")
                print(f"Trade ID: {active_trade_id}")
                print("==============================")

            elif final_action == "short":
                ladder_orders, ladder_amount, committed_notional = execute_entry_plan(
                    symbol, "short", final_order_size, entry_plan
                )
                if not ladder_orders:
                    print("Unable to place ladder orders for SHORT setup. Skipping trade.")
                    time.sleep(30)
                    continue

                entry_price = current_price
                sl_price = round(entry_price * (1 + risk_percent/100), 2)
                tp_price = round(entry_price * (1 - reward_percent/100), 2)

                # 트레이딩 내역 데이터베이스에 저장
                active_trade_id = save_trade(
                    action="short",
                    entry_price=entry_price,
                    amount=ladder_amount,
                    order_size=committed_notional,
                    leverage=leverage_level,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    kelly_fraction=kelly_fraction,
                    win_probability=win_probability,
                    volatility=market_volatility,
                    risk_percent=risk_percent,
                    reward_percent=reward_percent,
                    entry_plan=entry_plan,
                    ladder_orders=ladder_orders
                )

                print(f"\n=== SHORT Ladder Prepared ===")
                print(f"Reference Price: ${entry_price:,.2f}")
                print(f"Stop Loss: ${sl_price:,.2f} (+{risk_percent:.2f}%)")
                print(f"Take Profit: ${tp_price:,.2f} (-{reward_percent:.2f}%)")
                print(f"Total Position Size: {ladder_amount} BTC (${committed_notional:.2f} USDT)")
                print(f"Leverage: {leverage_level}x")
                print(f"Kelly Fraction: {kelly_fraction:.2%}")
                print(f"Win Probability: {win_probability:.2%}")
                print(f"Ladder Orders Planned: {len(ladder_orders)}")
                print(f"Trade ID: {active_trade_id}")
                print("===============================")
            else:
                print("final_action이 'long' 또는 'short'가 아니므로 주문을 실행하지 않습니다.")
        
        time.sleep(30)  # 30초마다 체크

    except Exception as e:
        print(f"\n Error: {e}")
        time.sleep(5)
