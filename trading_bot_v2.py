"""
ML Trading Bot v2 — Full Suite Backend
Features: Backtesting | Live Alerts | Analytics | Full REST API
Run: python trading_bot_v2.py
Then open trading_dashboard_v2.html
"""
import time, json, threading, smtplib, logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from typing import Optional, Callable
from email.mime.text import MIMEText
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from flask import Flask, jsonify, request
from flask_cors import CORS

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
log = logging.getLogger('MLBot')

# ─── CONFIG ───────────────────────────────────────────────────────────────────

class Config:
    STARTING_BALANCE   = 10_000.0
    RISK_PER_TRADE     = 0.02
    TAKE_PROFIT_PCT    = 0.03
    STOP_LOSS_PCT      = 0.015
    LOOKBACK_DAYS      = 120
    DEFAULT_SYMBOL     = "AAPL"
    SYMBOLS            = ["AAPL", "TSLA", "MSFT", "NVDA", "SPY"]
    MODEL_TYPE         = "random_forest"
    RETRAIN_INTERVAL   = 600
    CV_FOLDS           = 5
    BACKTEST_DAYS      = 252
    TRANSACTION_COST   = 0.001
    ALERT_EMAIL_FROM   = ""   # Set your Gmail here
    ALERT_EMAIL_PASS   = ""   # Gmail app password
    ALERT_EMAIL_TO     = ""   # Recipient email
    API_HOST           = "0.0.0.0"
    API_PORT           = 5000
    TICK_INTERVAL      = 30

# ─── DATA MODELS ──────────────────────────────────────────────────────────────

@dataclass
class Trade:
    id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: Optional[float] = None
    quantity: float = 0.0
    entry_time: str = ""
    exit_time: Optional[str] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    status: str = "open"
    exit_reason: str = ""

@dataclass
class Signal:
    symbol: str
    action: str
    confidence: float
    price: float
    rsi: float
    macd: float
    ma_cross: float
    timestamp: str
    features: dict = field(default_factory=dict)

@dataclass
class Alert:
    id: str
    symbol: str
    condition: str
    threshold: Optional[float]
    triggered: bool = False
    created_at: str = ""
    triggered_at: Optional[str] = None

@dataclass
class BacktestResult:
    strategy: str
    symbol: str
    start_date: str
    end_date: str
    starting_capital: float
    ending_capital: float
    total_return_pct: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float
    equity_curve: list
    trades: list

# ─── ANALYTICS ────────────────────────────────────────────────────────────────

class Analytics:
    @staticmethod
    def sharpe(returns, rf=0.05, periods=252):
        if len(returns) < 2: return 0.0
        ex = returns - rf/periods
        std = ex.std()
        return float(ex.mean()/std*np.sqrt(periods)) if std > 0 else 0.0

    @staticmethod
    def sortino(returns, rf=0.05, periods=252):
        if len(returns) < 2: return 0.0
        ex = returns - rf/periods
        dn = returns[returns < 0]
        std = dn.std() if len(dn) > 0 else 1e-10
        return float(ex.mean()/std*np.sqrt(periods))

    @staticmethod
    def max_drawdown(eq):
        curve = np.array(eq)
        peak = curve[0]
        mdd = 0.0
        pi = ti = 0
        cpi = 0
        for i, v in enumerate(curve):
            if v > peak:
                peak = v
                cpi = i
            dd = (peak - v) / peak
            if dd > mdd:
                mdd = dd
                pi = cpi
                ti = i
        return mdd, pi, ti

    @staticmethod
    def profit_factor(pnls):
        g = sum(p for p in pnls if p > 0)
        l = abs(sum(p for p in pnls if p < 0))
        return g/l if l > 0 else float('inf')

    @staticmethod
    def win_rate(pnls):
        return sum(1 for p in pnls if p > 0)/len(pnls) if pnls else 0.0

    @staticmethod
    def to_returns(eq):
        arr = np.array(eq)
        return np.diff(arr)/arr[:-1]

    @staticmethod
    def report(equity, trades, start_capital):
        pnls = [t.pnl for t in trades if t.pnl is not None]
        rets = Analytics.to_returns(equity)
        tr = (equity[-1] - start_capital) / start_capital
        ny = len(equity) / 252
        ann = (1 + tr) ** (1 / max(ny, 0.01)) - 1
        mdd, _, _ = Analytics.max_drawdown(equity)
        return {
            "total_return_pct":  round(tr*100, 2),
            "annualized_return": round(ann*100, 2),
            "sharpe_ratio":      round(Analytics.sharpe(rets), 2),
            "sortino_ratio":     round(Analytics.sortino(rets), 2),
            "calmar_ratio":      round(ann/mdd if mdd > 0 else 0, 2),
            "max_drawdown_pct":  round(mdd*100, 2),
            "win_rate":          round(Analytics.win_rate(pnls)*100, 1),
            "profit_factor":     round(Analytics.profit_factor(pnls), 2),
            "total_trades":      len(trades),
            "avg_pnl":           round(sum(pnls)/len(pnls), 2) if pnls else 0,
            "best_trade":        round(max(pnls), 2) if pnls else 0,
            "worst_trade":       round(min(pnls), 2) if pnls else 0,
        }

# ─── FEATURE ENGINEERING ──────────────────────────────────────────────────────

class FE:
    COLS = [
        'sma20', 'sma50', 'ema12', 'ema26', 'macd', 'macd_s', 'macd_h',
        'rsi', 'stoch_k', 'stoch_d', 'bb_w', 'bb_pct', 'atr',
        'ret1', 'ret5', 'ret10', 'vsma20', 'vsma50', 'macross', 'volr',
        'c_lag1', 'c_lag2', 'c_lag3', 'c_lag5', 'r_lag1', 'r_lag2', 'r_lag3', 'r_lag5'
    ]

    @staticmethod
    def compute(df):
        f = df.copy()
        f['sma20']  = SMAIndicator(f['close'], 20).sma_indicator()
        f['sma50']  = SMAIndicator(f['close'], 50).sma_indicator()
        f['ema12']  = EMAIndicator(f['close'], 12).ema_indicator()
        f['ema26']  = EMAIndicator(f['close'], 26).ema_indicator()
        m = MACD(f['close'])
        f['macd'] = m.macd()
        f['macd_s'] = m.macd_signal()
        f['macd_h'] = m.macd_diff()
        f['rsi'] = RSIIndicator(f['close'], 14).rsi()
        st = StochasticOscillator(f['high'], f['low'], f['close'])
        f['stoch_k'] = st.stoch()
        f['stoch_d'] = st.stoch_signal()
        bb = BollingerBands(f['close'])
        f['bb_w'] = bb.bollinger_hband() - bb.bollinger_lband()
        f['bb_pct'] = bb.bollinger_pband()
        f['atr'] = AverageTrueRange(f['high'], f['low'], f['close']).average_true_range()
        f['ret1']  = f['close'].pct_change(1)
        f['ret5']  = f['close'].pct_change(5)
        f['ret10'] = f['close'].pct_change(10)
        f['vsma20'] = (f['close'] - f['sma20']) / f['sma20']
        f['vsma50'] = (f['close'] - f['sma50']) / f['sma50']
        f['macross'] = f['sma20'] - f['sma50']
        f['volr'] = f['volume'] / f['volume'].rolling(20).mean()
        for lg in [1, 2, 3, 5]:
            f[f'c_lag{lg}'] = f['close'].shift(lg)
            f[f'r_lag{lg}'] = f['rsi'].shift(lg)
        return f.dropna()

    @staticmethod
    def labels(df, horizon=5, threshold=0.01):
        fut = df['close'].shift(-horizon) / df['close'] - 1
        lbl = pd.Series(0, index=df.index)
        lbl[fut >  threshold] =  1
        lbl[fut < -threshold] = -1
        return lbl

# ─── STRATEGIES ───────────────────────────────────────────────────────────────

class Strategies:
    @staticmethod
    def ma_cross(df):
        f = FE.compute(df)
        if len(f) < 2: return 'HOLD'
        if f['sma20'].iloc[-1] > f['sma50'].iloc[-1] and f['sma20'].iloc[-2] <= f['sma50'].iloc[-2]:
            return 'BUY'
        if f['sma20'].iloc[-1] < f['sma50'].iloc[-1] and f['sma20'].iloc[-2] >= f['sma50'].iloc[-2]:
            return 'SELL'
        return 'HOLD'

    @staticmethod
    def rsi_revert(df, oversold=30, overbought=70):
        f = FE.compute(df)
        r = f['rsi'].iloc[-1]
        return 'BUY' if r < oversold else 'SELL' if r > overbought else 'HOLD'

    @staticmethod
    def macd_cross(df):
        f = FE.compute(df)
        if len(f) < 2: return 'HOLD'
        if f['macd_h'].iloc[-1] > 0 and f['macd_h'].iloc[-2] <= 0:
            return 'BUY'
        if f['macd_h'].iloc[-1] < 0 and f['macd_h'].iloc[-2] >= 0:
            return 'SELL'
        return 'HOLD'

    @staticmethod
    def bollinger(df):
        f = FE.compute(df)
        bb = BollingerBands(df['close'])
        p = df['close'].iloc[-1]
        if p <= bb.bollinger_lband().iloc[-1]: return 'BUY'
        if p >= bb.bollinger_hband().iloc[-1]: return 'SELL'
        return 'HOLD'

# ─── BACKTESTER ───────────────────────────────────────────────────────────────

class Backtester:
    def __init__(self, ml_model=None, ml_scaler=None):
        self.ml_model = ml_model
        self.ml_scaler = ml_scaler

    def run(self, symbol, strategy, days=Config.BACKTEST_DAYS, capital=Config.STARTING_BALANCE):
        log.info(f"[BT] {strategy} on {symbol} {days}d ${capital:,.0f}")
        df = DataFetcher.fetch(symbol, days=days+60)
        if len(df) < 60:
            raise ValueError("Not enough data")
        bal = capital
        pos = None
        equity = [capital]
        trades = []
        n = 0
        for i in range(60, len(df)):
            win = df.iloc[:i]
            price = float(df['close'].iloc[i])
            sig = self._sig(win, strategy)
            if pos:
                ret = (price - pos['e']) / pos['e'] * (1 if pos['s'] == 'BUY' else -1)
                if ret >= Config.TAKE_PROFIT_PCT:
                    t = self._close(pos, price, i, 'tp', n, symbol)
                    bal += t.pnl * (1 - Config.TRANSACTION_COST)
                    trades.append(t); n += 1; pos = None
                elif ret <= -Config.STOP_LOSS_PCT:
                    t = self._close(pos, price, i, 'sl', n, symbol)
                    bal += t.pnl * (1 - Config.TRANSACTION_COST)
                    trades.append(t); n += 1; pos = None
            if sig == 'BUY' and not pos:
                qty = bal * Config.RISK_PER_TRADE / price
                pos = {'s': 'BUY', 'e': price, 'i': i, 'q': qty}
                bal -= price * qty * Config.TRANSACTION_COST
            elif sig == 'SELL' and pos and pos['s'] == 'BUY':
                t = self._close(pos, price, i, 'sig', n, symbol)
                bal += t.pnl * (1 - Config.TRANSACTION_COST)
                trades.append(t); n += 1; pos = None
            equity.append(round(bal, 2))
        if pos:
            t = self._close(pos, float(df['close'].iloc[-1]), len(df)-1, 'eod', n, symbol)
            bal += t.pnl
            trades.append(t)
            equity.append(round(bal, 2))
        pnls = [t.pnl for t in trades if t.pnl]
        mdd, _, _ = Analytics.max_drawdown(equity)
        rets = Analytics.to_returns(equity)
        tr = (bal - capital) / capital
        ny = days / 252
        ann = (1 + tr) ** (1 / max(ny, .01)) - 1
        return BacktestResult(
            strategy=strategy, symbol=symbol,
            start_date=str(df.index[60].date()), end_date=str(df.index[-1].date()),
            starting_capital=capital, ending_capital=round(bal, 2),
            total_return_pct=round(tr*100, 2), annualized_return=round(ann*100, 2),
            sharpe_ratio=round(Analytics.sharpe(rets), 2),
            sortino_ratio=round(Analytics.sortino(rets), 2),
            calmar_ratio=round(ann/mdd if mdd > 0 else 0, 2),
            max_drawdown_pct=round(mdd*100, 2),
            win_rate=round(Analytics.win_rate(pnls)*100, 1),
            profit_factor=round(Analytics.profit_factor(pnls), 2),
            total_trades=len(trades), avg_trade_duration=0.0,
            equity_curve=equity, trades=[asdict(t) for t in trades]
        )

    def _sig(self, df, strat):
        try:
            if strat == 'ml' and self.ml_model:
                feats = FE.compute(df)
                row = feats[FE.COLS].iloc[[-1]]
                return {1: 'BUY', -1: 'SELL', 0: 'HOLD'}.get(
                    int(self.ml_model.predict(self.ml_scaler.transform(row))[0]), 'HOLD'
                )
            return {
                'ma_cross':  Strategies.ma_cross,
                'rsi':       Strategies.rsi_revert,
                'macd':      Strategies.macd_cross,
                'bollinger': Strategies.bollinger
            }.get(strat, lambda df: 'HOLD')(df)
        except:
            return 'HOLD'

    def _close(self, pos, ep, idx, reason, n, sym):
        pnl = (ep - pos['e']) * pos['q'] * (1 if pos['s'] == 'BUY' else -1)
        return Trade(
            id=f"BT{n:04d}", symbol=sym, side=pos['s'],
            entry_price=pos['e'], exit_price=ep,
            quantity=pos['q'], entry_time=str(pos['i']), exit_time=str(idx),
            pnl=round(pnl, 2), pnl_pct=round(pnl/(pos['e']*pos['q'])*100, 2),
            status='closed', exit_reason=reason
        )

# ─── ALERT MANAGER ────────────────────────────────────────────────────────────

class AlertManager:
    def __init__(self):
        self.alerts = []
        self._cbs = []
        self._n = 0

    def add(self, symbol, condition, threshold=None):
        self._n += 1
        a = Alert(
            id=f"AL{self._n:04d}", symbol=symbol, condition=condition,
            threshold=threshold, created_at=datetime.now().isoformat()
        )
        self.alerts.append(a)
        log.info(f"[Alert] Added {symbol} {condition} {threshold}")
        return a

    def remove(self, aid):
        before = len(self.alerts)
        self.alerts = [a for a in self.alerts if a.id != aid]
        return len(self.alerts) < before

    def check(self, sym, price, signal=None):
        fired = []
        for a in self.alerts:
            if a.triggered or a.symbol != sym:
                continue
            ok = False
            if a.condition == 'above' and a.threshold and price >= a.threshold:
                ok = True
            if a.condition == 'below' and a.threshold and price <= a.threshold:
                ok = True
            if a.condition == 'buy_signal' and signal == 'BUY':
                ok = True
            if a.condition == 'sell_signal' and signal == 'SELL':
                ok = True
            if ok:
                a.triggered = True
                a.triggered_at = datetime.now().isoformat()
                msg = f"🔔 {a.symbol} {a.condition} ${a.threshold or ''} — now ${price:.2f}"
                log.info(f"[Alert] FIRED: {msg}")
                self._notify(a, msg)
                fired.append(a)
        self.alerts = [a for a in self.alerts if not a.triggered]
        return fired

    def _notify(self, a, msg):
        try:
            from plyer import notification
            notification.notify(title="MLBot Alert", message=msg, timeout=5)
        except:
            pass
        if Config.ALERT_EMAIL_FROM and Config.ALERT_EMAIL_TO:
            try:
                m = MIMEText(msg)
                m['Subject'] = f"MLBot: {a.symbol}"
                m['From'] = Config.ALERT_EMAIL_FROM
                m['To'] = Config.ALERT_EMAIL_TO
                with smtplib.SMTP_SSL('smtp.gmail.com', 465) as s:
                    s.login(Config.ALERT_EMAIL_FROM, Config.ALERT_EMAIL_PASS)
                    s.send_message(m)
            except Exception as e:
                log.warning(f"[Alert] Email failed: {e}")
        for cb in self._cbs:
            try:
                cb(a, msg)
            except:
                pass

    @property
    def active(self):
        return [a for a in self.alerts if not a.triggered]

# ─── ML MODEL ─────────────────────────────────────────────────────────────────

class MLModel:
    MODELS = {
        "random_forest":   lambda: RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        "gradient_boost":  lambda: GradientBoostingClassifier(n_estimators=100, learning_rate=.1, random_state=42),
        "logistic":        lambda: LogisticRegression(max_iter=500, random_state=42),
    }

    def __init__(self, mtype="random_forest"):
        self.model = self.MODELS[mtype]()
        self.scaler = StandardScaler()
        self.is_trained = False
        self.accuracy = 0.0
        self.cv_scores = []

    def train(self, df):
        f = FE.compute(df)
        X = f[FE.COLS].dropna()
        y = FE.labels(f).loc[X.index]
        mask = y.notna() & (y != 0)
        X, y = X[mask], y[mask]
        if len(X) < 50:
            return 0.0
        scores = []
        for tri, vi in TimeSeriesSplit(n_splits=Config.CV_FOLDS).split(X):
            sc = StandardScaler()
            self.model.fit(sc.fit_transform(X.iloc[tri]), y.iloc[tri])
            scores.append(accuracy_score(y.iloc[vi], self.model.predict(sc.transform(X.iloc[vi]))))
        self.scaler.fit(X)
        self.model.fit(self.scaler.transform(X), y)
        self.accuracy = float(np.mean(scores))
        self.cv_scores = scores
        self.is_trained = True
        log.info(f"[ML] Trained | CV={self.accuracy:.2%} ±{np.std(scores):.2%}")
        return self.accuracy

    def predict(self, df):
        if not self.is_trained:
            return Signal(
                symbol="", action="HOLD", confidence=.5, price=0,
                rsi=50, macd=0, ma_cross=0, timestamp=datetime.now().isoformat()
            )
        f = FE.compute(df)
        row = f[FE.COLS].iloc[[-1]]
        pred = self.model.predict(self.scaler.transform(row))[0]
        proba = self.model.predict_proba(self.scaler.transform(row))[0]
        return Signal(
            symbol=df.attrs.get('symbol', ''),
            action={1: 'BUY', -1: 'SELL', 0: 'HOLD'}.get(int(pred), 'HOLD'),
            confidence=float(max(proba)),
            price=float(f['close'].iloc[-1]),
            rsi=float(f['rsi'].iloc[-1]),
            macd=float(f['macd'].iloc[-1]),
            ma_cross=float(f['macross'].iloc[-1]),
            timestamp=datetime.now().isoformat()
        )

# ─── DATA FETCHER ─────────────────────────────────────────────────────────────

class DataFetcher:
    @staticmethod
    def fetch(symbol, days=90, interval="1d"):
        end = datetime.now()
        start = end - timedelta(days=days)
        df = yf.Ticker(symbol).history(start=start, end=end, interval=interval)
        if df.empty:
            raise ValueError(f"No data for {symbol}")
        df.columns = [c.lower() for c in df.columns]
        df = df[['open', 'high', 'low', 'close', 'volume']].dropna()
        df.attrs['symbol'] = symbol
        return df

# ─── PAPER TRADER ─────────────────────────────────────────────────────────────

class PaperTrader:
    def __init__(self, bal=Config.STARTING_BALANCE):
        self.balance = bal
        self.start = bal
        self.today = bal
        self.open = []
        self.closed = []
        self._n = 0

    def execute(self, sig):
        if sig.action == 'HOLD':
            return None
        self._close_opp(sig)
        if any(t.side == sig.action for t in self.open):
            return None
        return self._open(sig)

    def _open(self, sig):
        self._n += 1
        qty = (self.balance * Config.RISK_PER_TRADE) / sig.price
        t = Trade(
            id=f"T{self._n:04d}", symbol=sig.symbol, side=sig.action,
            entry_price=sig.price, quantity=round(qty, 4),
            entry_time=sig.timestamp, status="open"
        )
        self.open.append(t)
        return t

    def _close_t(self, t, ep, reason):
        t.exit_price = ep
        t.exit_time = datetime.now().isoformat()
        t.pnl = round((ep - t.entry_price) * t.quantity * (1 if t.side == 'BUY' else -1), 2)
        t.pnl_pct = round(t.pnl / (t.entry_price * t.quantity) * 100, 2) if t.entry_price * t.quantity else 0
        t.exit_reason = reason
        t.status = "closed"
        self.balance = round(self.balance + t.pnl, 2)
        self.open.remove(t)
        self.closed.append(t)

    def _close_opp(self, sig):
        opp = 'SELL' if sig.action == 'BUY' else 'BUY'
        for t in list(self.open):
            if t.side == opp:
                self._close_t(t, sig.price, 'signal')

    def check_tpsl(self, price):
        for t in list(self.open):
            ret = (price - t.entry_price) / t.entry_price * (1 if t.side == 'BUY' else -1)
            if ret >= Config.TAKE_PROFIT_PCT:
                self._close_t(t, price, 'take_profit')
            elif ret <= -Config.STOP_LOSS_PCT:
                self._close_t(t, price, 'stop_loss')

    @property
    def equity_curve(self):
        return [self.start] + [
            self.start + sum(t.pnl for t in self.closed[:i+1])
            for i in range(len(self.closed))
        ]

    @property
    def analytics(self):
        return Analytics.report(self.equity_curve, self.closed, self.start)

# ─── BOT ──────────────────────────────────────────────────────────────────────

class TradingBot:
    def __init__(self, symbol=Config.DEFAULT_SYMBOL):
        self.symbol = symbol
        self.model = MLModel(Config.MODEL_TYPE)
        self.trader = PaperTrader()
        self.alerts = AlertManager()
        self.backtester = Backtester()
        self.running = False
        self._thread = None
        self.last_signal = None
        self.last_df = None
        self._trained = 0.0
        self._boot()

    def _boot(self):
        try:
            df = DataFetcher.fetch(self.symbol)
            self.last_df = df
            self.model.train(df)
            self.backtester.ml_model = self.model.model
            self.backtester.ml_scaler = self.model.scaler
            self._trained = time.time()
        except Exception as e:
            log.warning(f"[Bot] Boot: {e}")

    def start(self):
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    def _loop(self):
        while self.running:
            try:
                self._tick()
            except Exception as e:
                log.error(f"[Bot] {e}")
            time.sleep(Config.TICK_INTERVAL)

    def _tick(self):
        df = DataFetcher.fetch(self.symbol)
        self.last_df = df
        if time.time() - self._trained > Config.RETRAIN_INTERVAL:
            self.model.train(df)
            self.backtester.ml_model = self.model.model
            self.backtester.ml_scaler = self.model.scaler
            self._trained = time.time()
        sig = self.model.predict(df)
        sig.symbol = self.symbol
        self.last_signal = sig
        price = float(df['close'].iloc[-1])
        self.trader.check_tpsl(price)
        self.trader.execute(sig)
        self.alerts.check(self.symbol, price, sig.action)

    def backtest(self, strategy, symbol, days, capital):
        return self.backtester.run(symbol, strategy, days, capital)

    def status(self):
        t = self.trader
        an = t.analytics if t.closed else {}
        return {
            "symbol":        self.symbol,
            "running":       self.running,
            "balance":       round(t.balance, 2),
            "start_balance": t.start,
            "total_pnl":     round(t.balance - t.start, 2),
            "today_pnl":     round(t.balance - t.today, 2),
            "trade_count":   len(t.closed),
            "open_trades":   [asdict(x) for x in t.open],
            "recent_trades": [asdict(x) for x in t.closed[-20:][::-1]],
            "equity_curve":  t.equity_curve,
            "analytics":     an,
            "last_signal":   asdict(self.last_signal) if self.last_signal else None,
            "model_accuracy":  round(self.model.accuracy * 100, 1),
            "model_cv_scores": [round(s*100, 1) for s in self.model.cv_scores],
            "active_alerts":   [asdict(a) for a in self.alerts.active],
        }

# ─── API ──────────────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)
bot = TradingBot()

@app.route("/status")
def status():
    return jsonify(bot.status())

@app.route("/start", methods=["POST"])
def start():
    bot.start()
    return jsonify({"ok": True})

@app.route("/stop", methods=["POST"])
def stop():
    bot.stop()
    return jsonify({"ok": True})

@app.route("/symbol", methods=["POST"])
def symbol():
    sym = request.json.get("symbol", Config.DEFAULT_SYMBOL)
    bot.stop()
    bot.__init__(sym)
    return jsonify({"ok": True, "symbol": sym})

@app.route("/backtest", methods=["POST"])
def backtest():
    d = request.json or {}
    try:
        r = bot.backtest(
            d.get("strategy", "ml"),
            d.get("symbol", Config.DEFAULT_SYMBOL),
            int(d.get("days", Config.BACKTEST_DAYS)),
            float(d.get("capital", Config.STARTING_BALANCE))
        )
        return jsonify(asdict(r))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/analytics")
def analytics():
    t = bot.trader
    if not t.closed:
        return jsonify({"message": "No closed trades yet"})
    return jsonify({**t.analytics, "equity_curve": t.equity_curve})

@app.route("/alerts", methods=["GET"])
def get_alerts():
    return jsonify({"alerts": [asdict(a) for a in bot.alerts.alerts]})

@app.route("/alerts", methods=["POST"])
def add_alert():
    d = request.json or {}
    a = bot.alerts.add(d.get("symbol", "AAPL"), d.get("condition", "above"), d.get("threshold"))
    return jsonify(asdict(a))

@app.route("/alerts/<aid>", methods=["DELETE"])
def del_alert(aid):
    return jsonify({"ok": bot.alerts.remove(aid)})

@app.route("/history")
def history():
    if bot.last_df is None:
        return jsonify({"error": "No data"}), 404
    try:
        f = FE.compute(bot.last_df).tail(80)
        return jsonify({
            "dates":  f.index.strftime("%Y-%m-%d").tolist(),
            "close":  f['close'].tolist(),
            "sma20":  f['sma20'].tolist(),
            "sma50":  f['sma50'].tolist(),
            "rsi":    f['rsi'].tolist(),
            "macd":   f['macd'].tolist(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════╗
║   ML TRADING BOT v2 — FULL SUITE                     ║
║   ✅ Backtesting  ✅ Alerts  ✅ Analytics  ✅ API    ║
║   ⚠  PAPER TRADING ONLY — No real money.            ║
║   Dashboard: open trading_dashboard_v2.html          ║
╚══════════════════════════════════════════════════════╝
API @ http://localhost:5000
""")
    app.run(host=Config.API_HOST, port=Config.API_PORT, debug=False)
