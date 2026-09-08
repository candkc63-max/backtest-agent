import pandas as pd
import numpy as np


def _parse_date_series(s: pd.Series) -> pd.Series:
    """Farklı CSV tarih formatlarını güvenli şekilde normalize eder."""
    if pd.api.types.is_numeric_dtype(s):
        numeric = pd.to_numeric(s, errors='coerce')
        finite = numeric.dropna()
        if finite.empty:
            return pd.Series(pd.NaT, index=s.index)

        med = float(finite.abs().median())

        # YYYYMMDD (örn. 20250909)
        if 19_000_000 <= med <= 21_999_999:
            txt = numeric.round().astype('Int64').astype(str)
            return pd.to_datetime(txt, format='%Y%m%d', errors='coerce')

        # Excel seri tarihi (yaklaşık 1954-2119 aralığı)
        if 20_000 <= med <= 80_000:
            return pd.to_datetime(numeric, unit='D', origin='1899-12-30', errors='coerce')

        # Unix nanos / millis / seconds
        if med > 100_000_000_000_000:
            return pd.to_datetime(numeric, unit='ns', errors='coerce')
        if med > 10_000_000_000:
            return pd.to_datetime(numeric, unit='ms', errors='coerce')
        if med > 100_000_000:
            return pd.to_datetime(numeric, unit='s', errors='coerce')

        # Sayısal ama bilinmeyen format: string olarak son şans.
        return pd.to_datetime(numeric.astype('Int64').astype(str), errors='coerce')

    text = s.astype(str).str.strip()
    parsed = pd.to_datetime(text, errors='coerce', dayfirst=False)
    if parsed.isna().mean() > 0.40:
        parsed2 = pd.to_datetime(text, errors='coerce', dayfirst=True)
        if parsed2.notna().sum() > parsed.notna().sum():
            parsed = parsed2
    return parsed


def _numeric_series(s: pd.Series) -> pd.Series:
    if not pd.api.types.is_object_dtype(s):
        return pd.to_numeric(s, errors='coerce')

    x = s.astype(str).str.strip().str.replace('\u00a0', '', regex=False)

    # Hem nokta hem virgül varsa son ayırıcıyı ondalık kabul et.
    both = x.str.contains(',', regex=False) & x.str.contains('.', regex=False)
    for idx in x[both].index:
        v = x.at[idx]
        if v.rfind(',') > v.rfind('.'):
            v = v.replace('.', '').replace(',', '.')
        else:
            v = v.replace(',', '')
        x.at[idx] = v

    comma_only = x.str.contains(',', regex=False) & ~x.str.contains('.', regex=False)
    x.loc[comma_only] = x.loc[comma_only].str.replace(',', '.', regex=False)
    return pd.to_numeric(x, errors='coerce')


def prepare_daily_flexible(df: pd.DataFrame) -> pd.DataFrame:
    """Günlük OHLCV CSV'lerini farklı sağlayıcılardan kabul eder."""
    out = df.copy()
    out.columns = [str(c).lower().strip() for c in out.columns]

    aliases = {
        'datetime': 'time', 'date': 'time', 'timestamp': 'time', 'tarih': 'time',
        'day': 'time', 'date/time': 'time',
        'opening': 'open', 'açılış': 'open', 'acilis': 'open',
        'highest': 'high', 'yüksek': 'high', 'yuksek': 'high',
        'lowest': 'low', 'düşük': 'low', 'dusuk': 'low',
        'last': 'close', 'closing': 'close', 'kapanış': 'close', 'kapanis': 'close',
        'adj close': 'close', 'adj_close': 'close',
        'vol': 'volume', 'hacim': 'volume', 'volume tl': 'volume',
    }
    rename = {}
    for c in out.columns:
        if c in aliases and aliases[c] not in out.columns:
            rename[c] = aliases[c]
    out = out.rename(columns=rename)

    if 'time' not in out.columns:
        for c in out.columns:
            lc = c.lower()
            if any(k in lc for k in ('date', 'time', 'tarih', 'timestamp', 'day')):
                out = out.rename(columns={c: 'time'})
                break

    required = {'time', 'open', 'high', 'low', 'close', 'volume'}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Eksik sütunlar: {sorted(missing)} | Bulunan: {list(out.columns)}")

    out['date'] = _parse_date_series(out['time'])
    for c in ['open', 'high', 'low', 'close', 'volume']:
        out[c] = _numeric_series(out[c])

    out = out.dropna(subset=['date', 'open', 'high', 'low', 'close', 'volume']).copy()
    out = out.sort_values('date').reset_index(drop=True)

    if len(out) < 350:
        raise ValueError(f"En az yaklaşık 350 günlük veri gerekli. Kullanılabilir satır: {len(out)}")

    # Günlük veriyi mum aralığıyla değil, aynı takvim gününde kaç kayıt olduğuyla doğrula.
    # Bazı sağlayıcılar günlük muma 03:00/06:00 gibi saat eklediği için saat farkı testi yanıltıcıdır.
    day_key = out['date'].dt.normalize()
    rows_per_day = day_key.value_counts()
    duplicate_day_ratio = float((rows_per_day > 1).mean()) if len(rows_per_day) else 1.0
    avg_rows_per_day = len(out) / max(day_key.nunique(), 1)

    if avg_rows_per_day > 1.20 or duplicate_day_ratio > 0.10:
        raise ValueError(
            f"Veri günlük görünmüyor: ortalama {avg_rows_per_day:.2f} kayıt/gün. 1D CSV gerekli."
        )

    # Aynı güne düşen nadir duplikeleri son kayıtla tekilleştir.
    out['_day'] = day_key
    out = out.drop_duplicates('_day', keep='last').drop(columns=['_day']).reset_index(drop=True)

    return out
