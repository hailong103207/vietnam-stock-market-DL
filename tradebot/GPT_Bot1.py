'''
Description:
Nguyên tắc chung của chiến lược
Chỉ vào lệnh khi có xu hướng rõ hoặc tín hiệu breakout mạnh.

Kết hợp 3 nhóm chỉ báo để lọc tín hiệu:
Xu hướng (Trend) → xác định hướng chính.
Động lượng (Momentum/Oscillator) → đo sức mạnh, tránh vào lúc quá mua/quá bán.
Khối lượng/biến động (Volume + Volatility) → xác nhận tín hiệu & đặt stop-loss hợp lý.
Quản trị rủi ro: ATR để đặt SL, RR ≥ 1:2.

Chiến lược đề xuất: Trend + Pullback + Volume Confirm

1. Xác định xu hướng chính
Dùng EMA 20 & EMA 50 (hoặc Ichimoku/Supertrend).
Xu hướng tăng = EMA20 > EMA50 và giá nằm trên EMA20.
Xu hướng giảm = EMA20 < EMA50 và giá nằm dưới EMA20.

2. Tìm điểm vào lệnh (Entry Signal)
Khi có pullback về EMA20/EMA50 hoặc chạm biên giữa Bollinger Bands.
Kết hợp RSI hoặc Stochastic:
Xu hướng tăng: RSI > 50 và Stoch không ở vùng quá mua.
Xu hướng giảm: RSI < 50 và Stoch không ở vùng quá bán.
MACD hoặc ADX để xác nhận xu hướng mạnh:
MACD histogram dương (tăng) hoặc âm (giảm).
ADX > 20 → xu hướng đủ mạnh để vào lệnh.

3. Xác nhận bằng Volume
Dùng MFI hoặc OBV/CMF:
Nếu breakout/tiếp diễn xu hướng kèm dòng tiền vào → tín hiệu mạnh.
Nếu volume yếu → bỏ qua, tránh fake breakout.

4. Đặt stop-loss và take-profit
Stop-loss: dựa vào ATR(14), đặt SL cách entry 1-1.5 x ATR.
Take-profit:
TP1 = Risk:Reward 1:2.
TP2 = vùng Fibonacci extension (127% hoặc 161.8%) hoặc biên Bollinger trên/dưới.

5. Thoát lệnh (Exit)
Nếu RSI/Stoch báo quá mua/quá bán cực mạnh + phân kỳ → thoát sớm.
Nếu giá phá EMA50 ngược xu hướng chính → cắt lỗ hoặc thoát toàn bộ.

Ví dụ minh họa (xu hướng tăng ngắn hạn)
EMA20 > EMA50 → xu hướng tăng.
Giá pullback về EMA20 + RSI vẫn trên 50.
MACD histogram > 0 và ADX ~ 25 (trend mạnh).

MFI tăng (dòng tiền vào).
Entry Buy.

SL = 1.2 x ATR dưới EMA50.
TP = 2 x RR hoặc Fibo 161.8%.
'''

import pandas as pd

class GPT_Bot1():
    pass