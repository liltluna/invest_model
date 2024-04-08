import numpy as np
import math


def calculate_RSI(prices, period=6):
    """
    calculate RSI indicator
    parameter:
    data (list): a list including the close price
    period (int): RSI period，default is 6

    return:
    list: a list including the RIS result.
    """
    rsi_values = []
    delta = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
    gain = [delta[i] if delta[i] > 0 else 0 for i in range(len(delta))]
    loss = [-delta[i] if delta[i] < 0 else 0 for i in range(len(delta))]

    avg_gain = sum(gain[:period]) / period
    avg_loss = sum(loss[:period]) / period

    for i in range(0, period):
        rsi_values.append(float('nan'))

    if avg_loss == 0:
        rsi_values.append(100)
    else:
        rsi_values.append(100 - (100 / (1 + (avg_gain / avg_loss))))

    for i in range(period, len(prices) - 1):
        avg_gain = (avg_gain * period + gain[i] - gain[i - period]) / period
        avg_loss = (avg_loss * period + loss[i] - loss[i - period]) / period
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        rsi_values.append(rsi)

    return rsi_values


def calculate_WILLIAMS_PERSENT_R(high, low, close, period=14):
    """
    description: Calculate Williams %R indicator.

    Args:
    high (list or numpy array): High prices for each period.
    low (list or numpy array): Low prices for each period.
    close (list or numpy array): Close prices for each period.
    period (int): Period for Williams %R calculation.
    Returns:
    list: Williams %R values.
    """
    williams_r_values = []
    for i in range(len(close)):
        if i >= period - 1:
            highest_high = max(high[i - period + 1:i + 1])
            lowest_low = min(low[i - period + 1:i + 1])
            if highest_high == lowest_low:
                williams_r_values.append(-100)  # Avoid division by zero
            else:
                williams_r_values.append(
                    ((highest_high - close[i]) / (highest_high - lowest_low)) * -100)
        else:
            # For the initial periods where enough data is not available
            williams_r_values.append(float('nan'))
    return williams_r_values


def calculate_SMA(prices, period=6):
    """
    Calculate a simple moving average of stock data（SMA）

    parameters:
    prices (list): a list consists of stock prices.
    period (int): period of SMA, the default value is 6.
    resturn values:
    sma_values (list): list consists of SMA，length is len(prices) - period + 1
    """
    sma_values = []
    for i in range(period - 1):
        sma_values.append(float('nan'))

    for i in range(len(prices) - period + 1):
        sma = sum(prices[i:i+period]) / period
        sma_values.append(sma)

    return sma_values


def calculate_EMA(prices, period=6):
    """
    Calculate the exponential moving average of stock data

    parameters:
    prices (list)
    period (int)

    returns:
    ema_values (list)

    Current EMA= ((Price(current) - previous EMA)) X multiplier) + previous EMA.
    The important factor is the smoothing constant that = 2/(1+N) where N = the number of days.

    """
    ema_values = []
    smoothing = 2 / (period + 1)

    for i in range(period - 1):
        ema_values.append(float('nan'))
    # for i in range(len(prices)):
    #     if prices[i]:
    #         ema = sum(prices[i : i+period]) / period

    # initial ema value
    ema = sum(prices[:period]) / period
    ema_values.append(ema)

    for price in prices[period:]:
        ema = (price - ema) * smoothing + ema
        ema_values.append(ema)

    return ema_values


def calculate_WMA(prices, period=10):
    """
    calculate series WMA

    parameters:
    prices (list)
    period (int)

    returns:
    wma_values (list)
    """
    wma_values = []
    for i in range(period - 1):
        wma_values.append(float('nan'))

    weights = list(range(1, period + 1))
    for i in range(len(prices) - period + 1):
        wma = sum(prices[i: i + period][:: -1][j] * weights[j]
                  for j in range(period)) / sum(weights)
        wma_values.append(wma)

    return wma_values


def WMA(data, window):
    """
    This is only for the smooth_HMA, to calculate series WMA, use calculate_wma
    """
    weights = np.arange(1, window + 1)
    return np.convolve(data[::-1], weights, mode='valid') / weights.sum()


def calculate_smooth_HMA(prices, period):
    # Ensure the length of data is at least as long as the period
    if len(prices) < period:
        raise ValueError(
            "Length of data should be at least as long as the period")

    # Calculate WMA1 and WMA2
    # wma1 = WMA(price[-period:], period // 2)
    # wma2 = WMA(price[-period:], period)
    wma1 = [x if not math.isnan(x) else 0 for x in calculate_WMA(prices, int(period / 2))]
    wma2 = [x if not math.isnan(x) else 0 for x in calculate_WMA(prices, int(period))]
    smooth_hma = []
    # Calculate Raw HMA
    raw_hma = [2 * x - y if x != 0 and y != 0 else float('nan') for x, y in zip(wma1, wma2) ]
    # Calculate smooth HMA
    smooth_hma += calculate_WMA(raw_hma, int(np.sqrt(period)))

    return smooth_hma

def calculate_triple_EMA(prices, period=6):
    """
    calculate Triple Exponential Moving Average

    parameters:
    prices (list)
    period (int)

    returns:
    tema_values (list)
    """

    if len(prices) < (period - 1) * 3:
        return [float('nan')] * len(prices)
    tema_values = []
    tema_values += [float('nan')] * ((period - 1) * 3)

    ema1 = EMA(prices, period)
    ema2 = EMA(ema1, period)
    ema3 = EMA(ema2, period)

    tema_values += [3 * ema1[i] - 3 * ema2[i] + ema3[i]
                    for i in range(len(ema3))]

    return tema_values


def EMA(prices, period):
    """
    EMA
    """
    ema_values = []
    smoothing = 2 / (period + 1)
    ema = sum(prices[:period]) / period
    ema_values.append(ema)

    for price in prices[period:]:
        ema = (price - ema) * smoothing + ema
        ema_values.append(ema)

    return ema_values


def calculate_CCI(high_prices, low_prices, close_prices, period=20):
    """
    Calculate CCI indicator.

    Args:
    high_prices (list or numpy array): High prices for each period.
    low_prices (list or numpy array): Low prices for each period.
    close_prices (list or numpy array): Close prices for each period.
    period (int): Period for CCI calculation, the default is 20.

    Returns:
    list: CCI valuess

    source: https://school.stockcharts.com/doku.php?id=technical_indicators:commodity_channel_index_cci
    """
    typical_prices = [(high + low + close) / 3 for high, low,
                      close in zip(high_prices, low_prices, close_prices)]
    mean_deviation = DEVIATION(typical_prices, period)
    sma_typical_prices = SMA(typical_prices, period)

    cci_values = []
    for i in range(len(typical_prices)):
        if i < period:
            cci_values.append(float('nan'))
            continue
        cci = (typical_prices[i] - sma_typical_prices[i]
               ) / (0.015 * mean_deviation[i])
        cci_values.append(cci)

    return cci_values


def DEVIATION(data, period):
    deviation_values = []
    for i in range(len(data)):
        if i >= period:
            sma_result = calculate_SMA(data, period)
            deviations = [abs(x - sma_result[i])
                          for x in data[i - period : i]]
            deviation_values.append(sum(deviations) / period)
        else:
            deviation_values.append(0)
    return deviation_values


def SMA(data, period):
    sma_values = []
    for i in range(len(data)):
        if i < period:
            sma_values.append(0)
        else:
            sma = sum(data[i - period:i]) / period
            sma_values.append(sma)
    return sma_values


def calculate_CMO(close_prices, period):
    """
    Calculate CMO indicator.

    Returns:
    list: CMO values.

    source: https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/cmo#:~:text=The%20CMO%20indicator%20is%20created,%2D100%20to%20%2B100%20range.
    """
    CMO_values = []
    for i in range(period - 1):
        CMO_values.append(float('nan'))

    for i in range(len(close_prices) - period + 1):
        Su = 0  # Sum of up days
        Sd = 0  # Sum of down days

        for j in range(i + 1, i + period):
            price_diff = close_prices[j] - close_prices[j - 1]

            if price_diff > 0:  # Up day
                Su += price_diff
            elif price_diff < 0:  # Down day
                Sd += abs(price_diff)

        if Su + Sd != 0:  # Avoid division by zero
            CMO = 100 * ((Su - Sd) / (Su + Sd))
        else:
            CMO = 0  # If no up or down days, CMO is 0
        CMO_values.append(CMO)

    return CMO_values


def calculate_MACD(prices):
    """
    Calculate MACD indicator.
    Returns: list: MACF values.

    source: https://www.investopedia.com/ask/answers/122414/what-moving-average-convergence-divergence-macd-formula-and-how-it-calculated.asp#:~:text=Moving%20Average%20Convergence%20Divergence%20(MACD)%20is%20calculated%20by%20subtracting%20the,to%20sell)%20its%20signal%20line.
    """
    period_12_EMA = calculate_EMA(prices=prices, period=12)
    period_26_EMA = calculate_EMA(prices=prices, period=26)
    MACD_LINE = [i - j for i, j in zip(period_12_EMA, period_26_EMA)]
    filtered_macd_line = [x for x in MACD_LINE if not math.isnan(x)]
    SIGNAL_LINE = 25 * [float('nan')] + calculate_EMA(filtered_macd_line, 9)
    return [i - j for i, j in zip(MACD_LINE, SIGNAL_LINE)]


def calculate_PPO(prices):
    """
    Calculate MACD indicator.
    Returns: list: MACF values.

    source: paper
    """
    p_12 = calculate_EMA(prices, 12)
    p_26 = calculate_EMA(prices, 26)
    res = [((i - j) / j) * 100 for i, j in zip(p_12, p_26)
           if not math.isnan(i) and not math.isnan(j)]
    return [float('nan')] * 25 + res


def calculate_ROC(prices, period=6):
    """
    calculate ROC

    parameters:
    prices (list)
    period (int)

    returns:
    roc_values (list)
    """
    roc_values = []

    for i in range(period):
        roc_values.append(float('nan'))

    for i in range(period, len(prices)):
        roc = ((prices[i] - prices[i - period]) / prices[i - period]) * 100
        roc_values.append(roc)

    return roc_values


def calculate_CMF(close_prices, high_prices, low_prices, volumes):
    """
    calculate Chaikin money flow indicator (CMFI)
    source: paper
    """
    period_21_CMF = []

    for i in range(21):
        period_21_CMF.append(float('nan'))

    multiplier = [((close - low) - (high - close)) / (high - low)
                  for close, high, low in zip(close_prices, high_prices, low_prices)]
    MFV = [x * y for x, y in zip(multiplier, volumes)]
    
    for i in range(len(MFV)):
        if i >= 21:
            period_21_CMF.append(sum(MFV[i - 21: i]) / sum(volumes[i - 21: i]))
    return period_21_CMF


def calculate_ADX(high, low, close, period=14):
    """
    calculate DMI
    source: https://www.investopedia.com/terms/d/dmi.asp
    """
    TR = [max(high[i] - low[i], abs(high[i] - close[i - 1]),
              abs(low[i] - close[i - 1])) for i in range(1, len(high))]
    plus_DM = [high[i] - high[i - 1] if high[i] - high[i - 1] >
               low[i - 1] - low[i] else 0 for i in range(1, len(high))]
    minus_DM = [low[i - 1] - low[i] if low[i - 1] - low[i] >
                high[i] - high[i - 1] else 0 for i in range(1, len(high))]

    ATR = [sum(TR[:period]) / period]
    plus_DM_smoothed = [sum(plus_DM[:period]) / period]
    minus_DM_smoothed = [sum(minus_DM[:period]) / period]

    for i in range(period, len(TR)):
        ATR.append((ATR[-1] * (period - 1) + TR[i]) / period)
        plus_DM_smoothed.append(
            (plus_DM_smoothed[-1] * (period - 1) + plus_DM[i]) / period)
        minus_DM_smoothed.append(
            (minus_DM_smoothed[-1] * (period - 1) + minus_DM[i]) / period)

    plus_DI = [(plus_DM_smoothed[i] / ATR[i]) * 100 for i in range(len(ATR))]
    minus_DI = [(minus_DM_smoothed[i] / ATR[i]) * 100 for i in range(len(ATR))]

    DX = [abs((plus_DI[i] - minus_DI[i]) / (plus_DI[i] + minus_DI[i]))
          * 100 for i in range(len(ATR))]

    ADX = [sum(DX[:period]) / period]

    for i in range(period, len(DX)):
        ADX.append((ADX[-1] * (period - 1) + DX[i]) / period)

    ADX = [float('nan')] * (len(high) - len(ADX)) + ADX

    return ADX


def calculate_SAR(high_prices, low_prices, acceleration=0.02, maximum=0.2):
    """
    Calculate the Stop and Reverse (SAR) value for a given set of high and low prices.

    Parameters:
    - high_prices: List or array-like object containing the high prices for each period.
    - low_prices: List or array-like object containing the low prices for each period.
    - acceleration: The acceleration factor used in the SAR calculation (default is 0.02).
    - maximum: The maximum value the acceleration factor can reach (default is 0.2).

    Returns:
    - List of SAR values corresponding to each period.

    source: https://www.mystockoptions.com/articles/stock-appreciation-rights-101-part-1
    """
    sar_values = [0] * len(high_prices)
    sar = low_prices[0]  # Initial SAR value

    # Choose the initial trend direction based on the first two periods
    trend = 1 if high_prices[1] > high_prices[0] else -1

    # Initialize acceleration factor and extreme price
    acceleration_factor = acceleration
    extreme_price = high_prices[0] if trend == 1 else low_prices[0]

    for i in range(2, len(high_prices)):
        prev_sar = sar_values[i - 1]
        current_high = high_prices[i]
        current_low = low_prices[i]

        if trend == 1:
            if current_high > extreme_price:
                extreme_price = current_high
                acceleration_factor = min(
                    acceleration_factor + acceleration, maximum)
            sar = prev_sar + acceleration_factor * (extreme_price - prev_sar)
            if current_low < sar_values[i - 2]:
                sar = extreme_price
                trend = -1
                acceleration_factor = acceleration
                extreme_price = current_low
        else:
            if current_low < extreme_price:
                extreme_price = current_low
                acceleration_factor = min(
                    acceleration_factor + acceleration, maximum)
            sar = prev_sar + acceleration_factor * (extreme_price - prev_sar)
            if current_high > sar_values[i - 2]:
                sar = extreme_price
                trend = 1
                acceleration_factor = acceleration
                extreme_price = current_high

        sar_values[i] = sar

    # replace 0 with 'nan'
    for index, element in enumerate(sar_values):
        if element == 0:
            sar_values[index] = float('nan')
        else:
            break

    return sar_values


def calculate_LABELS(prices):
    window_size = 11
    result = []

    for i in range(window_size - 1):
        result.append(float('nan'))

    for counter_row in range(len(prices)):
        min_index = max_index = counter_row
        min_price = max_price = prices[counter_row]

        if counter_row >= window_size - 1:
            window_begin_index = counter_row - window_size
            window_end_index = window_begin_index + window_size - 1
            window_middle_index = (window_begin_index + window_end_index) // 2

            for i in range(window_begin_index, window_end_index + 1):
                number = prices[i]
                if number < min_price:
                    min_price = number
                    min_index = i
                if number > max_price:
                    max_price = number
                    max_index = i

            if max_index == window_middle_index:
                # SELL
                result.append(1)
            elif min_index == window_middle_index:
                # BUY
                result.append(2)
            else:
                # HOLD
                result.append(0)

    return result
