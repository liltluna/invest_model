# Two methods to label the data

import numpy as np


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


def calculate_trichotomous_LABELS(prices, theta=0.2, t_forward=1):
    if len(prices) < t_forward + 1:
        raise ValueError("ERROE: PRICE LENGTH TOO SHORT")

    future_prices = prices[t_forward:]
    log_returns = np.log(future_prices) - np.log(prices[:-t_forward])

    sorted_returns = np.sort(log_returns)[::-1]
    idx_theta = int(len(sorted_returns) * theta)
    idx_one_minus_theta = int(len(sorted_returns) * (1 - theta))
    r_theta = sorted_returns[idx_theta]
    r_one_minus_theta = sorted_returns[idx_one_minus_theta]

    labels = []
    for ret in log_returns:
        if ret > r_theta:
            labels.append(1)
        elif ret < r_one_minus_theta:
            labels.append(-1)
        else:
            labels.append(0)

    return labels
