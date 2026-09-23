"""Quy chuẩn định dạng số và phần trăm dùng chung cho toàn bộ dự án."""

import decimal

PERCENT_DECIMALS = 3
VALUE_DECIMALS = 6
MAX_ESCALATION_DECIMALS = 18
MISSING = "—"


def decimal_of(value):
    """Chuyển giá trị sang `Decimal` qua biểu diễn thập phân ổn định."""
    try:
        return decimal.Decimal(repr(float(value)))
    except (TypeError, ValueError, decimal.InvalidOperation):
        return None


def round_without_lying(exact, decimals, upper_bound=None):
    """Tăng số chữ số khi cách làm tròn ban đầu làm mất ý nghĩa giá trị."""
    while decimals <= MAX_ESCALATION_DECIMALS:
        quantum = decimal.Decimal(1).scaleb(-decimals)
        rounded = exact.quantize(quantum, rounding=decimal.ROUND_HALF_UP)
        lies_as_zero = rounded == 0 and exact != 0
        lies_as_bound = (upper_bound is not None
                         and rounded == upper_bound and exact < upper_bound)
        if not (lies_as_zero or lies_as_bound):
            return rounded
        decimals += 1
    return exact


def _comma_decimal(text: str) -> str:
    """Đổi dấu thập phân sang quy ước tiếng Việt."""
    return text.replace(".", ",")


def percent(value, decimals: int = PERCENT_DECIMALS, vietnamese: bool = True,
            unit: str = "%") -> str:
    """Định dạng tỷ lệ trong khoảng [0, 1] thành phần trăm."""
    exact = decimal_of(value)
    if exact is None:
        return MISSING
    rounded = round_without_lying(exact * 100, decimals, upper_bound=100)
    text = f"{rounded.normalize():f}{unit}"
    return _comma_decimal(text) if vietnamese else text


def number(value, unit: str = "", decimals: int = VALUE_DECIMALS,
           vietnamese: bool = True) -> str:
    """Định dạng một số đo và giữ lại phần khác 0 khi cần."""
    exact = decimal_of(value)
    if exact is None:
        return MISSING
    rounded = round_without_lying(exact, decimals)
    text = f"{rounded.normalize():f}{unit}"
    return _comma_decimal(text) if vietnamese else text


def count(value, separator: str = "\u202f") -> str:
    """Định dạng số nguyên với dấu phân nhóm hàng nghìn."""
    if value is None:
        return MISSING
    return f"{int(value):,}".replace(",", separator)
