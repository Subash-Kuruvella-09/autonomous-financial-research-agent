"""
Calculation Engine Tool

Performs common financial calculations: growth rates, ratios,
and basic discounted cash flow (DCF) valuation.
"""


def _calc_growth_rate(inputs: dict) -> dict:
    """Calculate growth rate between two values."""
    initial = inputs.get("initial")
    final = inputs.get("final")

    if initial is None or final is None:
        return {"error": "Missing 'initial' and/or 'final' in inputs."}
    if initial == 0:
        return {"error": "Initial value cannot be zero."}

    rate = (final - initial) / abs(initial)

    return {
        "type": "growth_rate",
        "result": round(rate, 6),
        "result_pct": f"{round(rate * 100, 2)}%",
        "steps": [
            f"Initial value: {initial}",
            f"Final value: {final}",
            f"Growth = ({final} - {initial}) / |{initial}|",
            f"Growth = {round(rate, 6)} ({round(rate * 100, 2)}%)",
        ],
    }


def _calc_ratio(inputs: dict) -> dict:
    """Calculate a simple ratio."""
    numerator = inputs.get("numerator")
    denominator = inputs.get("denominator")

    if numerator is None or denominator is None:
        return {"error": "Missing 'numerator' and/or 'denominator' in inputs."}
    if denominator == 0:
        return {"error": "Denominator cannot be zero."}

    ratio = numerator / denominator

    return {
        "type": "ratio",
        "result": round(ratio, 6),
        "steps": [
            f"Numerator: {numerator}",
            f"Denominator: {denominator}",
            f"Ratio = {numerator} / {denominator}",
            f"Ratio = {round(ratio, 6)}",
        ],
    }


def _calc_dcf(inputs: dict) -> dict:
    """
    Basic Discounted Cash Flow calculation.

    Inputs:
        cash_flows:    List of projected future cash flows.
        discount_rate: Annual discount rate (e.g. 0.10 for 10%).
        terminal_growth_rate: (optional) growth rate for terminal value.
    """
    cash_flows = inputs.get("cash_flows")
    discount_rate = inputs.get("discount_rate")
    terminal_growth = inputs.get("terminal_growth_rate", 0.02)

    if not cash_flows or not isinstance(cash_flows, list):
        return {"error": "Missing or invalid 'cash_flows' list in inputs."}
    if discount_rate is None:
        return {"error": "Missing 'discount_rate' in inputs."}
    if discount_rate <= 0:
        return {"error": "Discount rate must be positive."}
    if discount_rate <= terminal_growth:
        return {"error": "Discount rate must be greater than terminal growth rate."}

    steps = []
    pv_total = 0.0

    # Discount each cash flow
    for i, cf in enumerate(cash_flows, 1):
        pv = cf / ((1 + discount_rate) ** i)
        pv_total += pv
        steps.append(
            f"Year {i}: CF = {cf:,.2f}, PV = {cf:,.2f} / (1+{discount_rate})^{i} = {pv:,.2f}"
        )

    # Terminal value (Gordon Growth Model)
    last_cf = cash_flows[-1]
    n = len(cash_flows)
    terminal_value = (last_cf * (1 + terminal_growth)) / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / ((1 + discount_rate) ** n)
    pv_total += pv_terminal

    steps.append(f"Terminal Value = {last_cf:,.2f} × (1+{terminal_growth}) / ({discount_rate}-{terminal_growth}) = {terminal_value:,.2f}")
    steps.append(f"PV of Terminal = {terminal_value:,.2f} / (1+{discount_rate})^{n} = {pv_terminal:,.2f}")
    steps.append(f"Total DCF Value = {pv_total:,.2f}")

    return {
        "type": "dcf",
        "result": round(pv_total, 2),
        "terminal_value": round(terminal_value, 2),
        "pv_terminal": round(pv_terminal, 2),
        "steps": steps,
    }


def _calc_cagr(inputs: dict) -> dict:
    """Compound Annual Growth Rate."""
    initial = inputs.get("initial")
    final = inputs.get("final")
    years = inputs.get("years")

    if initial is None or final is None or years is None:
        return {"error": "Missing 'initial', 'final', and/or 'years' in inputs."}
    if initial <= 0 or final <= 0:
        return {"error": "Initial and final values must be positive for CAGR."}
    if years <= 0:
        return {"error": "Years must be positive."}

    cagr = (final / initial) ** (1 / years) - 1

    return {
        "type": "cagr",
        "result": round(cagr, 6),
        "result_pct": f"{round(cagr * 100, 2)}%",
        "steps": [
            f"Initial value: {initial}",
            f"Final value: {final}",
            f"Period: {years} years",
            f"CAGR = ({final}/{initial})^(1/{years}) - 1",
            f"CAGR = {round(cagr, 6)} ({round(cagr * 100, 2)}%)",
        ],
    }


# ─── Calculation Router ──────────────────────────────────────────────────────

CALCULATION_TYPES = {
    "growth_rate": _calc_growth_rate,
    "ratio": _calc_ratio,
    "dcf": _calc_dcf,
    "cagr": _calc_cagr,
}


def calculation_engine(calculation_type: str, inputs: dict) -> dict:
    """
    Perform financial calculations.

    Args:
        calculation_type: One of "growth_rate", "ratio", "dcf", "cagr".
        inputs:           Dict of input values specific to the calculation type.

    Returns:
        Dict with result, steps, and type, or {"error": str}.
    """
    try:
        calc_type = calculation_type.lower().strip()

        if calc_type not in CALCULATION_TYPES:
            return {
                "error": (
                    f"Unknown calculation type '{calc_type}'. "
                    f"Supported: {', '.join(sorted(CALCULATION_TYPES.keys()))}"
                )
            }

        return CALCULATION_TYPES[calc_type](inputs)

    except Exception as e:
        return {"error": f"Calculation failed: {e}"}
