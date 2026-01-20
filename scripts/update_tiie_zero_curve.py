from mainsequence.instruments.interest_rates.etl.registry import DISCOUNT_CURVE_BUILDERS
from mainsequence.instruments.interest_rates.etl.nodes import CurveConfig, DiscountCurvesNode
from src.instruments.rates_curves import build_tiie_valmer


# Register the builder under the Constant *name* (not the resolved UID value).
DISCOUNT_CURVE_BUILDERS.register("ZERO_CURVE__VALMER_TIIE_28", build_tiie_valmer)


def main():
    configs = [
        CurveConfig(
            curve_const="ZERO_CURVE__VALMER_TIIE_28",
            name="Discount Curve TIIE 28 Mexder Valmer",

        ),
    ]

    for cfg in configs:
        node = DiscountCurvesNode(curve_config=cfg)
        node.run(force_update=True)


if __name__ == "__main__":
    main()
