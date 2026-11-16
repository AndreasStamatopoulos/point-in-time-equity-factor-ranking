from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FEATURES = (
    "momentum_12_1",
    "book_to_market",
    "operating_profitability",
    "asset_growth",
    "fcf_yield",
)

START_DATE = "2010-01-01"

SEC_CONCEPT_UNITS = {
    "StockholdersEquity": "USD",
    "Assets": "USD",
    "EntityCommonStockSharesOutstanding": "shares",
    "CommonStockSharesOutstanding": "shares",
    "NetCashProvidedByUsedInOperatingActivities": "USD",
    "PaymentsToAcquirePropertyPlantAndEquipment": "USD",
    "OperatingIncomeLoss": "USD",
}
