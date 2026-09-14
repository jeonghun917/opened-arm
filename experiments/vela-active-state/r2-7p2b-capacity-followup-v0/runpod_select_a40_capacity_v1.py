#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

GPU_ID = "NVIDIA A40"
PRICE_CAP = 0.49
MIN_MEMORY_GB = 48
ALLOWED_STOCK = {"High", "Medium"}


def load_rows(path: str):
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("gpus", "items"):
            rows = data.get(key)
            if isinstance(rows, list):
                return rows
    raise SystemExit("GPU_INVENTORY_SHAPE_UNSUPPORTED")


def select_capacity(rows, required_dc=None):
    gpu = next((x for x in rows if isinstance(x, dict) and x.get("gpuId") == GPU_ID), None)
    if gpu is None:
        raise SystemExit("A40_NOT_LISTED")
    if gpu.get("secureCloud") is not True or gpu.get("available") is not True:
        raise SystemExit("A40_NOT_AVAILABLE_SECURE")
    price = gpu.get("securePricePerHr")
    if not isinstance(price, (int, float)) or float(price) > PRICE_CAP:
        raise SystemExit(f"A40_PRICE_CAP_EXCEEDED:{price}")
    if int(gpu.get("memoryInGb") or 0) < MIN_MEMORY_GB:
        raise SystemExit(f"A40_MEMORY_DRIFT:{gpu.get('memoryInGb')}")

    dcs = gpu.get("dataCenterAvailability")
    if not isinstance(dcs, list):
        raise SystemExit("A40_DATACENTER_LIST_MISSING")
    normalized = []
    for dc in dcs:
        if not isinstance(dc, dict):
            continue
        dc_id = dc.get("dataCenterId")
        stock = dc.get("stockStatus")
        if not isinstance(dc_id, str) or not dc_id:
            raise SystemExit("A40_DATACENTER_SCHEMA_DRIFT:dataCenterId")
        normalized.append({"dataCenterId": dc_id, "stockStatus": stock})

    if required_dc:
        chosen = next((dc for dc in normalized if dc["dataCenterId"] == required_dc), None)
        if chosen is None:
            raise SystemExit(f"DATACENTER_NOT_LISTED:{required_dc}")
        if chosen["stockStatus"] not in ALLOWED_STOCK:
            raise SystemExit(f"DATACENTER_STOCK_NOT_SUFFICIENT:{required_dc}={chosen['stockStatus']}")
    else:
        high = sorted((dc for dc in normalized if dc["stockStatus"] == "High"), key=lambda x: x["dataCenterId"])
        medium = sorted((dc for dc in normalized if dc["stockStatus"] == "Medium"), key=lambda x: x["dataCenterId"])
        candidates = high or medium
        if not candidates:
            raise SystemExit("NO_A40_HIGH_OR_MEDIUM_DATACENTER")
        chosen = candidates[0]

    return {
        "status": "PASS",
        "paid_mutation": False,
        "gpuId": GPU_ID,
        "displayName": gpu.get("displayName"),
        "memoryInGb": gpu.get("memoryInGb"),
        "secureCloud": gpu.get("secureCloud"),
        "securePricePerHr": price,
        "selectedDataCenter": chosen,
        "inventoryStockStatus": gpu.get("stockStatus"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inventory")
    ap.add_argument("output")
    ap.add_argument("--require-data-center")
    args = ap.parse_args()
    result = select_capacity(load_rows(args.inventory), args.require_data_center)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
