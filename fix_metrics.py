import json

with open(r"c:\Userdata\antiiii\kaggle_train_deploy\kaggle_train_pipeline.ipynb", "r") as f:
    nb = json.load(f)

# Update Step 1 cell to include Precision and Recall
for cell in nb["cells"]:
    if cell["cell_type"] == "code" and "STEP 1: YOLO BB" in "".join(cell["source"]):
        new_source = []
        for line in cell["source"]:
            if "summary =" in line and "YOLO-BB" in line:
                new_source.append('    summary = {"model": "YOLO-BB", "mAP50": float(metrics.box.map50), "mAP50-95": float(metrics.box.map), "Precision": float(metrics.box.mp), "Recall": float(metrics.box.mr)}\n')
            else:
                new_source.append(line)
        cell["source"] = new_source

# Update Step 2 cell
for cell in nb["cells"]:
    if cell["cell_type"] == "code" and "STEP 2: SAM+BB PSEUDO-LABELS" in "".join(cell["source"]):
        new_source = []
        for line in cell["source"]:
            if "summary =" in line and "YOLO-seg (BB)" in line:
                new_source.append('    summary = {"model": "YOLO-seg (BB)", "box_mAP50": float(metrics.box.map50), "mask_mAP50": float(metrics.seg.map50), "Precision": float(metrics.box.mp), "Recall": float(metrics.box.mr)}\n')
            else:
                new_source.append(line)
        cell["source"] = new_source

# Update Step 3 cell
for cell in nb["cells"]:
    if cell["cell_type"] == "code" and "STEP 3: SAM-AUTO PSEUDO-LABELS" in "".join(cell["source"]):
        new_source = []
        for line in cell["source"]:
            if "summary =" in line and "YOLO-seg (Auto)" in line:
                new_source.append('    summary = {"model": "YOLO-seg (Auto)", "box_mAP50": float(metrics.box.map50), "mask_mAP50": float(metrics.seg.map50), "Precision": float(metrics.box.mp), "Recall": float(metrics.box.mr)}\n')
            else:
                new_source.append(line)
        cell["source"] = new_source

# Update final summary cell to output Excel
for cell in nb["cells"]:
    if cell["cell_type"] == "code" and "=== FINAL BENCHMARK ===" in "".join(cell["source"]):
        new_source = []
        for line in cell["source"]:
            if "df.to_csv" in line:
                new_source.append('df.to_csv(OUT_ROOT / "benchmark_summary.csv", index=False)\n')
                new_source.append('df.to_excel(OUT_ROOT / "benchmark_summary.xlsx", index=False)\n')
            else:
                new_source.append(line)
        cell["source"] = new_source

with open(r"c:\Userdata\antiiii\kaggle_train_deploy\kaggle_train_pipeline.ipynb", "w") as f:
    json.dump(nb, f, indent=2)
