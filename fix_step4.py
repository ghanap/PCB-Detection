import json

with open(r"c:\Userdata\antiiii\kaggle_train_deploy\kaggle_train_pipeline.ipynb", "r") as f:
    nb = json.load(f)

step4_cell = {
  "cell_type": "code",
  "execution_count": None,
  "metadata": {},
  "source": [
    "# STEP 4: YOLO -> SAM Inference Pipeline\n",
    "def step4_yolo_then_sam():\n",
    "    print(\"\\n--- STEP 4: YOLO -> SAM INFERENCE PIPELINE ---\")\n",
    "    out_dir = OUT_ROOT / \"step4-yolo-sam-pipeline\"\n",
    "    out_dir.mkdir(parents=True, exist_ok=True)\n",
    "    \n",
    "    yolo_weights = OUT_ROOT / \"step1-yolo-bb\" / \"train\" / \"weights\" / \"best.pt\"\n",
    "    yolo = YOLO(str(yolo_weights))\n",
    "    sam = sam_model_registry[\"vit_b\"](checkpoint=sam_ckpt).to(\"cuda\")\n",
    "    sam.eval()\n",
    "    predictor = SamPredictor(sam)\n",
    "    \n",
    "    img_files = get_image_files(DATASET / \"test\" / \"images\")\n",
    "    lbl_dir = DATASET / \"test\" / \"labels\"\n",
    "    all_ious = []\n",
    "    \n",
    "    for img_p in img_files:\n",
    "        img = cv2.imread(str(img_p))\n",
    "        if img is None: continue\n",
    "        h, w = img.shape[:2]\n",
    "        \n",
    "        results = yolo.predict(str(img_p), conf=0.25, verbose=False)\n",
    "        if not results or len(results[0].boxes) == 0: continue\n",
    "        \n",
    "        predictor.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))\n",
    "        det_boxes = results[0].boxes\n",
    "        \n",
    "        gt_boxes = parse_yolo_labels(lbl_dir / f\"{img_p.stem}.txt\", w, h)\n",
    "        \n",
    "        for j in range(len(det_boxes)):\n",
    "            box = det_boxes.xyxy[j].cpu().numpy().astype(int).tolist()\n",
    "            masks, _, _ = predictor.predict(box=np.array(box)[None, :], multimask_output=False)\n",
    "            \n",
    "            best_gt_iou = 0.0\n",
    "            for _, gt_box in gt_boxes:\n",
    "                gt_mask = np.zeros((h, w), dtype=np.uint8)\n",
    "                gt_mask[gt_box[1]:gt_box[3], gt_box[0]:gt_box[2]] = 1\n",
    "                iou = compute_iou(masks[0], gt_mask)\n",
    "                best_gt_iou = max(best_gt_iou, iou)\n",
    "            all_ious.append(best_gt_iou)\n",
    "            \n",
    "    summary = {\"model\": \"YOLO->SAM pipeline\", \"mean_iou_vs_gt\": float(np.mean(all_ious)) if all_ious else 0.0}\n",
    "    with open(out_dir / \"results.json\", \"w\") as f: json.dump(summary, f)\n",
    "    print(summary)\n",
    "\n",
    "step4_yolo_then_sam()"
  ]
}

# Insert before the last summary cell
nb["cells"].insert(-1, step4_cell)

with open(r"c:\Userdata\antiiii\kaggle_train_deploy\kaggle_train_pipeline.ipynb", "w") as f:
    json.dump(nb, f, indent=2)
