import json

with open(r"c:\Userdata\antiiii\kaggle_train_deploy\kaggle_train_pipeline.ipynb", "r") as f:
    nb = json.load(f)

fix_cell = {
  "cell_type": "code",
  "execution_count": None,
  "metadata": {},
  "source": [
    "# FIX WINDOWS PATH IN DATASET YAML\n",
    "import yaml\n",
    "yaml_path = DATASET / \"fics_pcb.yaml\"\n",
    "# Copy it to working directory so we have write access\n",
    "local_yaml = Path(\"/kaggle/working/fics_pcb.yaml\")\n",
    "with open(yaml_path, \"r\") as f:\n",
    "    cfg = yaml.safe_load(f)\n",
    "\n",
    "cfg[\"path\"] = str(DATASET)\n",
    "\n",
    "with open(local_yaml, \"w\") as f:\n",
    "    yaml.dump(cfg, f)\n",
    "\n",
    "# Update DATA_YAML to point to the fixed local copy\n",
    "DATA_YAML = local_yaml\n",
    "print(f\"Fixed YAML path to: {DATA_YAML}\")\n"
  ]
}

# Insert it before step 1
nb["cells"].insert(3, fix_cell)

with open(r"c:\Userdata\antiiii\kaggle_train_deploy\kaggle_train_pipeline.ipynb", "w") as f:
    json.dump(nb, f, indent=2)


