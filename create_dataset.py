import os
import json
from glob import glob
from tqdm import tqdm
import re

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    s = s.replace('"', '').replace("'", "")
    s = s.replace("<", "〈").replace(">", "〉")
    s = re.sub(r"\s+", " ", s)
    s = s.replace("\n", " ").replace("\r", "")
    return s

def convert_json(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    json_files = glob(os.path.join(input_dir, "*.json"))
    count = 0
    seen = set()

    for json_path in tqdm(json_files, desc=f"Converting {input_dir}"):
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except:
                continue

        anns = data.get("learning_data_info", {}).get("annotation", [])
        if not anns:
            continue

        image_id = data.get("source_data_info", {}).get("source_data_name_jpg", None)
        if image_id is None:
            image_id = os.path.basename(json_path).replace(".json", ".jpg")

        for ann in anns:
            class_name = ann.get("class_name", "").strip().lower()
            query = ann.get("visual_instruction", "")
            bbox = ann.get("bounding_box", None)

            class_name = clean_text(class_name)
            query = clean_text(query)

            if not bbox or len(bbox) != 4:
                continue

            if class_name not in ["표", "차트", "그래프", "표/차트", "시각요소"]:
                continue

            if not query:
                continue

            key = (image_id, query)
            if key in seen:
                continue
            seen.add(key)

            output = {
                "image_id": image_id,
                "query": query,
                "class_name": class_name,
                "bbox": bbox
            }

            out_name = os.path.basename(json_path).replace(".json", f"_{count}.json")
            with open(os.path.join(output_dir, out_name), "w", encoding="utf-8") as wf:
                json.dump(output, wf, ensure_ascii=False, indent=2)

            count += 1

def main():
    base = "dataset"

    convert_json(
        input_dir=os.path.join(base, "train_json"),
        output_dir=os.path.join(base, "train_json_out")
    )

    convert_json(
        input_dir=os.path.join(base, "val_json"),
        output_dir=os.path.join(base, "val_json_out")
    )

if __name__ == "__main__":
    main()
