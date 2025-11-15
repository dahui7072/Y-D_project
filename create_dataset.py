import os
import json
from glob import glob
from tqdm import tqdm
import re


# 🔥 텍스트 정제 함수 (query, class_name에 공통 적용)
def clean_text(s: str) -> str:
    if not s:
        return ""

    # 양쪽 공백 제거
    s = s.strip()

    # 따옴표 제거
    s = s.replace('"', '').replace("'", "")

    # 꺾쇠 → 안전한 문자
    s = s.replace("<", "〈").replace(">", "〉")

    # 중복 공백 제거
    s = re.sub(r"\s+", " ", s)

    # 기타 CSV 깨지는 특수문자 제거
    s = s.replace("\n", " ").replace("\r", "")

    return s


def convert_json(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    json_files = glob(os.path.join(input_dir, "*.json"))
    count = 0

    # 🔥 중복 제거용
    seen = set()   # (image_id, query) 조합


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

            # 🔥 전처리
            class_name = clean_text(class_name)
            query = clean_text(query)

            # 정상적인 bounding box인지 검사
            if not bbox or len(bbox) != 4:
                continue

            # 관심있는 클래스만 선택 (전처리 후 기준)
            if class_name not in ["표", "차트", "그래프", "표/차트", "시각요소"]:
                continue

            if not query:
                continue

            # 🔥 중복 제거 (image_id + query = 고유 조합)
            key = (image_id, query)
            if key in seen:
                continue

            seen.add(key)

            # 출력
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
