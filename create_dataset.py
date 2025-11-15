import os
import json
from glob import glob
from tqdm import tqdm

def convert_json(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    json_files = glob(os.path.join(input_dir, "*.json"))
    count = 0

    for json_path in tqdm(json_files, desc=f"Converting {input_dir}"):
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except:
                continue

        # 🔥 핵심: 기존 JSON 구조는 learning_data_info.annotation 안에 annotation 배열이 있다
        anns = data.get("learning_data_info", {}).get("annotation", [])
        if not anns:
            continue

        # 이미지 파일명 추출
        image_id = data.get("source_data_info", {}).get("source_data_name_jpg", None)
        if image_id is None:
            # json 파일명 기준으로 추정
            image_id = os.path.basename(json_path).replace(".json", ".jpg")

        for ann in anns:
            class_name = ann.get("class_name", "").strip().lower()
            bbox = ann.get("bounding_box", None)
            query = ann.get("visual_instruction", "").strip()

            # 관심 있는 클래스만
            if class_name not in ["표", "차트", "그래프", "표/차트", "시각요소"]:
                continue

            if not query:
                continue

            if not bbox or len(bbox) != 4:
                continue

            # 출력 json 구조
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
