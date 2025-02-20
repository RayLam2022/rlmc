import argparse
import os
import os.path as osp
import sqlite3
import hashlib
from typing import List

from PIL import ImageTk, Image
import numpy as np
import torch
import cv2
from tqdm import tqdm
import clip
import faiss

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parser = argparse.ArgumentParser("image_search_image")
parser.add_argument(
    "-r",
    "--img_root",
    required=True,
)
parser.add_argument(
    "-m",
    "--reference_image_for_matching",
    required=True,
)
parser.add_argument(
    "-a",
    "--alternative",  # 备选结果数
    default=5,
)

parser.add_argument(
    "-f",
    "--is_rebuild_faiss_index",
    action="store_true",
    default=False,
)


args = parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
# https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt


def generate_id(string: str):
    sha256 = hashlib.sha256()
    sha256.update(string.encode("utf-8"))
    img_id = int.from_bytes(sha256.digest()[:6], byteorder="big")
    return img_id


def clipv(root: str):
    dimension = 512
    idx_flat = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIDMap(idx_flat)

    faiss_idx_file = osp.join(root, "faissidx.idx")
    db = osp.join(root, "imgsnames.db")
    img_paths = find_files_with_ext(root)

    if osp.exists(faiss_idx_file):
        index = faiss.read_index(faiss_idx_file)
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS images (id INTEGER PRIMARY KEY, name TEXT)"""
    )
    conn.commit()

    exist_ids = set(row[0] for row in cursor.execute("SELECT id FROM images"))
    collector = []
    for img_path in img_paths:
        image_id = generate_id(img_path)
        if image_id not in exist_ids:
            collector.append(img_path)

    counter = 1
    for img_path in collector:

        img_id = generate_id(img_path)
        img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            img_feat = model.encode_image(img)
        img_feat_np = img_feat.cpu().numpy()
        norms = np.linalg.norm(img_feat_np, axis=1, keepdims=True)
        img_feat_np /= norms
        index.add_with_ids(img_feat_np, np.array([img_id]))
        try:
            cursor.execute(
                "INSERT INTO images (id, name) VALUES (?, ?)", (img_id, img_path)
            )
        except Exception as e:
            print(f"Error {img_path}: {str(e)}")
        if counter % 200 == 0:
            conn.commit()
            faiss.write_index(index, faiss_idx_file)
        counter += 1

    conn.commit()
    faiss.write_index(index, faiss_idx_file)
    print("clip vectorized." + "\n")


def search_vec(ref_image: str, root: str, alternative: int = 5):

    img_id = generate_id(ref_image)
    img = preprocess(Image.open(ref_image)).unsqueeze(0).to(device)
    with torch.no_grad():
        img_feat = model.encode_image(img)
    img_feat_np = img_feat.cpu().numpy()
    norms = np.linalg.norm(img_feat_np, axis=1, keepdims=True)
    img_feat_np /= norms

    faiss_index_file = os.path.join(root, "faissidx.idx")
    index = faiss.read_index(faiss_index_file)
    norm = np.linalg.norm(img_feat_np, axis=1, keepdims=True)
    img_feat_np /= norm

    db = os.path.join(root, "imgsnames.db")
    D, I = index.search(img_feat_np, alternative)
    img_paths = []
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    for i, idx in enumerate(I[0]):
        img_id = idx
        cursor.execute("SELECT name FROM images WHERE id = ?", (str(img_id),))
        res = cursor.fetchone()
        if res:
            img_name = res[0]
            img_paths.append(str(img_name))
        else:
            print(f"{i+1}: Image not found(ID: {img_id}, Distance: {D[0][i]})")
    conn.close()
    print(f"Results: {img_paths}")


def find_files_with_ext(
    folder_root: str,
    extensions: List[str] = [
        ".PNG",
        ".JPG",
        ".JPEG",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
    ],
):
    file_list = []

    for root, dirs, files in os.walk(folder_root):
        for file in files:
            if osp.splitext(file)[-1] in extensions:
                file_list.append(os.path.join(root, file))

    return file_list


def main() -> None:
    if args.is_rebuild_faiss_index:
        try:
            os.remove(args.img_root + "/faissidx.idx")
        except:
            pass
        try:
            os.remove(args.img_root + "/imgsnames.db")
        except:
            pass
        clipv(args.img_root)
    else:
        if not osp.exists(args.img_root + "/faissidx.idx") or not osp.exists(
            args.img_root + "/imgsnames.db"
        ):
            clipv(args.img_root)
        else:
            ...
    search_vec(args.reference_image_for_matching, args.img_root, args.alternative)


if __name__ == "__main__":
    main()
