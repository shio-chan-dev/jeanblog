import os
import datetime

SRC_DIR = "old_markdown"
DEST_DIR = "content/posts"

def ensure_dest_dir():
    os.makedirs(DEST_DIR, exist_ok=True)

def safe_yaml_string(s: str) -> str:
    """
    将字符串安全地用于 YAML 双引号字符串：
    - 内部 " 替换为 \"
    - 内部反斜杠 \\ 也要转义
    """
    s = s.replace("\\", "\\\\")
    s = s.replace("\"", "\\\"")
    return s

def extract_title(filename):
    name = os.path.splitext(filename)[0]
    title = name.replace("_", " ").replace("-", " ")
    return safe_yaml_string(title)

def build_front_matter(title):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    return f"""---
title: "{title}"
date: {today}
draft: false
---
"""

def import_markdown_files():
    ensure_dest_dir()

    for root, dirs, files in os.walk(SRC_DIR):
        for file in files:
            if file.lower().endswith(".md"):
                src_path = os.path.join(root, file)
                dest_path = os.path.join(DEST_DIR, file)

                with open(src_path, "r", encoding="utf-8") as f:
                    body = f.read()

                title = extract_title(file)
                front_matter = build_front_matter(title)

                with open(dest_path, "w", encoding="utf-8") as f:
                    f.write(front_matter + "\n" + body)

                print(f"Imported: {src_path} → {dest_path}")

if __name__ == "__main__":
    import_markdown_files()

