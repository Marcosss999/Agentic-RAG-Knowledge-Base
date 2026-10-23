"""
一键项目安全打包脚本 (Project Packaging Utility) - 02-AI_Agentic_RAG高级企业知识库平台
功能：
1. 严格过滤排除敏感私钥配置文件 (.env)，防止 LLM/向量网关 API Key 意外泄露给他人。
2. 排除 Python 编译缓存文件 (__pycache__、*.pyc)。
3. 保留开箱即用的测试知识库样例文档与规范的 Ragas 评测样例底稿。
4. 打包输出标准 zip 归档至上一级目录，便于直接发送给朋友、导师或技术面试官交流。
"""

import os
import zipfile
from pathlib import Path


def package():
    project_dir = Path(__file__).resolve().parent
    parent_dir = project_dir.parent
    zip_output_path = parent_dir / "02-AI_Agentic_RAG高级企业知识库平台_交付源码.zip"

    print("=" * 60)
    print(f"[*] 正在打包项目: {project_dir.name}")
    print(f"[*] 目标输出文件: {zip_output_path}")
    print("=" * 60)

    file_count = 0
    with zipfile.ZipFile(zip_output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(project_dir):
            # 排除编译缓存和隐藏工作区
            dirs[:] = [d for d in dirs if d not in ["__pycache__", ".git", ".vscode", ".idea"]]
            for file in files:
                file_path = Path(root) / file
                rel_path = file_path.relative_to(project_dir)

                # 严重安全红线：严格排除私密 .env 文件
                if file == ".env":
                    print("  [!] 安全过滤：已自动跳过包含私密密钥的 .env 文件 (保留 .env.example 供配置参考)")
                    continue
                if file.endswith(".pyc"):
                    continue

                # 排除打包脚本自身生成的临时压缩包
                if file.endswith(".zip"):
                    continue

                archive_name = f"{project_dir.name}/{str(rel_path).replace(os.sep, '/')}"
                zf.write(file_path, archive_name)
                file_count += 1

    size_kb = round(zip_output_path.stat().st_size / 1024, 1)
    print("=" * 60)
    print(f"[OK] 打包完成！共归档 {file_count} 个核心文件，压缩包大小: {size_kb} KB")
    print(f"[OK] 压缩包绝对路径: {zip_output_path}")
    print("=" * 60)


if __name__ == "__main__":
    package()
