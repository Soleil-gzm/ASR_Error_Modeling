import os
import subprocess
import argparse
from pathlib import Path

def get_git_tracked_files(project_path):
    """返回 Git 跟踪的文件列表（相对于项目根目录）"""
    try:
        result = subprocess.run(
            ['git', 'ls-files'],
            cwd=project_path,
            capture_output=True,
            text=True,
            check=True
        )
        files = result.stdout.strip().splitlines()
        return [f for f in files if f]
    except subprocess.CalledProcessError:
        print("错误：当前目录不是 Git 仓库或 git 命令不可用")
        return []

def get_all_files(project_path, exclude_dirs=None):
    """
    返回项目下所有文件的相对路径（递归，跳过指定目录）
    默认跳过 .git 目录，可通过 exclude_dirs 额外指定
    """
    if exclude_dirs is None:
        exclude_dirs = {'.git'}
    project_path = Path(project_path).resolve()
    all_files = []
    for root, dirs, files in os.walk(project_path):
        # 修改 dirs 列表以原地跳过排除的目录
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for file in files:
            full_path = Path(root) / file
            rel_path = full_path.relative_to(project_path)
            all_files.append(str(rel_path))
    return all_files

def build_tree(files):
    """从文件列表构建树形结构字符串"""
    tree = {}
    for file_path in files:
        parts = file_path.split('/')
        current = tree
        for part in parts:
            if part not in current:
                current[part] = {}
            current = current[part]
    
    lines = []
    def walk(node, prefix=''):
        items = list(node.items())
        for i, (name, subnode) in enumerate(items):
            is_last = (i == len(items) - 1)
            connector = '└── ' if is_last else '├── '
            lines.append(f"{prefix}{connector}{name}")
            if subnode:
                extension = '    ' if is_last else '│   '
                walk(subnode, prefix + extension)
    walk(tree)
    return lines

def main():
    parser = argparse.ArgumentParser(description='将项目目录树写入 README.md')
    parser.add_argument('--mode', choices=['git', 'all'], default='git',
                        help='选择目录树模式：git（仅Git追踪的文件）或 all（所有文件，跳过.git等目录）')
    parser.add_argument('--path', default=os.getcwd(),
                        help='项目根目录路径（默认当前工作目录）')
    parser.add_argument('--exclude', nargs='*', default=['.git'],
                        help='在 all 模式下额外排除的目录名（默认已排除 .git）')
    args = parser.parse_args()

    project_path = os.path.abspath(args.path)
    readme_path = os.path.join(project_path, "README.md")

    if not os.path.exists(readme_path):
        print(f"警告：{readme_path} 不存在，将创建新文件")

    if args.mode == 'git':
        files = get_git_tracked_files(project_path)
        title = "项目结构（Git 跟踪）"
        if not files:
            print("未获取到 Git 跟踪的文件")
            return
    else:  # all
        exclude_dirs = set(args.exclude)
        files = get_all_files(project_path, exclude_dirs)
        title = "项目结构（所有文件）"
        if not files:
            print("未获取到任何文件")
            return

    print(f"获取到 {len(files)} 个文件（模式: {args.mode}）")

    tree_lines = build_tree(files)
    structure_text = "\n".join(tree_lines)
    print("生成的目录树预览（前10行）：")
    for line in tree_lines[:10]:
        print(line)

    # 追加到 README.md（避免重复追加相同标题）
    with open(readme_path, 'a', encoding='utf-8') as f:
        f.write(f"\n\n## {title}\n\n```\n")
        f.write(structure_text)
        f.write("\n```\n")

    print(f"✅ 已追加到 {readme_path}")

if __name__ == "__main__":
    main()

'''
仅 Git 追踪的文件（默认）：     python generate_structure.py --mode git

全部文件（跳过 .git 等目录）：  python generate_structure.py --mode all

指定项目路径：                python generate_structure.py --mode all --path /path/to/project

在 all 模式下额外排除目录（例如排除 __pycache__ 和 node_modules）：

python generate_structure.py --mode all --exclude .git __pycache__ node_modules

'''