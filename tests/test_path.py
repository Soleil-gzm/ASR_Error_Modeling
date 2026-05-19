from pathlib import Path

config = {
    'paths': {
        'output': {
            'base_dir': './results',
            'temp_dir': './temp'
        }
    }
}

base_dir = Path(config['paths']['output']['base_dir'])
print(type(base_dir))   # <class 'pathlib.PosixPath'> (或 WindowsPath)
print(base_dir)         # 输出：./results
new_path = base_dir / "subfolder" / "file.txt"
print(new_path)   # ./results/subfolder/file.txt

# 创建完整子目录
sub_dir = base_dir / "images" / "2024"
print(sub_dir)
sub_dir.mkdir(parents=True, exist_ok=True)
print(sub_dir)
# 生成文件路径
file_path = sub_dir / "result.png"
file_path.write_text("some content")   # 举例
print(file_path)