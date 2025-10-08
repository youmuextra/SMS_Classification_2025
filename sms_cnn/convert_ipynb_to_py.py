import json
import sys
import os
def ipynb_to_py(ipynb_file, py_file=None):
    """
    将Jupyter notebook转换为Python脚本

    Args:
        ipynb_file: 输入的.ipynb文件路径
        py_file: 输出的.py文件路径（可选）
    """

    if py_file is None:
        py_file = ipynb_file.replace('.ipynb', '.py')

    try:
        # 读取ipynb文件
        with open(ipynb_file, 'r', encoding='utf-8') as f:
            notebook = json.load(f)

        # 提取代码单元格
        code_cells = []
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                # 获取代码内容
                source = cell['source']
                code = ''.join(source)

                # 跳过空单元格
                if code.strip():
                    code_cells.append(code)

        # 写入py文件
        with open(py_file, 'w', encoding='utf-8') as f:
            f.write('# 从Jupyter Notebook转换而来\n')
            f.write(f'# 源文件: {os.path.basename(ipynb_file)}\n\n')

            for i, code in enumerate(code_cells, 1):
                f.write(f'# 单元格 {i}\n')
                f.write(code)
                f.write('\n\n')

        print(f"转换成功: {ipynb_file} -> {py_file}")
        return True

    except Exception as e:
        print(f"转换失败: {e}")
        return False


# 使用示例
if __name__ == "__main__":
    ipynb_to_py("word2vec_train.ipynb")
    print("用法: python convert_ipynb_to_py.py <notebook.ipynb>")