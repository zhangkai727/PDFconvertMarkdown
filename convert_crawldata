import os
import re
import json
import time
from glob import glob
from tqdm import tqdm
from marker.convert import convert_single_pdf
from marker.output import markdown_exists, save_markdown
from marker.models import load_all_models

# 设置环境变量
os.environ["HF_DATASETS_CACHE"] = "xxx/xxx/xxx"     #指定了 Hugging Face Datasets 库缓存数据集的位置
os.environ["HF_HOME"] = "xxx/xxx/xxx"               #指定了 Hugging Face 库的主目录
os.environ["HUGGINGFACE_HUB_CACHE"] = "xxx/xxx/xxx" #指定了 Hugging Face Hub 库缓存模型和数据集的位置
os.environ["TRANSFORMERS_CACHE"] = "xxx/xxx/xxx"    #指定了 Hugging Face Transformers 库缓存模型的位置。
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" #指定了 Hugging Face 库访问模型和数据集的服务器地址

# 定义路径
base_path = 'xxx/xxx/xxx'    #选择需要转换数据文件的地址
output_dir = 'xxx/xxx/xxx'    #选择转换完成后保存的地址

# 加载所有模型
model_lst = load_all_models()

# 定义将输入的 Markdown 文本分解为结构化数据的函数
def structure_markdown(markdown_text):
    structured_data = []
    lines = markdown_text.split('\n')

    for line in lines:
        if line.startswith('# '): # 如果行以 '# ' 开头，表示这是一个标题
            structured_data.append({'type': 'title', 'content': line[2:]})
        elif line.startswith('## '):
            structured_data.append({'type': 'subtitle', 'content': line[3:]})
        elif re.match(r'!\[.*\]\(.*\)', line):
            structured_data.append({'type': 'image', 'content': line})
        elif line:
            structured_data.append({'type': 'paragraph', 'content': line})

    return structured_data

def convert(pdf_file_path):
    # 进行pdf到markdown的转化逻辑，调用convert_single_pdf函数来处理单个PDF文件
    full_text, images, out_meta = convert_single_pdf(pdf_file_path, model_lst, max_pages=10, langs=['zh','en'], batch_multiplier=1, start_page=0)
    fname = os.path.basename(pdf_file_path).replace('.pdf', '')  # 从PDF文件路径中提取文件名，去掉.pdf后缀
    markdown_file_path = os.path.join(output_dir, fname + '.md')  # 创建Markdown文件的路径

    # 保存Markdown文件
    with open(markdown_file_path, 'w') as f:
        f.write(full_text)

    # 提取结构化数据
    structured_data = structure_markdown(full_text)

    return markdown_file_path, structured_data  # 返回Markdown文件的路径和结构化数据

def collect_all_target_pdf():
    all_pdf_files = glob(base_path + '/*/*.pdf')  # 使用glob模块搜索指定路径下的所有PDF文件
    markdown_files_info = []  # 初始化一个列表，用于存储转换后的Markdown文件的信息

    for idx in tqdm(range(len(all_pdf_files))):  # 使用tqdm库来显示进度条，循环处理每个PDF文件
        pdf_file = all_pdf_files[idx]
        markdown_file, structured_data = convert(pdf_file)  # 调用convert函数处理每个PDF文件，并获取Markdown文件的路径和结构化数据

        # 创建Markdown文件的信息
        markdown_info = {
            'source_pdf': pdf_file,
            'markdown_file': markdown_file,
            'conversion_time': time.ctime(),
            'structured_data': structured_data
        }

        markdown_files_info.append(markdown_info)  # 将Markdown文件的信息添加到列表中

    # 将所有Markdown文件的信息写入result.json文件
    with open('result.json', 'w') as f:
        json.dump(markdown_files_info, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    collect_all_target_pdf()
