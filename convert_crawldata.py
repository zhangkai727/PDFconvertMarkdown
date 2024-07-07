import os
import re
import json
import time
from glob import glob
from tqdm import tqdm
from marker.convert import convert_single_pdf
from marker.output import markdown_exists, save_markdown
from marker.models import load_all_models
from PIL import Image
import concurrent.futures
import logging

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


# 定义将 PDF 文件转换为 Markdown 的函数
def convert(pdf_file_path):
    try:
        # 调用 convert_single_pdf 函数处理单个 PDF 文件
        result = convert_single_pdf(pdf_file_path, model_lst, max_pages=10, langs=['zh', 'en'], batch_multiplier=1, start_page=0)
        
        # 检查返回结果是否符合预期
        if not isinstance(result, tuple) or len(result) != 3:
            raise ValueError("convert_single_pdf 返回的结果不是包含三个元素的元组")
        
        full_text, doc_images, out_meta = result
        
        # 检查 doc_images 是否是字典
        if not isinstance(doc_images, dict):
            raise ValueError("doc_images 不是一个字典")
        
        fname = os.path.basename(pdf_file_path).replace('.pdf', '')
        output_folder = os.path.join(output_dir, fname)
        os.makedirs(output_folder, exist_ok=True)  # 确保输出文件夹存在，如果不存在，就创建它
        
        markdown_file_path = os.path.join(output_folder, fname + '.md')
        
        # 保存 Markdown 文件
        with open(markdown_file_path, 'w') as f:
            f.write(full_text)
        
        # 保存图像
        for i, image in enumerate(doc_images.values()):
            image_path = os.path.join(output_folder, f'image_{i}.png')
            
            if isinstance(image, Image.Image):
                image.save(image_path)
            elif isinstance(image, str):
                image_data = base64.b64decode(image)
                with open(image_path, 'wb') as img_f:
                    img_f.write(image_data)
            else:
                raise TypeError("Unsupported image type")
        
        # 保存元数据
        meta_file_path = os.path.join(output_folder, f'{fname}_meta.json')
        with open(meta_file_path, 'w') as meta_f:
            json.dump(out_meta, meta_f, ensure_ascii=False, indent=4)

        # 提取结构化数据
        structured_data = structure_markdown(full_text)

        return {
            'source_pdf': pdf_file_path,
            'markdown_file': markdown_file_path,
            'conversion_time': time.ctime(),
            'structured_data': structured_data,
            'output_folder': output_folder
        }
    except Exception as e:
        logging.error(f"Error processing {pdf_file_path}: {e}")
        return None

# 定义收集所有目标 PDF 文件并进行转换的函数
def collect_all_target_pdf():
    all_pdf_files = glob(base_path + '/*/*.pdf')  # 获取所有 PDF 文件的路径
    markdown_files_info = []

    # 使用 ThreadPoolExecutor 进行并行处理
    max_workers = 2  # 设置最大线程数为2
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(convert, pdf_file): pdf_file for pdf_file in all_pdf_files}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(all_pdf_files)):
            try:
                result = future.result()
                if result is not None:
                    markdown_files_info.append(result)
            except Exception as e:
                logging.error(f"Error in future result: {e}")

    # 将所有 Markdown 文件的信息写入 result.json 文件
    with open('result.json', 'w') as f:
        json.dump(markdown_files_info, f, ensure_ascii=False, indent=4)

# 主函数，执行收集和转换过程
if __name__ == '__main__':
    collect_all_target_pdf()
