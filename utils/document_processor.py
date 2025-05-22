from typing import List, Dict
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from config import settings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import cv2
import pytesseract
from docx import Document
class DocumentProcessor:
    def __init__(self):
        self.model = SentenceTransformer(settings.embedding_model)
        self.index = None
        self.documents: List[Dict] = []
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            #chunk_overlap=settings.chunk_overlap,  # 可以根据需要调整重叠大小
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )

    def process_documents(self, docs_dir: Path) -> None:
        """处理目录下的所有文本、PDF和Word文件"""
        self.documents = []

        for file_path in docs_dir.glob("**/*"):
            try:
                print(f"开始处理文件: {file_path}")
                docs = []
                image_contents = []

                # 加载文档内容
                if file_path.suffix.lower() == ".txt":
                    loader = TextLoader(str(file_path), encoding='utf-8')
                    docs = loader.load()
                elif file_path.suffix.lower() == ".pdf":
                    # 使用PyPDFLoader处理文本内容
                    loader = PyPDFLoader(str(file_path))
                    docs = loader.load()
                    # 额外处理PDF中的图片
                    image_contents = self._extract_images_from_pdf(file_path)
                elif file_path.suffix.lower() in [".docx", ".doc"]:
                    if file_path.suffix.lower() == ".doc":
                        print(f"Warning: Old .doc format not fully supported: {file_path}")
                        continue
                    # 使用UnstructuredWordDocumentLoader处理文本内容
                    loader = UnstructuredWordDocumentLoader(str(file_path))
                    docs = loader.load()
                    # 额外处理Word文档中的图片
                    image_contents = self._extract_images_from_docx(file_path)
                else:
                    continue

                # 处理文本内容
                for doc in docs:
                    splits = self.text_splitter.split_text(doc.page_content)
                    for i, text in enumerate(splits):
                        if text.strip():
                            self.documents.append({
                                "text": text,
                                "source": file_path.name,
                                "chunk_id": f"text_{i}",
                                "type": "text",
                                "page_num": getattr(doc, 'metadata', {}).get('page', 1)
                            })

                # 处理图片内容
                for img_content in image_contents:
                    if img_content['extracted_text'].strip():
                        self.documents.append({
                            "text": img_content['extracted_text'],
                            "source": file_path.name,
                            "chunk_id": f"image_{img_content['image_index']}",
                            "type": img_content['type'],
                            "page_num": img_content['page_num']
                        })

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    @classmethod
    def _process_image_content(cls, image_data: bytes, image_ext: str) -> Dict[str, any]:
        """处理图片内容，返回提取的文字和图片类型"""
        # 将图片数据转换为OpenCV格式
        image_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        # 检测是否为流程图
        def is_flowchart(img):
            # 使用边缘检测和形状分析来判断是否为流程图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            flowchart_shapes = 0
            total_shapes = 0

            for contour in contours:
                if cv2.contourArea(contour) < 500:  # 忽略太小的形状
                    continue
                total_shapes += 1
                approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
                if len(approx) in [4, 6, 8]:
                    flowchart_shapes += 1

            # 如果流程图形状占比超过30%且总形状数大于3才认为是流程图
            return total_shapes > 3 and (flowchart_shapes / total_shapes) > 0.3

        # 图片预处理
        def preprocess_image(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            denoised = cv2.fastNlMeansDenoising(gray)
            _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary

        # 检测图片是否包含文字
        def has_text(img):
            preprocessed = preprocess_image(img)
            text = pytesseract.image_to_string(preprocessed, lang='chi_sim+eng')
            return len(text.strip()) > 0

        # 处理流程图
        def process_flowchart(img):
            preprocessed = preprocess_image(img)
            text = pytesseract.image_to_string(preprocessed, lang='chi_sim+eng')
            return text.strip()

        # 处理普通图片中的文字
        def process_regular_image(img):
            preprocessed = preprocess_image(img)
            text = pytesseract.image_to_string(preprocessed, lang='chi_sim+eng')
            return text.strip()

        result = {
            'has_text': False,
            'is_flowchart': False,
            'extracted_text': '',
            'type': 'image'
        }

        # 判断图片类型并处理
        if is_flowchart(img):
            result['is_flowchart'] = True
            result['type'] = 'flowchart'
            extracted_text = process_flowchart(img)
            if extracted_text:
                result['has_text'] = True
                result['extracted_text'] = extracted_text
        elif has_text(img):
            result['has_text'] = True
            result['extracted_text'] = process_regular_image(img)

        return result

    def _extract_images_from_pdf(self, pdf_path: Path) -> List[Dict]:
        """从PDF中提取图片内容"""
        import fitz  # 仅在需要时导入PyMuPDF
        image_contents = []
        doc = fitz.open(pdf_path)
        
        for page_num, page in enumerate(doc):
            for img_index, img in enumerate(page.get_images()):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    
                    # 处理图片内容
                    image_result = self._process_image_content(
                        base_image["image"],
                        base_image["ext"]
                    )
                    
                    # 只保存包含文字的图片信息
                    if image_result['has_text']:
                        image_contents.append({
                            'page_num': page_num + 1,
                            'image_index': img_index,
                            'extracted_text': image_result['extracted_text'],
                            'type': image_result['type']
                        })
                except Exception as e:
                    print(f"Error processing image in PDF: {e}")
                    
        return image_contents

    def _extract_images_from_docx(self, docx_path: Path) -> List[Dict]:
        """从Word文档中提取图片内容"""
        doc = Document(docx_path)
        image_contents = []
        image_index = 0

        # 处理文档主体中的图片
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                try:
                    image_data = rel.target_part.blob
                    # 处理图片内容
                    image_result = self._process_image_content(
                        image_data,
                        rel.target_ref.split('.')[-1]
                    )
                    
                    if image_result['has_text']:
                        image_contents.append({
                            'page_num': 1,  # Word文档默认页码
                            'image_index': image_index,
                            'extracted_text': image_result['extracted_text'],
                            'type': image_result['type']
                        })
                    image_index += 1
                except Exception as e:
                    print(f"Error processing image in Word document: {e}")


        return image_contents


    def build_index(self) -> None:
        """构建向量索引，使用余弦相似度"""
        print("开始构建向量索引...")
        if not self.documents:
            raise ValueError("No documents processed yet")
            
        # 获取文档嵌入
        embeddings = self.model.encode([doc["text"] for doc in self.documents])
        
        # 归一化向量
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
        
        # 构建FAISS索引 - 使用内积（等价于余弦相似度，因为向量已归一化）
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        print("构建向量索引完成")

    def save_index(self, path: Path) -> None:
        """保存索引和文档数据到文件"""
        if self.index is None:
            raise ValueError("No index built yet")
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存FAISS索引
        faiss.write_index(self.index, str(path))
        
        # 保存documents数据到同目录下的json文件
        import json
        docs_path = path.parent / "documents.json"
        with open(docs_path, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
        
        print(f"向量索引已保存到: {path}")
        print(f"文档数据已保存到: {docs_path}")

    def load_index(self, path: Path) -> None:
        """从文件加载索引和文档数据"""
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")
            
        # 加载FAISS索引
        self.index = faiss.read_index(str(path))
        
        # 加载documents数据
        import json
        docs_path = path.parent / "documents.json"
        if not docs_path.exists():
            raise FileNotFoundError(f"Documents data not found: {docs_path}")
            
        with open(docs_path, 'r', encoding='utf-8') as f:
            self.documents = json.load(f)
            
        print(f"已加载向量索引，包含 {len(self.documents)} 个文档片段")

    def search(self, query: str, k: int = None) -> List[Dict]:
        """搜索最相关的文档片段"""
        if self.index is None:
            raise ValueError("No index built yet")
        k=min(k or settings.top_k, self.index.ntotal)
        # 编码并归一化查询向量
        query_vector = self.model.encode([query])
        query_vector = query_vector / np.linalg.norm(query_vector)
        
        # 使用内积搜索，得到的分数越高表示越相似
        scores, indices = self.index.search(np.array(query_vector).astype('float32'), k)
        
        results = []
        if len(indices) > 0:
            for idx, score in zip(indices[0], scores[0]):
                if idx != -1:
                    doc = self.documents[idx].copy()
                    doc['similarity_score'] = float(score)
                    results.append(doc)
        return results


