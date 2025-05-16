from typing import List, Dict
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from app.config import settings
import fitz
import spacy

class DocumentProcessor:
    def __init__(self):
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.index = None
        self.documents: List[Dict] = []

    def process_documents(self, docs_dir: Path) -> None:
        """处理目录下的所有文本和PDF文件"""
        self.documents = []
        nlp = spacy.load("zh_core_web_sm")
        
        for file_path in docs_dir.glob("**/*"):
            if file_path.suffix == ".txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    content = {'text': f.read(), 'images': []}
            elif file_path.suffix == ".pdf":
                content = self._extract_pdf_content(file_path)
            else:
                continue

            # 处理文本内容
            if content['text']:
                text_chunks = self._semantic_split(content['text'], nlp)
                for i, chunk in enumerate(text_chunks):
                    self.documents.append({
                        "text": chunk,
                        "source": str(file_path),
                        "chunk_id": f"text_{i}",
                        "type": "text"
                    })
            
            # 处理图片中提取的文字内容
            for i, img in enumerate(content['images']):
                if img['extracted_text']:
                    self.documents.append({
                        "text": img['extracted_text'],
                        "source": str(file_path),
                        "chunk_id": f"image_{i}",
                        "type": img['type'],
                        "page_num": img['page_num']
                    })

    def _extract_pdf_content(self, pdf_path: Path) -> Dict:
        """提取PDF中的文本和图像内容"""
        doc = fitz.open(pdf_path)
        text_content = []
        image_content = []
        
        for page in doc:
            # 提取文本
            text = page.get_text()
            if text.strip():
                text_content.append(text)
            
            # 提取图像
            for img_index, img in enumerate(page.get_images()):
                xref = img[0]
                base_image = doc.extract_image(xref)
                
                # 处理图片内容
                image_result = self._process_image_content(
                    base_image["image"],
                    base_image["ext"]
                )
                
                # 只保存包含文字的图片信息
                if image_result['has_text']:
                    image_content.append({
                        'page_num': page.number,
                        'image_index': img_index,
                        'extracted_text': image_result['extracted_text'],
                        'type': image_result['type']
                    })
        
        return {
            'text': '\n'.join(text_content),
            'images': image_content
        }
    
    def _split_text(self, text: str) -> List[str]:
        """将文本分割成小块"""
        # 简单实现，后续可以改进分块策略
        words = text.split()
        chunks = []
        for i in range(0, len(words), settings.CHUNK_SIZE):
            chunk = " ".join(words[i:i + settings.CHUNK_SIZE])
            chunks.append(chunk)
        return chunks
    
    def build_index(self) -> None:
        """构建向量索引"""
        if not self.documents:
            raise ValueError("No documents processed yet")
            
        # 获取文档嵌入
        embeddings = self.model.encode([doc["text"] for doc in self.documents])
        
        # 构建FAISS索引
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        
    def save_index(self, path: Path) -> None:
        """保存索引到文件"""
        if self.index is None:
            raise ValueError("No index built yet")
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        
    def load_index(self, path: Path) -> None:
        """从文件加载索引"""
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")
        self.index = faiss.read_index(str(path))
        
    def search(self, query: str, k: int = None) -> List[Dict]:
        """搜索最相关的文档片段"""
        if self.index is None:
            raise ValueError("No index built yet")
            
        k = k or settings.TOP_K
        query_vector = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_vector).astype('float32'), k)
        
        results = []
        for idx in indices[0]:
            if idx != -1:  # FAISS可能返回-1表示无效结果
                results.append(self.documents[idx])
        return results 

    def _process_image_content(self, image_data: bytes, image_ext: str) -> Dict[str, any]:
        """处理图片内容，返回提取的文字和图片类型"""
        import cv2
        import numpy as np
        import pytesseract
        from PIL import Image
        import io

        # 将图片数据转换为OpenCV格式
        image_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        # 检测是否为流程图
        def is_flowchart(img):
            # 使用边缘检测和形状分析来判断是否为流程图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 检查是否存在规则的几何形状（矩形、菱形等）
            for contour in contours:
                approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
                if len(approx) in [4, 6, 8]:  # 流程图常见的形状边数
                    return True
            return False

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
            # 提取流程图中的文字
            text = pytesseract.image_to_string(preprocessed, lang='chi_sim+eng')
            # 可以添加更多流程图专门的处理逻辑
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