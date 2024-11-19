import os
from typing import List, Optional, Dict, Any
import pdf2image
from PIL import Image
import numpy as np
from tqdm import tqdm
from quipucamayoc.aws_extract_tables import aws_extract_tables
from pathlib import Path

class EconTableExtractor:
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, 
                 file_path: str,
                 layout_detector: str = 'yolo',
                 ocr_model: str = 'AWS'):
        """
        Initialize the table extractor using singleton pattern
        """
        if not self._initialized:
            self.layout_detector = layout_detector
            self.layout_config = self._get_default_config()
            self.ocr_model = ocr_model
            self.file_path = file_path
            self.pdf_output_dir = f"output\{Path(file_path).stem}"
            self.initialize_layout_detector()
            self.__class__._initialized = True
        else:
            # 只更新文件路径相关的属性
            self.file_path = file_path
            self.pdf_output_dir = f"output\{Path(file_path).stem}"

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        if self.layout_detector == 'yolo':
            return {
                "img_size": 1024,
                "conf_thres": 0.25,
                "iou_thres": 0.45,
                "model_path": "pdf-extract-kit-1.0/models/Layout/YOLO/yolov10l_ft.pt",
                "visualize": True
            }
        else:  # layoutlmv3
            return {
                "model_path": "pdf-extract-kit-1.0/models/Layout/LayoutLMv3/model_final.pth"
            }

    def initialize_layout_detector(self):
        """Initialize the layout detector"""
        # 延迟导入，只在第一次初始化时导入
        from pdf_extract_kit.tasks.layout_detection.models.yolo import LayoutDetectionYOLO
        from pdf_extract_kit.tasks.layout_detection.models.layoutlmv3 import LayoutDetectionLayoutlmv3
        from pdf_extract_kit.tasks.layout_detection.task import LayoutDetectionTask
        
        if self.layout_detector == 'yolo':
            self.detector = LayoutDetectionYOLO(self.layout_config)
        elif self.layout_detector == 'layoutlmv3':
            model = LayoutDetectionLayoutlmv3(self.layout_config)
            self.detector = LayoutDetectionTask(model)
        else:
            raise ValueError(f"Invalid layout detector: {self.layout_detector}")

    def pdf_to_images(self, 
                     pdf_path: str, 
                     output_dir: str,
                     dpi: int = 200) -> List[Image.Image]:
        """
        Transform the PDF to images.

        Args:
            pdf_path (str): The path to the PDF file.
            output_dir (str, optional): The directory to save the images.
            dpi (int): The DPI of the images.

        Returns:
            List[Image.Image]: 图像列表
        """
        images = pdf2image.convert_from_path(pdf_path, dpi=dpi)

        image_path = []
        for i, image in tqdm(enumerate(images), total=len(images), desc="Converting PDF to images"):
            file_output_dir = os.path.join(output_dir, f'page_{i+1}')
            if not os.path.exists(file_output_dir):
                os.makedirs(file_output_dir) 
            image.save(os.path.join(file_output_dir,f'page_{i+1}.jpg'))
            image_path.append(os.path.join(file_output_dir, f'page_{i+1}.jpg'))
        
        return images,image_path

    def detect_tables(self, 
                     images: List[Image.Image]) -> List[Dict]:
        """
        Detect tables in images.

        Args:
            images (List[Image.Image]): The list of images.
            result_path (str): The path to save the results.

        Returns:
            List[Dict]: The list of detection results. List[{"boxes": np.ndarray, "classes": np.ndarray,"scores": np.ndarray}]
        """

        results = []
        result_paths = []
        
        if self.layout_detector == 'yolo':
            for i, image in tqdm(enumerate(images), total=len(images), desc="Detecting tables"):
                result = self.detector.predict(image, Path(image).parent)
                results.append(result)
                result_paths.append(Path(image).parent)
        elif self.layout_detector == 'layoutlmv3':
            for i, image in tqdm(enumerate(images), total=len(images), desc="Detecting tables"):
                result = self.detector.predict(image, Path(image).parent)
                results.append(result)
                result_paths.append(Path(image).parent)
        else:
            raise ValueError(f"Invalid layout detector: {self.layout_detector}")
        
        
        return results,result_paths

    def crop_tables(self, 
                   image: Image.Image, 
                   boxes: np.ndarray, 
                   classes: np.ndarray,
                   save_dir: Optional[str] = None) -> List[Image.Image]:
        """
        Crop the table regions from the image.

        Args:
            image (Image.Image): The input image. 
            boxes (np.ndarray): The bounding box coordinates.
            classes (np.ndarray): The class labels.
            save_dir (str, optional): The directory to save the images.

        Returns:
            List[Image.Image]: The list of cropped table images.
        """
        # 检查并创建保存目录
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        table_regions = []
        table_image_path = []
        
        # 表格类别ID（根据实际情况调整）
        table_class_id = 5  # 假设5是表格的类别ID
        
        count = 0
        for box, cls in zip(boxes, classes):
            if cls == table_class_id:
                x1, y1, x2, y2 = map(int, box)
                table_region = image.crop((x1, y1, x2, y2))
                table_regions.append(table_region)
                count += 1
                
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    table_region.save(os.path.join(save_dir, f'table_{count}.jpg'))
                    table_image_path.append(os.path.join(save_dir, f'table_{count}.jpg'))
        return table_regions,table_image_path

    def extract_tables(self) -> List[Dict]:
        """
        The main function to extract tables from a PDF.

        Args:
            pdf_path (str): The path to the PDF file.
            output_dir (str): The directory to save the results.
            save_intermediate (bool): Whether to save the intermediate results.

        Returns:
            List[Dict]: The list of dictionaries containing the table information.
        """
        # 1. PDF转图像
        print("Converting PDF to images...")
        images,images_path = self.pdf_to_images(self.file_path,self.pdf_output_dir)
        
        # 2. 检测表格
        detection_results,detection_result_path = self.detect_tables(images_path)
        
        table_image_page_paths = []
        for i, (image, result,result_path) in tqdm(enumerate(zip(images, detection_results,detection_result_path)), total=len(images), desc="Cropping tables"):
            table_image_dir = os.path.join(result_path, 'Table_image')
            if not os.path.exists(table_image_dir):
                os.makedirs(table_image_dir)
            tables, table_image_path = self.crop_tables(
                image, 
                result[0]['boxes'], 
                result[0]['classes'],
                table_image_dir
            )
            table_image_page_paths.append(table_image_path)

        table_image_paths = [item for sublist in table_image_page_paths for item in sublist]
        table_file_paths = []
        
        for i, table_image_path in tqdm(enumerate(table_image_paths), total=len(table_image_paths), desc="Extracting tables"):
            table_df_path = os.path.join(Path(table_image_path).parent.parent,"Table_Df", f'{Path(table_image_path).stem}.tsv')
            self.table_to_dataframe(table_image_path,table_df_path)

    def table_to_dataframe(self,
                           file_path:str,
                           output_path:str):
        """
        Convert the table image to a DataFrame 

        Args:
            table_image (Image.Image): The table image.

        Returns:
            pd.DataFrame: The converted DataFrame.
        """
        aws_extract_tables(filename=file_path,
                           output_path=output_path)
        

# 使用示例
if __name__ == "__main__":
    extractor = EconTableExtractor(
        file_path="test.pdf"
    )
    extractor.extract_tables()