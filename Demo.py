import streamlit as st
import os
from PIL import Image
import pandas as pd
from pathlib import Path
import sys
import importlib

# 防止重复导入造成的注册错误
if 'EconTableExtractor' in sys.modules:
    importlib.reload(sys.modules['EconTableExtractor'])
from EconTableExtractor import EconTableExtractor

class PDFTableExtractorApp:
    def __init__(self):
        st.set_page_config(layout="wide")
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'upload'
        if 'pdf_path' not in st.session_state:
            st.session_state.pdf_path = None
        if 'current_pdf_page' not in st.session_state:
            st.session_state.current_pdf_page = 0
        if 'extractor' not in st.session_state:
            st.session_state.extractor = None
            
    def upload_page(self):
        st.title("EconTableExtractor")
        uploaded_file = st.file_uploader("Upload PDF file", type=['pdf'])
        
        if uploaded_file is not None:
            # 保存上传的文件    
            if st.button("Start processing"):
                try:
                    if 'extractor' not in st.session_state or st.session_state.extractor is None:
                        st.session_state.pdf_path = uploaded_file.name
                        try:
                            st.session_state.extractor = EconTableExtractor(file_path=st.session_state.pdf_path)
                        except Exception as e:
                            st.error(f"Error initializing extractor: {str(e)}")
                            return
                    with st.spinner("Processing PDF into images..."):
                        # 转换PDF到图像
                        images, image_paths = st.session_state.extractor.pdf_to_images(
                            st.session_state.pdf_path, 
                            st.session_state.extractor.pdf_output_dir
                        )
                        # 检测表格
                    with st.spinner("Detecting tables..."):
                        detection_results, detection_paths = st.session_state.extractor.detect_tables(image_paths)
                        
                    with st.spinner("Cropping tables..."):
                        table_image_page_paths = []
                        for i, (image, result,result_path) in enumerate(zip(images, detection_results,detection_paths)):
                            table_image_dir = os.path.join(result_path, 'Table_image')
                            if not os.path.exists(table_image_dir):
                                os.makedirs(table_image_dir)
                            tables, table_image_path = st.session_state.extractor.crop_tables(
                                image, 
                                result[0]['boxes'], 
                                result[0]['classes'],
                                table_image_dir
                            )
                            table_image_page_paths.append(table_image_path)

                    new_image_paths = []
                    ## set the st.state
                    for image, table_image in zip(image_paths, table_image_page_paths):
                        if table_image != []:
                            new_image_paths.append(image)
                    
                    table_image_page_paths = [table_image for table_image in table_image_page_paths if table_image != []]

                    st.session_state.image_paths = new_image_paths
                    st.session_state.detection_results = detection_results
                    st.session_state.detection_paths = detection_paths
                    st.session_state.table_image_paths = table_image_page_paths
                    st.session_state.current_page = 'review'
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")

    def review_page(self):
        st.title("Review extracted tables")
        
        if not hasattr(st.session_state, 'image_paths'):
            st.error("Please upload and process the PDF file first")
            return
        
        # 显示页码和导航按钮
        col1, col2, col3 = st.columns([1,3,1])
        with col1:
            if st.button("Previous page", disabled=st.session_state.current_pdf_page == 0):
                st.session_state.current_pdf_page -= 1
                st.rerun()
        with col2:
            st.write(f"Page {st.session_state.current_pdf_page + 1} of {len(st.session_state.image_paths)}")
        with col3:
            if st.button("Next page", disabled=st.session_state.current_pdf_page == len(st.session_state.image_paths)-1):
                st.session_state.current_pdf_page += 1
                st.rerun()

        # 显示原始页面和提取的表格
        col_pdf, col_tables = st.columns([1,1])
        
        with col_pdf:
            st.subheader("Original PDF page")
            original_image = Image.open(st.session_state.image_paths[st.session_state.current_pdf_page])
            st.image(original_image, use_container_width=True)

        with col_tables:
            st.subheader("Extracted tables")
            current_page_table_paths = st.session_state.table_image_paths[st.session_state.current_pdf_page]
            for table_path in current_page_table_paths:
                table_image = Image.open(table_path)
                st.image(table_image, use_container_width=True)
            if len(current_page_table_paths) > 1:
                if st.button("Combine all tables"):
                    self.combine_images(current_page_table_paths)
        
        # 添加底部导航按钮
        st.markdown("---")  # 添加分隔线
        nav_col1, nav_col2 = st.columns(2)
        
        with nav_col1:
            if st.button("⬅️ Back to upload page"):
                st.session_state.current_page = 'upload'
                st.rerun()
        
        with nav_col2:
            if st.button("Go to digitization page ➡️"):
                if len(current_page_table_paths) > 0:
                    if 'table_df_list' not in st.session_state:
                        st.session_state.table_df_list = [[] for _ in range(len(st.session_state.table_image_paths))]
                    st.session_state.current_page = 'digitize'
                    st.session_state.current_pdf_page = 0
                    st.rerun()
                else:
                    st.error("No tables to digitize on the current page")

    def digitize_page(self):
        st.title("Digitizing tables")
        
        col1, col2, col3 = st.columns([1,3,1])
        with col1:
            if st.button("Previous page", disabled=st.session_state.current_pdf_page == 0):
                # 清除当前的DataFrame
                if 'current_df' in st.session_state:
                    del st.session_state.current_df
                    del st.session_state.current_df_path
                st.session_state.current_pdf_page -= 1
                st.rerun()
        with col2:
            st.write(f"Page {st.session_state.current_pdf_page + 1} of {len(st.session_state.image_paths)}")
        with col3:
            if st.button("Next page", disabled=st.session_state.current_pdf_page == len(st.session_state.image_paths)-1):
                # 清除当前的DataFrame
                if 'current_df' in st.session_state:
                    del st.session_state.current_df
                    del st.session_state.current_df_path
                st.session_state.current_pdf_page += 1
                st.rerun()

        col_image, col_data = st.columns([1,1])
        
        with col_image:
            st.subheader("Table images")
            for table_image_path in st.session_state.table_image_paths[st.session_state.current_pdf_page]:
                table_image = Image.open(table_image_path)
                st.image(table_image, use_container_width=True)
                if st.button(f"Digitize {Path(table_image_path).stem}"):
                    with st.spinner("Processing..."):
                        output_path = Path(table_image_path).parent.parent / "Table_Df" / f"{Path(table_image_path).stem}.tsv"
                        st.session_state.extractor.table_to_dataframe(
                            table_image_path,
                            str(output_path)
                        )
                        if output_path.exists():
                            st.session_state.table_df_list[st.session_state.current_pdf_page].append(output_path)
                            st.rerun()

        
        with col_data:
            st.subheader("Extracted data")
            for table_df_path in st.session_state.table_df_list[st.session_state.current_pdf_page]:
                if table_df_path != []:
                    df = pd.read_csv(table_df_path, sep='\t')
                    # 显示可编辑的数据表格
                    edited_df = st.data_editor(df)
                    
                    # 添加保存按钮
                    if st.button("Save", key=f"save_{table_df_path}"):
                        edited_df.to_csv(table_df_path, sep='\t', index=False)
                        st.success("Data saved")
        
        if st.button("Back to review page"):
            st.session_state.current_page = 'review'
            st.rerun()

    def combine_images(self, current_page_table_paths):
        """
        将多张表格图片垂直合并，第一张在上，第二张在下
        """
        if not current_page_table_paths or len(current_page_table_paths) < 2:
            st.warning("Need at least two images to combine")
            return
        
        # 打开图片
        images = [Image.open(path) for path in current_page_table_paths]
        
        # 计算合并后图片的总宽度和高度
        max_width = max(img.width for img in images)
        total_height = sum(img.height for img in images)
        
        # 创建新的空白图片
        combined_image = Image.new('RGB', (max_width, total_height), 'white')
        
        # 垂直拼接图片
        current_height = 0
        for img in images:
            # 如果图片宽度小于最大宽度，将其居中
            if img.width < max_width:
                x_offset = (max_width - img.width) // 2
            else:
                x_offset = 0
                
            combined_image.paste(img, (x_offset, current_height))
            current_height += img.height
        
        # 删除原始图片
        for path in current_page_table_paths:
            if os.path.exists(path):
                os.remove(path)
                
        # 保存合并后的图片
        
        combined_image.save(current_page_table_paths[0])

        
        # 更新session state中的当前表格路径
        st.session_state.table_image_paths[st.session_state.current_pdf_page] = [current_page_table_paths[0]]
        st.rerun()
        
            

    def run(self):
        if st.session_state.current_page == 'upload':
            self.upload_page()
        elif st.session_state.current_page == 'review':
            self.review_page()
        elif st.session_state.current_page == 'digitize':
            self.digitize_page()

if __name__ == "__main__":

    app = PDFTableExtractorApp()
    app.run()

