# from paddleocr import PPStructureV3

# pipeline = PPStructureV3(use_doc_orientation_classify=False, use_doc_unwarping=False)

# # For Image
# output = pipeline.predict(
#     input="C:/Users/zhimin qin/OneDrive - M.C. DEAN, INC/Pictures/IAD06_Mechanical/1.png",
# )

# # 可视化结果并保存 json 结果
# for res in output:
#     print(res["overall_ocr_res"]["rec_texts"])
from paddleocr import PPStructureV3
import os

os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin")
# 初始化 PaddleOCR 实例
ocr = PPStructureV3(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=True,
    text_detection_model_dir=r"C:\Users\zhimin qin\.paddlex\official_models\PP-OCRv5_server_det",
    text_recognition_model_dir=r"C:\Users\zhimin qin\.paddlex\official_models\PP-OCRv5_server_rec",
    textline_orientation_model_dir=r"C:\Users\zhimin qin\.paddlex\official_models\PP-LCNet_x1_0_textline_ori",
    wireless_table_cells_detection_model_dir=r"C:\Users\zhimin qin\.paddlex\official_models\RT-DETR-L_wireless_table_cell_det",
    chart_recognition_model_dir=r"C:\Users\zhimin qin\.paddlex\official_models\PP-Chart2Table",
    formula_recognition_model_dir=r"C:\Users\zhimin qin\.paddlex\official_models\PP-FormulaNet_plus-L",
    layout_detection_model_dir=r"C:\Users\zhimin qin\.paddlex\official_models\PP-DocLayout_plus-L",
)

# 对示例图像执行 OCR 推理
result = ocr.predict(
    input="C:/Users/zhimin qin/OneDrive - M.C. DEAN, INC/Pictures/LSB_Panel/table3.png"
)

# 可视化结果并保存 json 结果
for res in result:
    res.save_to_json(save_path="output")
    # res.save_to_img("output")
