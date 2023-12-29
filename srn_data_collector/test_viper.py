import torch

from srn_data_collector.viper_parser import get_page_dict

print(torch.cuda.is_available())
sample_file = "/cluster/home/repo/my_llm_experiments/864f83e0-41d4-4784-899e-1d1212c45371.pdf"
parser = get_page_dict(
    pdf_path=sample_file,
    model_path="/cluster/home/repo/my_llm_experiments/FRCNN_cbo_275.viper",
    img_path="/cluster/home/repo/my_llm_experiments/esrs_data_collection/srn_data_collector/pdf2img_images",
)
doc_dict = parser(pdf_path=sample_file, language="en")
print(doc_dict)
