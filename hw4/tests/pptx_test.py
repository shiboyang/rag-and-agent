from src.ppt_loader import PPTXLoader


def test_pptx_loader():
    test_filepath = "/home/shiby/Desktop/人脸识别技术分享.pptx"
    ppt_img_dir = "/home/shiby/code/rag-and-agent/hw4/data/imgs"
    pptx_loader = PPTXLoader(test_filepath, ppt_img_dir)
    docs = pptx_loader.lazy_load()
    for doc in docs:
        page = doc.metadata["page"]
        img_count = doc.metadata["image_count"]
        content_length = len(doc.page_content)
        print(f"第{page}页ppt")
        print(f"内容长度：{content_length}")
        print(f"图片数量：{img_count}")
        print(f"内容预览：{doc.page_content[:50].replace("\n", "")}")


test_pptx_loader()
