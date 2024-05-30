from PIL import Image

img_path = "C://Users//wangxiaodong//Downloads//arxiv-Final-3D-instance-20240527T134209Z-001//arxiv-Final-3D-instance//Two boys are sitting in a forest with the lake and mountains in the background_masked_1024_1024_16764650300.png"
img = Image.open(img_path)
img = img.resize((512, 512))
img.save("C://Users//wangxiaodong//Downloads//arxiv-Final-3D-instance-20240527T134209Z-001//arxiv-Final-3D-instance//boy2.jpg")