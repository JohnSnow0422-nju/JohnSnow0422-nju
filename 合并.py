from PIL import Image

# 打开两张图片
img1 = Image.open('Perplexity.png')
img2 = Image.open('coherence.png')

# 获取两张图片的尺寸
width1, height1 = img1.size
width2, height2 = img2.size

# 新图片的宽度为两张图片的宽度之和，高度为较高的一张图片的高度
new_width = width1 + width2
new_height = max(height1, height2)

# 创建一张新的空白图片
new_img = Image.new('RGB', (new_width, new_height))

# 将第一张图片粘贴到新图片的左边
new_img.paste(img1, (0, 0))

# 将第二张图片粘贴到新图片的右边
new_img.paste(img2, (width1, 0))

# 保存新的合并后的图片
new_img.save('merged_image.png')
