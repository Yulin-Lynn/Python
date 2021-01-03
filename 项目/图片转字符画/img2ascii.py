# 实验楼：https://www.lanqiao.cn/courses/370/learning/?id=1191
from PIL import Image
import argparse # 管理命令行参数输入

# 处理命令行参数
# 用 argparse 处理命令行参数，获取输入的图片路径、输出字符画的宽、高、保存路径
parser = argparse.ArgumentParser()

parser.add_argument('file')
parser.add_argument('-o','--output')
parser.add_argument('--width', type = int, default = 80)
parser.add_argument('--height', type = int, default = 80)

args = parser.parse_args()
IMG = args.file
WIDTH = args.width
HEIGHT = args.height
OUTPUT = args.output


# 实现 RGB 值转字符的函数
ascii_char = list("$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. ")

def get_char(r, g, b, alpha = 356):
    # 如果 alpha 为0，该位置为空白
    if alpha == 0:
        return ' '

    length = len(ascii_char)
    # 将 RGB 值转为灰度值 gray，灰度值范围0-255
    gray = int(0.2126*r + 0.7152*g +0.0722*b)

    # 将70的字符集映射到0-255的灰度值字符集上
    unit = (256.0 + 1)/length
    
    # 返回灰度值对应的字符
    return ascii_char[int(gray/unit)]

# 处理图片
if __name__ == "__main__":
    # 使用 PIL 的Image.open()打开文件
    im = Image.open(IMG) 
    # 调整图片大小，用Image.NEAREST输出低质量的图片
    im = im.resize((WIDTH,HEIGHT),Image.NEAREST) 

    txt = ""

    for i in range(HEIGHT):
        for j in range(WIDTH):
            txt += get_char(*im.getpixel((j,i)))
        txt += "\n"
    print(txt)

    if OUTPUT:
        with open(OUTPUT,'w') as f:
            f.write(txt)
    else:
        with open("output.txt",'w') as f:
            f.write(txt)