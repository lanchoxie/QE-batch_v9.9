import sys
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QSizePolicy, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

import argparse

# 封装提取尺寸信息的逻辑到一个函数中
def extract_size_from_filename(filename):
    try:
        parts = filename.split(".")[0].split("_")[-2:]
        x_size, y_size = int(parts[0]), int(parts[1])
        return x_size, y_size
    except (IndexError, ValueError):
        raise ValueError("Filename must contain size information in the format 'name_xSize_ySize.ext'")

# 创建一个解析器并添加参数
parser = argparse.ArgumentParser(description="Xming matplotlib shower")
parser.add_argument('fig_in', type=str, help='figure name')
parser.add_argument('--size_x','-x', type=int, help='x size')
parser.add_argument('--size_y','-y', type=int, help='y size')

args = parser.parse_args()

fig_in=args.fig_in.split("/")[-1]
dir_in=args.fig_in.split(fig_in)[0] if "/" in args.fig_in else "./"
# 如果用户没有指定尺寸，那么从文件名中提取
if args.size_x is None or args.size_y is None:
    try:
        args.size_x, args.size_y = extract_size_from_filename(fig_in)
    except ValueError as e:
        parser.error(str(e))


size_x=args.size_x
size_y=args.size_y

# 现在您可以使用 args.fig_in, args.size_x, 和 args.size_y
print(f"Figure Name: {fig_in}, Size: {args.size_x}x{args.size_y}")


class ScatterPlotApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(100, 100, size_x*100, size_y*100)
        # 设置窗口标题
        self.setWindowTitle("Scatter Plot Viewer")

        # 创建一个主窗口部件
        main_widget = QWidget(self)
        self.setCentralWidget(main_widget)

        # 创建一个垂直布局
        layout = QVBoxLayout(main_widget)

        # 创建一个标签用于显示图像
        self.image_label = QLabel(self)
        layout.addWidget(self.image_label)

        # 调整图像标签的大小策略，以自适应图像大小
        #self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        #self.image_label.setScaledContents(True)

        # 加载并显示图像
        self.load_image(f"{dir_in}{fig_in}")

    def load_image(self, image_path):
        #将图像等比放大100倍
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(size_x * 100, size_y * 100, aspectRatioMode=Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)

def main():
    windows = []
    app = QApplication(sys.argv)
    window = ScatterPlotApp()
    window.show()
    windows.append(window)
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

