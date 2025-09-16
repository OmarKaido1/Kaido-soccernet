import sys
from PyQt5.QtWidgets import QApplication
import os

# Add the parent directory of 'src' to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from interface.main_window import MainWindow
if __name__ == "__main__":

	application = QApplication(sys.argv)
	window = MainWindow()
	sys.exit(application.exec_())