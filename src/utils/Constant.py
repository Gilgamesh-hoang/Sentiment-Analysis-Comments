import os
import sys

# Thêm thư mục gốc vào sys.path để import module dễ dàng
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Lấy thư mục gốc của project (cha của src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print('BASE_DIR:', BASE_DIR)
TEEN_CODE_PATH = os.path.join(BASE_DIR, 'Nomalize', 'teencode.txt')
STOPWORD_PATH = os.path.join(BASE_DIR, 'Nomalize', 'vietnamese-stopwords.txt')