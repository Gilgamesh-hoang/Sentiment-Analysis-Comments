import pickle
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from numpy import number

from src.preprocess import preprocess_fn


def count(file_path):
    # Đọc file dataset.txt
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Tạo danh sách chứa tất cả các label
    labels = [line.strip().split("\t")[-1] for line in lines if "\t" in line]

    # Đếm số lượng từng label
    label_counts = Counter(labels)

    # In kết quả
    print("Số lượng nhãn khác nhau:", len(label_counts))
    print("Chi tiết số lượng mỗi nhãn:")
    for label, count in label_counts.items():
        print(f"{label}: {count}")


def xlsx():
    # Danh sách các file cần gộp
    file_list = ["E:\\Sentiment-Analysis-Comments\\Dataset\\raw\\train_nor.xlsx",
                 "E:\\Sentiment-Analysis-Comments\\Dataset\\raw\\test_nor.xlsx",
                 "E:\\Sentiment-Analysis-Comments\\Dataset\\raw\\valid_nor.xlsx"]

    dataframes = []

    for file in file_list:
        # Đọc file, bỏ qua dòng header đầu tiên
        df = pd.read_excel(file)

        # Giữ lại chỉ 2 cột 'Emotion' và 'Sentence'
        df = df[["Emotion", "Sentence"]]

        dataframes.append(df)

    # Gộp tất cả các file lại thành một DataFrame
    df_merged = pd.concat(dataframes, ignore_index=True)

    # Xuất ra file mới
    df_merged.to_excel("E:\\Sentiment-Analysis-Comments\\Dataset\\raw\\merged_dataset.xlsx", index=False)

    print("Gộp file thành công! Dữ liệu được lưu trong 'merged_dataset.xlsx'")


def convert_xlsx_to_txt(input_file, output_file):
    # Đọc file Excel
    df = pd.read_excel(input_file)

    # Chuẩn hóa tên cột để tránh lỗi
    df.columns = df.columns.str.strip().str.lower()

    # Đảm bảo cột đúng định dạng
    if "emotion" not in df.columns or "sentence" not in df.columns:
        print("LỖI: File không có đủ cột 'Emotion' và 'Sentence'")
        return

    # Đổi thứ tự cột thành 'Sentence' trước, 'Emotion' sau
    df = df[["sentence", "emotion"]]

    # Xuất dữ liệu ra file TXT với tab phân cách
    df.to_csv(output_file, sep='\t', index=False, header=False, encoding='utf-8')

    print(f"Chuyển đổi thành công! Dữ liệu được lưu trong '{output_file}'")


def plot_label_distribution(txt_file):
    # Đọc file TXT với tab phân cách
    df = pd.read_csv(txt_file, sep='\t', names=["sentence", "emotion"], encoding='utf-8')

    # Vẽ biểu đồ số lượng mỗi label
    plt.figure(figsize=(10, 5))
    df["emotion"].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
    plt.xlabel("Label")
    plt.ylabel("Số lượng")
    plt.title("Phân bố số lượng của mỗi label trong dataset (TXT)")
    plt.xticks(rotation=45)
    plt.show()


def reduce_other_label(input_file, output_file, target_other_count=2300):
    # Đọc file TXT (tách bằng tab)
    df = pd.read_csv(input_file, sep='\t', names=["sentence", "emotion"], encoding='utf-8')

    # Lọc nhãn 'Other'
    df_other = df[df["emotion"] == "Other"]
    df_other_reduced = df_other.sample(n=target_other_count, random_state=42)  # Chọn ngẫu nhiên 2300 dòng

    # Lọc các nhãn khác
    df_other_removed = df[df["emotion"] != "Other"]

    # Gộp lại
    df_final = pd.concat([df_other_reduced, df_other_removed], ignore_index=True)

    # Lưu file mới
    df_final.to_csv(output_file, sep='\t', index=False, header=False, encoding='utf-8')

    print(f"✅ Đã giảm nhãn 'Other' xuống {target_other_count} và lưu vào {output_file}")


def process_dataset(input_file: str, output_file: str):
    """
    Read a dataset file, preprocess each line, and save the processed text to a new file.

    Args:
        input_file (str): Path to the input dataset file.
        output_file (str): Path to save the processed dataset.
    """
    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            parts = line.strip().split("\t")  # Giả sử dữ liệu phân tách bằng tab (\t)
            if len(parts) != 2:
                print(f"Dòng không hợp lệ: {line}")
                continue  # Bỏ qua dòng không hợp lệ

            text, label = parts
            processed_text = preprocess_fn(text)  # Gọi hàm tiền xử lý

            outfile.write(f"{processed_text}\t{label}\n")  # Ghi vào file mới


def process_dataset2(input_file: str, output_file: str):
    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            parts = line.strip().split("\t")  # Giả sử dữ liệu phân tách bằng tab (\t)
            if len(parts) != 2:
                print(f"Dòng không hợp lệ: {line}")
                continue  # Bỏ qua dòng không hợp lệ

            text, label = parts
            if len(text) < 2:
                continue

            outfile.write(f"{text}\t{label}\n")  # Ghi vào file mới


def a(input_file: str):
    with open(input_file, "r", encoding="utf-8") as infile:
        # get line and index
        lines = infile.readlines()
        for idx, line in enumerate(lines):
            parts = line.strip().split("\t")
            if len(parts) != 2:
                print(f"Line {idx + 1} is invalid: {line}")
                continue
            text, label = parts
            if len(text) <= 3:
                print(f"Line {idx + 1} is invalid: {line}")
                continue


def remove_lines(input_file: str, lines_num: list[int], output_file: str):
    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        # get line and index
        lines = infile.readlines()
        for idx, line in enumerate(lines):
            parts = line.strip().split("\t")
            if idx + 1 in lines_num:
                continue
            text, label = parts
            outfile.write(f"{text}\t{label}\n")


def remove_duplicate_lines(input_file: str, output_file: str):
    """
    Read a text file, remove duplicate lines (keeping only the first occurrence), and save the result.

    Args:
        input_file (str): Path to the input text file.
        output_file (str): Path to save the processed file.
    """
    unique_lines = set()  # Dùng set để lưu các dòng đã xuất hiện
    filtered_lines = []  # Danh sách lưu dòng đã lọc

    with open(input_file, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:  # Bỏ qua dòng trống
                continue

            parts = line.split("\t")  # Tách phần nội dung và nhãn
            if len(parts) != 2:  # Bỏ qua dòng không hợp lệ
                print(f"Skipping invalid line: {line}")
                continue

            text, label = parts
            if text not in unique_lines:  # Nếu chưa có nội dung này thì thêm vào
                unique_lines.add(text)
                filtered_lines.append(f"{text}\t{label}")

    # Ghi lại file với các dòng đã được lọc trùng
    with open(output_file, "w", encoding="utf-8") as outfile:
        outfile.write("\n".join(filtered_lines))


# Gọi hàm với file input và output mong muốn

if __name__ == "__main__":
    #
    # count("E:\\Sentiment-Analysis-Comments\\Dataset\\processed_dataset_final.txt")
    with open("/Dataset/dataset_embeddings.pkl", "rb") as f:
        labels, vectors = pickle.load(f)
        print(len(labels))
        print(len(vectors))
    with open("E:\\Sentiment-Analysis-Comments\\Dataset\\vectors2.pkl", "rb") as f:
        labels, vectors = pickle.load(f)
        print(len(labels))
        print(len(vectors))

# Other: 2214
# Disgust: 1333
# Enjoyment: 1950
# Anger: 1605
# Surprise: 850
# Sadness: 1636
# Fear: 734
# Happy: 1782
