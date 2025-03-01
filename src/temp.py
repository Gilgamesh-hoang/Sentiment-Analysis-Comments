from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
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


if __name__ == "__main__":
    input_xlsx = "E:\\Sentiment-Analysis-Comments\\Dataset\\raw\\merged_dataset.xlsx"

# Other: 2300
# Disgust: 1338
# Enjoyment: 1965
# Anger: 1666
# Surprise: 886
# Sadness: 1687
# Fear: 794
# Happy: 1923
