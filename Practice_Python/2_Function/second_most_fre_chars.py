#%%
def second_most_frequent_chars(input_str):
    # Chuyển đổi chuỗi thành chữ thường và loại bỏ khoảng trắng
    cleaned_str = input_str.lower().replace(" ", "")

    # Tạo một từ điển để lưu trữ tần suất xuất hiện của mỗi ký tự
    frequency = {}
    for char in cleaned_str:
        frequency[char] = frequency.get(char, 0) + 1

    # Sắp xếp từ điển theo giảm dần theo tần suất xuất hiện
    sorted_char_frequency = sorted(frequency.items(), key=lambda x: x[1], reverse=True)

    # Lọc ra các ký tự có tần suất xuất hiện nhiều thứ 2
    second_most_frequent_chars = [char for char, freq in sorted_char_frequency[1:] if
                                  freq == sorted_char_frequency[1][1]]

    return second_most_frequent_chars

#%% - Ví dụ sử dụng
input_string = input("Enter your string: ")
result = second_most_frequent_chars(input_string)
print("Second most frequent characters:", result)
