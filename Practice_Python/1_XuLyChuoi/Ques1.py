"""
1. cắt từ và tạo dic
2. convert thành chữ thường và duyệt qua chuỗi
3. đếm số lần xuất hiện
4. sort dic lại theo tần suất và reverse lại
"""

#%% - xử lý dấu
from unidecode import unidecode

#%%
def word_frequency(input_str):

    # split từng từ
    words = input_str.split()
    word_count = {}

    # duyệt qua chuỗi để đếm tần suất
    for word in words:
        # chuyển từ về không dấu và chuyển đổi thành chữ thường
        cleaned_word = ''.join(char.lower() for char in unidecode(word) if char.isalpha())
        if cleaned_word:
            # nếu giống từ đã có trong word_count thì cộng 1 vào value của tuple đó
            word_count[cleaned_word] = word_count.get(cleaned_word, 0) + 1

    # sắp xếp lại word_count sử dụng value để sắp xếp
    sorted_word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

    for word, count in sorted_word_count:
        print(f'{word}: {count}')


#%%
input_str = input("Enter the string: ")
word_frequency(input_str)
