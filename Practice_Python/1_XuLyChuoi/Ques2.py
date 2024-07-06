"""
1. convert thành chữ thường
2. so sánh 2 chuỗi
"""

def similar_chars_check(word1, word2):
    # convert các ký tự trong 2 chuỗi thành chữ thươờng và các ký tự trong mỗi chuỗi theo thứ tự tăng dần
    sorted_word1 = ''.join(sorted(word1.lower()))
    sorted_word2 = ''.join(sorted(word2.lower()))

    # so sánh 2 chuỗi
    return sorted_word1 == sorted_word2

str1 = input("Enter the first word: ")
str2 = input("Enter the second word: ")
result = similar_chars_check(str1, str2)
print("Result: ", result)
