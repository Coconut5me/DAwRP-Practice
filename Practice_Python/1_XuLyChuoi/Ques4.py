def check_contain_all(str):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'

    # convert các ký tự thành chữ thường và so sánh với alphabet
    for char in alphabet:
        # nếu có ký tự trong alphabet không có trong str thì trả về false
        if char.lower() not in str.lower():
            return False
    return True


str = input("Enter the string: ")
result = check_contain_all(str)
print("Ressult: ", result)
