def saveTxt(content, path, mode):
    #f = open(path, mode, encoding="utf-8")
    #f.write(content)
    #f.close()
    with open(path. mode, encoding="utf-8") as f:
        f.write(content)

def readTxt(path):
    f = open(path, "r", encoding="utf-8")
    s = []
    for line in f:
        s.append(line.strip())
    return s