def decrypt(string):
    arr = string.split(" ")
    len1 = len(arr)
    lst = []
    for i in range(len1-1):
        tmplist = arr[i].split('$')
        lst.append(tmplist[1])
    return lst