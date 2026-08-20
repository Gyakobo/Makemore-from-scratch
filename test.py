words = open("names.txt", "r").read().splitlines()


"""
Here basically you're storing the information in a dictionary,
however, it's a bit inconvenient as it's way better to store
such information in a `tensor` 2D array specifically
"""
b = {}
for w in words:

    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1

# sorted(b.items(), key=lambda kv: -kv[1])
# print(f"{b=}")

"""
Here is a rendition of the same approach only in a 2D tensor array
"""
