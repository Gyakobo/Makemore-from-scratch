words = open("names.txt", "r").read().splitlines()

for w in words[:2]:

    chs = ["."] + list(w) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        print(ch1, ch2)
