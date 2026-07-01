f = open("hallucination_preview.txt", "r")

notStarved = 0
starved = 0
for line in f:
    for ch in line[:3]:
        if ch.isalpha():
            if ch.lower() in "abcdefghijklm":
                notStarved += 1
            else:
                starved += 1

print("Well-trained = ", notStarved)
print("Non trained = ", starved )
f.close()