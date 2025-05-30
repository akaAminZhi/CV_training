import easyocr

reader = easyocr.Reader(
    ["en"]
)  # this needs to run only once to load the model into memory
result = reader.readtext("page_0001.png")
print(result)
