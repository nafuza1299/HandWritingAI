import PredictionResult as prediction
import Canvas as canvas
import Compare2Img as comparison

timesnewromans_index = ["TimesNewRomans/0.png", "TimesNewRomans/1.png", "TimesNewRomans/2.png", "TimesNewRomans/3.png",
                        "TimesNewRomans/4.png", "TimesNewRomans/5.png", "TimesNewRomans/6.png", "TimesNewRomans/7.png",
                        "TimesNewRomans/8.png", "TimesNewRomans/9.png"]

courier_index = ["Courier/0.png", "Courier/1.png", "Courier/2.png", "Courier/3.png",
                        "Courier/4.png", "Courier/5.png", "Courier/6.png", "Courier/7.png",
                        "Courier/8.png", "Courier/9.png"]

verdana_index = ["Verdana/0.png", "Verdana/1.png", "Verdana/2.png", "Verdana/3.png",
                        "Verdana/4.png", "Verdana/5.png", "Verdana/6.png", "Verdana/7.png",
                        "Verdana/8.png", "Verdana/9.png"]

arial_index = ["Arial/0.png", "Arial/1.png", "Arial/2.png", "Arial/3.png",
                        "Arial/4.png", "Arial/5.png", "Arial/6.png", "Arial/7.png",
                        "Arial/8.png", "Arial/9.png"]

comicsansserif_index = ["ComicSansSerif/0.png", "ComicSansSerif/1.png", "ComicSansSerif/2.png", "ComicSansSerif/3.png",
                        "ComicSansSerif/4.png", "ComicSansSerif/5.png", "ComicSansSerif/6.png", "ComicSansSerif/7.png",
                        "ComicSansSerifl/8.png", "ComicSansSerif/9.png"]
canvas.Paint()
img = "input.png"
imgpred = prediction.compare(img)
print("The prediction for the input image is the number",imgpred)
correct_number = int(input("Is this the correct number?\n1. Yes\n2. No"))
if correct_number == 2:
    imgpred = int(input("Enter the correct number:"))
compare = comparison.comparisonClass
font_selector = int(input("Select the font you want to compare\n1. Times New Roman\n2. Courier\n3. Arial\n4. Verdana\n5. Comic Sans Serif"))
if font_selector == 1:
    compare.inputimage(img, timesnewromans_index[imgpred])
if font_selector == 2:
    compare.inputimage(img, courier_index[imgpred])
if font_selector == 3:
    compare.inputimage(img, verdana_index[imgpred])
if font_selector == 4:
    compare.inputimage(img, arial_index[imgpred])
if font_selector == 5:
    compare.inputimage(img, comicsansserif_index[imgpred])