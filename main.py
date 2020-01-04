import predict_2 as pred2
import Scribble as canvas
import Compare2Img as comp1
canvas.Paint()

img = "pls.png"
img_3 = "3img3.png"
img_4 = "4.jpg"
imgpred = pred2.compare(img)
print(imgpred)
compare = comp1.comparisonClass

if imgpred == 3:
    compare.inputimage(img, img_3)
if imgpred == 4:
    compare.inputimage(img, img_4)
