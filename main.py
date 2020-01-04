import predict_2 as pred2
import Scribble as canvas
import Compare2Img as comp1
canvas.Paint()

img = "file.png"
img_0 = "0.png"
img_1 = "1.png"
img_2 = "2.png"
img_3 = "3.png"
img_4 = "4.png"
img_5 = "5.png"
img_6 = "6.png"
img_7 = "7.png"
img_8 = "8.png"
img_9 = "9.png"
img_index = [img_0, img_1, img_2, img_3, img_4, img_5, img_6, img_7, img_8, img_9]

imgpred = pred2.compare(img)
print(imgpred)
compare = comp1.comparisonClass
compare.inputimage(img, img_index[imgpred])
