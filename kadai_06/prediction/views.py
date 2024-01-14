from django.shortcuts import render
from .forms import ImageUploadForm
#import random
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from io import BytesIO
import os
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions


def predict(request):
    if request.method == "GET" :
        # GETリクエストによるアクセス時の処理を記述
        form = ImageUploadForm()
        return render(request, "home.html", {"form" : form})
    
    if request.method == "POST" :
        # POSTリクエストによるアクセス時の処理を記述
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
                #img_file = form.cleaned_data["image"]
                #img_file = BytesIO(img_file.read())
                img_file = BytesIO(request.FILES['image'].read())  # BytesIOを使って画像を読み込む 
                img = load_img(img_file, target_size=(224, 224))

                # VGG16モデルをロードする
                vgg16_model = VGG16(weights='imagenet')

                # 画像を前処理する
                #img = load_img(img_file, target_size=(224, 224))
                img_array = img_to_array(img)
                img_array = img_array.reshape((1, 224, 224, 3))
                img_array = preprocess_input(img_array)

                # 予測を行う
                predictions = vgg16_model.predict(img_array)

               # 予測結果をデコードして上位5つを表示する
                decoded_predictions = decode_predictions(predictions, top=5)[0]
                top_predictions = [(label, round(score * 100, 2)) for (_, label, score) in decoded_predictions]

                # 予測結果をアップロードされた画像と一緒に表示する
                img_data = request.POST.get("img_data")  # img_dataを取得する
                return render(request, "home.html", {"form": form, "predictions": top_predictions, "img_data": img_data})

        #     img_file = form.cleaned_data["image"]
        #     # 4章で画像ファイル(img_file)の前処理を追加
        #     img_file = BytesIO(img_file.read()) 
        #     img = load_img(img_file, target_size=(256,256))
        #     img_array = img_to_array(img)
        #     img_array = img_array.reshape((1,256,256,3))
        #     img_array = img_array / 255
           
        #     # 4章で判定結果のロジックを追加
        #     model_path = os.path.join(settings.BASE_DIR,"prediction", "models", "model.h5")
        #     model = load_model(model_path)
        #     result = model.predict(img_array)
        #     print(result)
        #     if result[0][0] > result[0][1]:
        #         prediction = "猫"
        #     else:
        #         prediction = "犬"
            
        #     # 暫定でダミー判定結果としてpredictionにランダムで「猫」「犬」を格納
        #     #prediction = random.choice(["猫", "犬"])
        #     img_data = request.POST.get("img_data")
        #     return render(request, "home.html", {"form":form, "prediction":prediction, "img_data":img_data})
        # else:
        #     form = ImageUploadForm()
        #     return render(request, "home.html", {"form":form})
        