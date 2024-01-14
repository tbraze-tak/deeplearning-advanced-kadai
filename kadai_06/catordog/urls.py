from django.urls import path
from prediction.views import predict

urlpatterns = [
    #path("admin/", admin.site.urls),
    #path("", predict, name="predict" ),
    path('', predict, name='predict'),
]
