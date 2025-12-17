from django import forms
from .models import FaceImage

class UploadImageForm(forms.ModelForm):
    class Meta:
        model = FaceImage
        fields = ['image']
