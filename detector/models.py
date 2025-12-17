from django.db import models

class FaceImage(models.Model):
    image = models.ImageField(upload_to='uploads/')
    processed_image = models.ImageField(upload_to='processed/', blank=True, null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Image {self.id} uploaded at {self.uploaded_at}"
