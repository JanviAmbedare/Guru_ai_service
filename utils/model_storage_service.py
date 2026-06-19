import cloudinary
import cloudinary.uploader


class ModelStorageService:

    @staticmethod
    def upload_intent_model(
        model_path
    ):

        result = (
            cloudinary.uploader.upload(
                model_path,
                resource_type="raw",
                folder="guru/models/intent"
            )
        )

        return result["secure_url"]